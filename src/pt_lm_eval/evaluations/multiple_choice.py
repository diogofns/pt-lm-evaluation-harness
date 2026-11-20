from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Sequence

from ..core import (
    GenerateRequest,
    GenerateResponse,
    LoglikelihoodRequest,
    LoglikelihoodResponse,
    MCQItem,
    Scored,
)
from . import prompts
from .base import Evaluation

# "Resposta: X", normalizado (maiúsculas, markdown, parênteses, pontuação). A letra
# maiúscula não pode iniciar palavra ("Resposta: Brasília"); minúscula só entre
# parênteses ou no fim da linha, para não confundir o artigo "a".
_FORMAT = re.compile(
    r"(?i:resposta)\**\s*:\s*\**\s*"
    r"(?:\(([A-Ja-j])\)|([A-J])(?![A-Za-zÀ-ÿ])|([a-j])(?=\s*[.)]?\**\s*$))",
    re.M,
)
# A resposta inteira é a letra, ou começa com "B)"/"B.": completa o "Resposta:" do prompt.
_ONLY_LETTER = re.compile(r"^\s*\**\(?([A-J])\)?[.):]?\**\s*$")
_LEADING_LETTER = re.compile(r"^\s*\**\(?([A-J])[).:]\**\s")


def extract_letter(text: str, letters: str) -> str | None:
    """Alternativa no formato pedido (a última, se o modelo se corrigir); None se não houver."""
    matches = _FORMAT.findall(text)
    if matches:
        letter: str | None = next(g for g in matches[-1] if g).upper()
    else:
        match = _ONLY_LETTER.search(text) or _LEADING_LETTER.search(text)
        letter = match.group(1) if match else None
    return letter if letter is not None and letter in letters else None


def argmax(values: Sequence[float]) -> int:
    """Índice do maior valor; em empate, o primeiro (como numpy.argmax)."""
    return max(range(len(values)), key=values.__getitem__)


class LoglikelihoodMCQ(Evaluation):
    """Pontua cada opção por log P(opção | prompt); a de maior valor é a escolhida."""

    item_type = MCQItem
    needs = frozenset({"loglikelihood"})

    @abstractmethod
    def render(self, item: MCQItem, with_answer: bool) -> str: ...

    @abstractmethod
    def targets(self, item: MCQItem) -> Sequence[str]: ...

    def choices(self, item: MCQItem) -> list[str]:
        preserve = item.protocol == "global_piqa"
        return [prompts.TARGET_DELIMITER + (t if preserve else t.strip()) for t in self.targets(item)]

    def build(
        self, item: MCQItem, shots: Sequence[MCQItem], multiturn: bool = False
    ) -> list[LoglikelihoodRequest]:
        context = prompts.with_shots([self.render(s, True) for s in shots], self.render(item, False))
        context = prompts.description(item) + context
        return [LoglikelihoodRequest(context, c) for c in self.choices(item)]

    def metrics(self, item: MCQItem, responses: Sequence[LoglikelihoodResponse]) -> dict[str, float]:
        acc = float(argmax([r.logprob for r in responses]) == item.gold)
        return {"acc": acc, **({"acc_norm": acc} if item.protocol == "mmmlu" else {})}

    def score(self, item: MCQItem, responses: Sequence[LoglikelihoodResponse]) -> Scored:
        lls = [r.logprob for r in responses]
        return Scored(
            metrics={**self.metrics(item, responses), "random_baseline": 1 / len(item.options)},
            details={
                "gold": item.letters[item.gold],
                "pred": item.letters[argmax(lls)],
                "logprobs": [round(ll, 6) for ll in lls],
                "n_tokens": [r.n_tokens for r in responses],
            },
        )


class MultipleChoiceFormulation(LoglikelihoodMCQ):
    """MCF: as opções aparecem no prompt e o modelo escolhe a letra."""

    name = "mcf"

    def render(self, item: MCQItem, with_answer: bool) -> str:
        return prompts.mcf(item, with_answer=with_answer)

    def targets(self, item: MCQItem) -> Sequence[str]:
        return item.letters


class ClozeFormulation(LoglikelihoodMCQ):
    """CF: as opções não aparecem; o modelo pontua o texto de cada uma."""

    name = "cf"

    def render(self, item: MCQItem, with_answer: bool) -> str:
        return prompts.cf(item, with_answer=with_answer)

    def targets(self, item: MCQItem) -> Sequence[str]:
        return item.options

    def metrics(self, item: MCQItem, responses: Sequence[LoglikelihoodResponse]) -> dict[str, float]:
        lls = [r.logprob for r in responses]
        texts = (
            list(item.options) if item.protocol == "global_piqa" else [c.strip() for c in self.choices(item)]
        )
        char = [ll / max(1, len(t)) for ll, t in zip(lls, texts, strict=True)]
        token = [ll / max(1, r.n_tokens) for ll, r in zip(lls, responses, strict=True)]
        metrics = {
            **super().metrics(item, responses),
            "acc_char": float(argmax(char) == item.gold),
            "acc_token": float(argmax(token) == item.gold),
        }
        if item.protocol == "global_piqa":
            byte = [ll / max(1, len(t.encode("utf-8"))) for ll, t in zip(lls, texts, strict=True)]
            metrics["acc_bytes"] = float(argmax(byte) == item.gold)
        if item.protocol == "mmmlu":
            metrics.pop("acc_norm", None)  # no cloze de texto equivalente no protocolo oficial
        return metrics


class GenerativeMCQ(Evaluation):
    """gen: o modelo escreve a resposta e a letra é extraída do formato pedido."""

    name = "gen"
    item_type = MCQItem
    needs = frozenset({"generate"})
    supports_multiturn = True
    scoring_version = "3"

    def build(
        self, item: MCQItem, shots: Sequence[MCQItem], multiturn: bool = False
    ) -> list[GenerateRequest]:
        if item.protocol == "global_piqa":
            return [
                GenerateRequest(
                    prompt=prompts.gen(item, shots),
                    max_tokens=2048,
                    params={"temperature": 0.8, "top_p": 0.95},
                    enforce_generation_config=True,
                )
            ]
        if multiturn and shots:
            messages = prompts.gen_messages(item, shots)
            prompt = prompts.FEWSHOT_DELIMITER.join(f"[{m['role']}] {m['content']}" for m in messages)
            return [GenerateRequest(prompt=prompt, messages=messages)]
        return [GenerateRequest(prompt=prompts.gen(item, shots), stop=prompts.GEN_STOP)]

    def score(self, item: MCQItem, responses: Sequence[GenerateResponse]) -> Scored:
        response = responses[0]
        gold, pred = (
            item.letters[item.gold],
            extract_letter(response.text, item.letters),
        )
        return Scored(
            metrics={
                "acc": float(pred == gold),
                **({"exact_match": float(pred == gold)} if item.protocol == "global_piqa" else {}),
                "extracted": float(pred is not None),
                "truncated": float(response.finish_reason == "length"),
                "random_baseline": 1 / len(item.options),
            },
            details={
                "gold": gold,
                "pred": pred,
                **response.details(),
            },
        )
