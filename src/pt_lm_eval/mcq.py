"""Múltipla escolha: prompts em português, log-likelihood (mcf, cf) e geração (gen).

- mcf: opções no prompt; pontua " A", " B", … após "Resposta:". Métrica: acc.
- cf:  opções fora do prompt; pontua " <texto da opção>". Métricas: acc, acc_char
       (normalizado por caracteres, o acc_norm do lm-eval) e acc_token (por tokens).
- gen: o modelo escreve e a letra é extraída do formato pedido ("Resposta: X"),
       normalizado, como no guia do OpenEvals. Métricas: acc, extracted (seguiu o
       formato) e truncated (parou por limite de tokens).

Todas registram random_baseline (1/nº de opções). Toda mudança de texto aqui muda o
hash da trava de prompts. Formato de um exemplo MCF:

    Pergunta: <pergunta>
    A. <opção>
    B. <opção>
    Resposta: B
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .core import (
    EVALUATIONS,
    GenerateRequest,
    GenerateResponse,
    LoglikelihoodRequest,
    LoglikelihoodResponse,
    MCQItem,
    Scored,
)

FEWSHOT_DELIMITER = "\n\n"
TARGET_DELIMITER = " "
ANSWER_PREFIX = "Resposta:"
GEN_INSTRUCTION = 'Responda com a letra da alternativa correta no formato "Resposta: X".'
GEN_STOP = ("\n\nPergunta:",)


# --- Prompts ----------------------------------------------------------------------


def question(item: MCQItem) -> str:
    lines = [f"Contexto: {item.context.strip()}"] if item.context else []
    lines.append(f"Pergunta: {item.question.strip()}")
    return "\n".join(lines)


def options(item: MCQItem) -> str:
    return "\n".join(
        f"{letter}. {opt.strip()}" for letter, opt in zip(item.letters, item.options, strict=True)
    )


def mcf(item: MCQItem, *, with_answer: bool) -> str:
    text = f"{question(item)}\n{options(item)}\n{ANSWER_PREFIX}"
    return text + TARGET_DELIMITER + item.letters[item.gold] if with_answer else text


def cf(item: MCQItem, *, with_answer: bool) -> str:
    text = f"{question(item)}\n{ANSWER_PREFIX}"
    return text + TARGET_DELIMITER + item.options[item.gold].strip() if with_answer else text


def gen_question(item: MCQItem) -> str:
    return f"{question(item)}\n{options(item)}\n\n{GEN_INSTRUCTION}"


def gen(item: MCQItem, shots: Sequence[MCQItem]) -> str:
    # Termina em "Resposta:" para modelos base completarem com a letra.
    query = f"{gen_question(item)}\n{ANSWER_PREFIX}"
    return FEWSHOT_DELIMITER.join([*(mcf(s, with_answer=True) for s in shots), query])


def gen_messages(item: MCQItem, shots: Sequence[MCQItem]) -> tuple[dict[str, str], ...]:
    """Few-shot multi-turn: cada exemplo é uma pergunta do usuário e a resposta do assistente."""
    messages: list[dict[str, str]] = []
    for s in shots:
        messages.append({"role": "user", "content": gen_question(s)})
        messages.append({"role": "assistant", "content": f"{ANSWER_PREFIX} {s.letters[s.gold]}"})
    messages.append({"role": "user", "content": gen_question(item)})
    return tuple(messages)


# --- Extração da letra ------------------------------------------------------------

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
        letter = next(g for g in matches[-1] if g).upper()
    else:
        match = _ONLY_LETTER.search(text) or _LEADING_LETTER.search(text)
        letter = match.group(1) if match else None
    return letter if letter is not None and letter in letters else None


# --- Avaliações -------------------------------------------------------------------


def argmax(values: Sequence[float]) -> int:
    """Índice do maior valor; em empate, o primeiro (como numpy.argmax)."""
    return max(range(len(values)), key=lambda i: (values[i], -i))


class MCQLoglikelihood:
    item_type = MCQItem
    needs = frozenset({"loglikelihood"})

    def __init__(self, formulation: str) -> None:
        self.name = formulation  # "mcf" | "cf"

    def _render(self, item: MCQItem, with_answer: bool) -> str:
        return (mcf if self.name == "mcf" else cf)(item, with_answer=with_answer)

    def _choices(self, item: MCQItem) -> list[str]:
        texts = item.letters if self.name == "mcf" else item.options
        return [TARGET_DELIMITER + t.strip() for t in texts]

    def build(self, item: MCQItem, shots: Sequence[MCQItem]) -> list[LoglikelihoodRequest]:
        context = FEWSHOT_DELIMITER.join([*(self._render(s, True) for s in shots), self._render(item, False)])
        return [LoglikelihoodRequest(context, c) for c in self._choices(item)]

    def score(self, item: MCQItem, responses: Sequence[LoglikelihoodResponse]) -> Scored:
        lls = [r.logprob for r in responses]
        metrics = {"acc": float(argmax(lls) == item.gold)}
        if self.name == "cf":
            texts = [c.strip() for c in self._choices(item)]
            char = [ll / max(1, len(t)) for ll, t in zip(lls, texts, strict=True)]
            token = [ll / max(1, r.n_tokens) for ll, r in zip(lls, responses, strict=True)]
            metrics |= {
                "acc_char": float(argmax(char) == item.gold),
                "acc_token": float(argmax(token) == item.gold),
            }
        return Scored(
            metrics={**metrics, "random_baseline": 1 / len(item.options)},
            details={
                "gold": item.letters[item.gold],
                "pred": item.letters[argmax(lls)],
                "logprobs": [round(ll, 6) for ll in lls],
                "n_tokens": [r.n_tokens for r in responses],
            },
        )


class MCQGenerate:
    name = "gen"
    item_type = MCQItem
    needs = frozenset({"generate"})
    supports_multiturn = True
    scoring_version = "3"

    def build(
        self, item: MCQItem, shots: Sequence[MCQItem], multiturn: bool = False
    ) -> list[GenerateRequest]:
        if multiturn and shots:
            messages = gen_messages(item, shots)
            prompt = FEWSHOT_DELIMITER.join(f"[{m['role']}] {m['content']}" for m in messages)
            return [GenerateRequest(prompt=prompt, messages=messages)]
        return [GenerateRequest(prompt=gen(item, shots), stop=GEN_STOP)]

    def score(self, item: MCQItem, responses: Sequence[GenerateResponse]) -> Scored:
        response = responses[0]
        gold, pred = item.letters[item.gold], extract_letter(response.text, item.letters)
        return Scored(
            metrics={
                "acc": float(pred == gold),
                "extracted": float(pred is not None),
                "truncated": float(response.finish_reason == "length"),
                "random_baseline": 1 / len(item.options),
            },
            details={
                "gold": gold,
                "pred": pred,
                "response": response.text,
                "reasoning": response.reasoning,
                "finish_reason": response.finish_reason,
                "error": response.error,
            },
        )


for _evaluation in (MCQLoglikelihood("mcf"), MCQLoglikelihood("cf"), MCQGenerate()):
    EVALUATIONS.register(_evaluation.name, _evaluation)
