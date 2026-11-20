# Portions copyright (c) 2020 EleutherAI, MIT. See NOTICE and LICENSE-MIT.

from __future__ import annotations

import re
import string
from collections.abc import Sequence

from ..core import (
    CompletionItem,
    GenerateRequest,
    GenerateResponse,
    LoglikelihoodResponse,
    MCQItem,
    Scored,
)
from .base import Evaluation
from .multiple_choice import argmax


def extract(pattern: str, text: str, group: int = 0) -> str:
    matches = re.findall(pattern, text)
    if not matches:
        return "[invalid]"
    result = matches[group]
    if isinstance(result, tuple):
        result = next((part for part in result if part), "[invalid]")
    return result.strip()


def gsm8k_normalize(text: str) -> str:
    # Exact order of regexes_to_ignore in gsm8k.yaml; no extra whitespace cleanup.
    for pattern in (",", r"\$", r"(?s).*#### ", r"\.$"):
        text = re.sub(pattern, "", text)
    return text.lower()


def normalize_letter(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


class ReferenceGeneration(Evaluation):
    supports_base_generation = True
    name = "exact_match"
    item_type = CompletionItem
    needs = frozenset({"generate"})
    supports_multiturn = True
    scoring_version = "exact-match-1"

    def build(self, item: CompletionItem, shots: Sequence[CompletionItem], multiturn=False):
        protocol = item.protocol
        if protocol not in {"gsm8k", "mmlu_pro", "mmlu_redux"}:
            raise ValueError(f"protocolo textual desconhecido: {protocol}")
        context = "".join(f"{s.prompt} {s.target}\n\n" for s in shots) + item.prompt
        # User/assistant turns use the same q/a functions as the single-turn request.
        messages = []
        if item.description:
            messages.append({"role": "system", "content": item.description})
        if multiturn:
            for shot in shots:
                messages.extend(
                    [
                        {"role": "user", "content": shot.prompt},
                        {"role": "assistant", "content": shot.target},
                    ]
                )
            messages.append({"role": "user", "content": item.prompt})
        else:
            messages.append({"role": "user", "content": context})
        prefix = item.description
        if prefix and not prefix[-1].isspace():
            prefix += "\n\n"
        prompt = prefix + context
        stop = {
            "gsm8k": ("Question:", "</s>", "<|im_end|>"),
            "mmlu_pro": ("Question:",),
            "mmlu_redux": ("</s>",),
        }[protocol]
        return [
            GenerateRequest(
                prompt=prompt,
                messages=tuple(messages),
                max_tokens=2048 if protocol == "mmlu_pro" else 256,
                stop=stop,
                params={"temperature": 0},
                enforce_generation_config=True,
            )
        ]

    def score(self, item: CompletionItem, responses: Sequence[GenerateResponse]) -> Scored:
        response = responses[0]
        text = response.text
        if item.protocol == "gsm8k":
            preds = {
                "strict-match": extract(r"#### (\-?[0-9\.\,]+)", text),
                "flexible-extract": extract(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text, -1),
            }
            normalize = gsm8k_normalize
        elif item.protocol == "mmlu_pro":
            preds = {"custom-extract": extract(r"answer is \(?([ABCDEFGHIJ])\)?", text)}
            normalize = normalize_letter
        elif item.protocol == "mmlu_redux":
            preds = {"default": extract(r"([ABCD])", text)}
            normalize = normalize_letter
        else:
            raise ValueError(f"protocolo textual desconhecido: {item.protocol}")
        return Scored(
            metrics={
                f"exact_match,{f}": float(normalize(p) == normalize(item.target)) for f, p in preds.items()
            },
            details={
                "target": item.target,
                "predictions": preds,
                **response.details(),
            },
        )


class GPQAMultipleChoice(Evaluation):
    supports_fewshot = False
    name = "gpqa_mcf"
    item_type = MCQItem
    needs = frozenset({"loglikelihood"})
    scoring_version = "gpqa-1"

    def build(self, item: MCQItem, shots: Sequence[MCQItem], multiturn=False):
        from ..core import LoglikelihoodRequest

        if shots:
            raise ValueError("GPQA zeroshot exige num_fewshot=0")
        prompt = f"What is the correct answer to this question:{item.question}\nChoices:\n"
        prompt += "\n".join(f"({c}) {o}" for c, o in zip(item.letters, item.options, strict=True))
        prompt += "\nAnswer:"
        return [LoglikelihoodRequest(prompt, f" ({letter})") for letter in item.letters]

    def score(self, item: MCQItem, responses: Sequence[LoglikelihoodResponse]) -> Scored:
        lls = [r.logprob for r in responses]
        acc = float(argmax(lls) == item.gold)
        # All targets, '(A)' through '(D)', have the same character length.
        return Scored(
            metrics={"acc": acc, "acc_norm": acc},
            details={
                "gold": item.letters[item.gold],
                "pred": item.letters[argmax(lls)],
                "logprobs": lls,
                "n_tokens": [r.n_tokens for r in responses],
            },
        )
