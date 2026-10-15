"""Múltipla escolha por geração: o modelo escreve a resposta e extraímos a letra.

Métricas por item:
- acc:       a letra extraída é a correta
- extracted: foi possível extrair uma letra (o complemento é a taxa de falha)
- truncated: a geração parou por limite de tokens
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.protocols import EVALUATIONS
from ..core.types import GenerateRequest, GenerateResponse, MCQItem, Scored
from ..metrics.extraction import extract_letter
from . import prompts_pt as pt


class MCQGenerate:
    name = "gen"
    item_type = MCQItem
    needs = frozenset({"generate"})

    def build(self, item: MCQItem, shots: Sequence[MCQItem]) -> list[GenerateRequest]:
        return [GenerateRequest(prompt=pt.gen(item, shots), stop=pt.GEN_STOP)]

    def score(self, item: MCQItem, responses: Sequence[GenerateResponse]) -> Scored:
        response = responses[0]
        gold = item.letters[item.gold]
        pred = extract_letter(response.text, item.letters)
        return Scored(
            metrics={
                "acc": float(pred == gold),
                "extracted": float(pred is not None),
                "truncated": float(response.finish_reason == "length"),
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


EVALUATIONS.register("gen", MCQGenerate())
