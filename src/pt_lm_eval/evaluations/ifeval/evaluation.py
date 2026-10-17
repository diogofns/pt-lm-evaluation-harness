"""Avaliação de seguimento de instruções (IFEval), com as instruções do Portuguese-IFEval.

Métricas por item, como na referência: prompt_level_{strict,loose}_acc (todas as
instruções seguidas), inst_level_{strict,loose}_acc (fração seguida; o agregado é sobre
instruções, então o item pesa pelo número delas) e truncated.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...core import GenerateRequest, GenerateResponse, InstructionItem, Scored
from ..base import Evaluation
from .instructions import INSTRUCTIONS, instruction


class InstructionFollowing(Evaluation):
    name = "ifeval"
    item_type = InstructionItem
    needs = frozenset({"generate"})
    scoring_version = "1"
    max_tokens = 2048  # o da geração de referência (vLLM, temperatura 0)

    def build(
        self, item: InstructionItem, shots: Sequence[InstructionItem], multiturn: bool = False
    ) -> list[GenerateRequest]:
        unknown = [i for i in item.instruction_ids if i not in INSTRUCTIONS]
        if unknown:
            raise KeyError(f"item {item.id}: instruções sem verificador: {unknown}")
        return [GenerateRequest(prompt=item.prompt, max_tokens=self.max_tokens)]

    def score(self, item: InstructionItem, responses: Sequence[GenerateResponse]) -> Scored:
        response = responses[0]
        checks = [
            instruction(i, kw, item.prompt) for i, kw in zip(item.instruction_ids, item.kwargs, strict=True)
        ]
        strict = [c.follows_strict(response.text) for c in checks]
        loose = [c.follows_loose(response.text) for c in checks]
        n = len(checks)
        return Scored(
            metrics={
                "prompt_level_strict_acc": float(all(strict)),
                "inst_level_strict_acc": sum(strict) / n,
                "prompt_level_loose_acc": float(all(loose)),
                "inst_level_loose_acc": sum(loose) / n,
                "truncated": float(response.finish_reason == "length"),
            },
            details={
                "instructions": list(item.instruction_ids),
                "strict": strict,
                "loose": loose,
                "response": response.text,
                "reasoning": response.reasoning,
                "finish_reason": response.finish_reason,
                "error": response.error,
            },
            weights={"inst_level_strict_acc": n, "inst_level_loose_acc": n},
        )
