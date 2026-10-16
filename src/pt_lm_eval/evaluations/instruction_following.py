"""Seguimento de instruções verificáveis (IFEval), com os verificadores do Portuguese-IFEval.

O prompt vai como está (modelos de chat o recebem como mensagem do usuário) e a
resposta final, já sem o raciocínio, passa pelos verificadores de cada instrução.

Métricas por item, como na referência:
- prompt_level_strict_acc / prompt_level_loose_acc: todas as instruções seguidas
- inst_level_strict_acc / inst_level_loose_acc: fração das instruções seguidas; o
  agregado é sobre instruções (o item pesa pelo número delas), não sobre prompts
- truncated: a geração parou por limite de tokens

loose aceita a resposta se alguma variante passar (sem a primeira e/ou a última
linha, sem asteriscos), para não penalizar "Claro! Aqui está:" e markdown.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.protocols import EVALUATIONS
from ..core.types import GenerateRequest, GenerateResponse, InstructionItem, Scored
from ..metrics.instruction_following import CHECKERS, follows_loose, follows_strict

MAX_TOKENS = 2048  # o da geração de referência (vLLM, temperatura 0)


class InstructionFollowing:
    name = "ifeval"
    item_type = InstructionItem
    needs = frozenset({"generate"})
    scoring_version = "1"

    def build(self, item: InstructionItem, shots: Sequence[InstructionItem]) -> list[GenerateRequest]:
        unknown = [i for i in item.instruction_ids if i not in CHECKERS]
        if unknown:
            raise KeyError(f"item {item.id}: instruções sem verificador: {unknown}")
        return [GenerateRequest(prompt=item.prompt, max_tokens=MAX_TOKENS)]

    def score(self, item: InstructionItem, responses: Sequence[GenerateResponse]) -> Scored:
        response = responses[0]
        pairs = list(zip(item.instruction_ids, item.kwargs, strict=True))
        strict = [follows_strict(response.text, i, kw, item.prompt) for i, kw in pairs]
        loose = [follows_loose(response.text, i, kw, item.prompt) for i, kw in pairs]
        n = len(pairs)
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


EVALUATIONS.register("ifeval", InstructionFollowing())
