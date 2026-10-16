"""Múltipla escolha por geração: o modelo escreve a resposta e extraímos a letra.

Métricas por item (regras de extração em metrics/extraction.py):
- acc:       a letra no formato pedido ("Resposta: X") é a correta
- extracted: a resposta seguiu o formato pedido
- acc_flex:  acerto aceitando também menções fora do formato ("a alternativa correta é C")
- truncated: a geração parou por limite de tokens
- random_baseline: acerto esperado chutando (1/nº de opções), para comparar com acc

Com `multiturn=True` (modelos de chat), os exemplos few-shot viram turnos
usuário/assistente em vez de texto corrido numa só mensagem.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.protocols import EVALUATIONS
from ..core.types import GenerateRequest, GenerateResponse, MCQItem, Scored
from ..metrics.extraction import extract_letter, extract_letter_flexible
from . import prompts_pt as pt


class MCQGenerate:
    name = "gen"
    item_type = MCQItem
    needs = frozenset({"generate"})
    supports_multiturn = True
    scoring_version = "2"  # muda quando a pontuação muda: amostras antigas são refeitas

    def build(
        self, item: MCQItem, shots: Sequence[MCQItem], multiturn: bool = False
    ) -> list[GenerateRequest]:
        if multiturn and shots:
            messages = pt.gen_messages(item, shots)
            prompt = pt.FEWSHOT_DELIMITER.join(f"[{m['role']}] {m['content']}" for m in messages)
            return [GenerateRequest(prompt=prompt, messages=messages)]
        return [GenerateRequest(prompt=pt.gen(item, shots), stop=pt.GEN_STOP)]

    def score(self, item: MCQItem, responses: Sequence[GenerateResponse]) -> Scored:
        response = responses[0]
        gold = item.letters[item.gold]
        pred = extract_letter(response.text, item.letters)
        pred_flex = extract_letter_flexible(response.text, item.letters)
        return Scored(
            metrics={
                "acc": float(pred == gold),
                "extracted": float(pred is not None),
                "acc_flex": float(pred_flex == gold),
                "truncated": float(response.finish_reason == "length"),
                "random_baseline": 1 / len(item.options),
            },
            details={
                "gold": gold,
                "pred": pred,
                "pred_flex": pred_flex,
                "response": response.text,
                "reasoning": response.reasoning,
                "finish_reason": response.finish_reason,
                "error": response.error,
            },
        )


EVALUATIONS.register("gen", MCQGenerate())
