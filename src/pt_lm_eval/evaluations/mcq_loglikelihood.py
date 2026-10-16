"""Múltipla escolha por log-likelihood: MCF (letras) e CF (texto das opções).

- mcf:    opções no prompt; pontua " A", " B", … após "Resposta:".
- cf:     opções fora do prompt; pontua " <texto da opção>".
- cf_pmi: cf + acc_pmi, que custa o dobro de pedidos.

Todas registram random_baseline (1/nº de opções). No CF, o resultado agregado ganha
acc_norm, a normalização fixada pelo benchmark (`cf_normalization`, padrão acc_char).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.protocols import EVALUATIONS
from ..core.types import LoglikelihoodRequest, LoglikelihoodResponse, MCQItem, Scored
from ..metrics.accuracy import argmax, choice_accuracy
from . import prompts_pt as pt


class MCQLoglikelihood:
    item_type = MCQItem
    needs = frozenset({"loglikelihood"})

    def __init__(self, name: str, formulation: str, metrics: tuple[str, ...], pmi: bool = False):
        self.name = name
        self.formulation = formulation
        self.metrics = metrics
        self.pmi = pmi

    def _render(self, item: MCQItem, with_answer: bool) -> str:
        render = pt.mcf if self.formulation == "mcf" else pt.cf
        return render(item, with_answer=with_answer)

    def _choices(self, item: MCQItem) -> list[str]:
        texts = item.letters if self.formulation == "mcf" else item.options
        return [pt.TARGET_DELIMITER + t.strip() for t in texts]

    def build(self, item: MCQItem, shots: Sequence[MCQItem]) -> list[LoglikelihoodRequest]:
        context = pt.with_shots([self._render(s, with_answer=True) for s in shots], self._render(item, False))
        requests = [LoglikelihoodRequest(context, c) for c in self._choices(item)]
        if self.pmi:
            requests += [LoglikelihoodRequest(pt.ANSWER_PREFIX, c) for c in self._choices(item)]
        return requests

    def score(self, item: MCQItem, responses: Sequence[LoglikelihoodResponse]) -> Scored:
        n = len(item.options)
        conditional, unconditional = responses[:n], (responses[n:] if self.pmi else None)
        texts = [c.strip() for c in self._choices(item)]
        metrics = choice_accuracy(conditional, item.gold, texts, unconditional)
        return Scored(
            metrics={**{m: metrics[m] for m in self.metrics}, "random_baseline": 1 / n},
            details={
                "gold": item.letters[item.gold],
                "pred": item.letters[argmax([r.logprob for r in conditional])],
                "logprobs": [round(r.logprob, 6) for r in conditional],
                "n_tokens": [r.n_tokens for r in conditional],
            },
        )


EVALUATIONS.register("mcf", MCQLoglikelihood("mcf", "mcf", ("acc",)))
EVALUATIONS.register("cf", MCQLoglikelihood("cf", "cf", ("acc", "acc_char", "acc_token")))
EVALUATIONS.register(
    "cf_pmi", MCQLoglikelihood("cf_pmi", "cf", ("acc", "acc_char", "acc_token", "acc_pmi"), pmi=True)
)
