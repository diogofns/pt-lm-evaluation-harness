"""Acurácia de múltipla escolha a partir de log-likelihoods.

- acc:       maior log P(opção | contexto)
- acc_char:  normalizado pelo número de caracteres da opção (o `acc_norm` do lm-eval)
- acc_token: normalizado pelo número de tokens da opção
- acc_pmi:   log P(opção | contexto) − log P(opção | "Resposta:")
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.types import LoglikelihoodResponse


def argmax(values: Sequence[float]) -> int:
    """Índice do maior valor; em empate, o primeiro (como numpy.argmax)."""
    return max(range(len(values)), key=lambda i: (values[i], -i))


def choice_accuracy(
    responses: Sequence[LoglikelihoodResponse],
    gold: int,
    choice_texts: Sequence[str],
    unconditional: Sequence[LoglikelihoodResponse] | None = None,
) -> dict[str, float]:
    lls = [r.logprob for r in responses]
    scores = {
        "acc": lls,
        "acc_char": [ll / max(1, len(t)) for ll, t in zip(lls, choice_texts, strict=True)],
        "acc_token": [ll / max(1, r.n_tokens) for ll, r in zip(lls, responses, strict=True)],
    }
    if unconditional is not None:
        scores["acc_pmi"] = [ll - u.logprob for ll, u in zip(lls, unconditional, strict=True)]
    return {name: float(argmax(s) == gold) for name, s in scores.items()}
