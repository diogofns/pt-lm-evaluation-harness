"""Teste de sanidade logo depois que o endpoint de um modelo sobe.

Se passar, o modelo está servindo e respondendo, e as avaliações seguem sem
checagens por pedido. Se falhar, o modelo é pulado antes de gastar tempo e o
experimento termina com erro depois dos demais modelos.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..core.protocols import Backend
from ..core.types import GenerateRequest, LoglikelihoodRequest

CONTEXT, CONTINUATION = "A capital do Brasil é", " Brasília"
PROMPT = "Responda apenas com a palavra: ok"


class SmokeTestError(RuntimeError):
    pass


async def smoke_test(backend: Backend, needs: Iterable[str]) -> None:
    """Um pedido de cada tipo que o modelo vai usar; levanta SmokeTestError se algum falhar."""
    needs = set(needs)
    problems = []
    if "loglikelihood" in needs:
        try:
            (r,) = await backend.loglikelihood([LoglikelihoodRequest(CONTEXT, CONTINUATION)])
            if not (math.isfinite(r.logprob) and r.logprob < 0 and r.n_tokens > 0):
                problems.append(f"log-likelihood inválido: {r}")
        except Exception as exc:
            problems.append(f"log-likelihood falhou: {type(exc).__name__}: {exc}")
    if "generate" in needs:
        (g,) = await backend.generate([GenerateRequest(prompt=PROMPT)])
        if g.error:
            problems.append(f"geração falhou: {g.error}")
        elif not (g.text or g.reasoning):
            problems.append(f"geração vazia (finish_reason={g.finish_reason})")
    if problems:
        raise SmokeTestError("; ".join(problems))
