"""Pedidos ao modelo, respostas e a pontuação de um item.

Os nomes e padrões dos campos dos pedidos entram no hash da trava de prompts.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any, ClassVar


@dataclass(frozen=True)
class LoglikelihoodRequest:
    """log P(continuation | context)."""

    kind: ClassVar[str] = "loglikelihood"
    context: str
    continuation: str


@dataclass(frozen=True)
class GenerateRequest:
    """Geração livre; o cliente aplica o chat template se o modelo usa."""

    kind: ClassVar[str] = "generate"
    prompt: str
    max_tokens: int | None = None  # None = padrão do modelo
    stop: tuple[str, ...] = ()  # aplicado só sem chat template (modelos base)
    # Conversa pronta ({"role", "content"}); se presente, modelos de chat a recebem no
    # lugar de `prompt`, que fica como versão em texto para o registro.
    messages: tuple[dict[str, str], ...] | None = None


Request = LoglikelihoodRequest | GenerateRequest


@dataclass(frozen=True)
class LoglikelihoodResponse:
    logprob: float
    is_greedy: bool  # a continuação inteira seria gerada por decodificação gulosa
    n_tokens: int


@dataclass(frozen=True)
class GenerateResponse:
    text: str  # resposta final, sem o raciocínio
    reasoning: str | None = None
    finish_reason: str | None = None  # "stop", "length", …
    error: str | None = None  # preenchido quando o pedido falhou de vez


Response = LoglikelihoodResponse | GenerateResponse


@dataclass(frozen=True)
class Scored:
    """Pontuação de um item: métricas (agregadas como média ± stderr) e detalhes para auditoria."""

    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
    # Peso do item na média de uma métrica (padrão 1); ex.: o IFEval por instrução pesa
    # pelo número de instruções do prompt.
    weights: dict[str, float] = field(default_factory=dict)


def non_default(obj: object) -> dict[str, Any]:
    """Campos de um dataclass fora do valor padrão: um campo novo não muda hashes antigos."""
    out = {}
    for f in fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, f.name)
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if value != default:
            out[f.name] = value
    return out
