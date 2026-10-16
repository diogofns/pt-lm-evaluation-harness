"""Tipos de dados que circulam entre benchmarks, avaliações e backends.

Tudo aqui é imutável e sem comportamento além de validação: um item descreve
uma questão, um pedido descreve o que perguntar ao modelo, uma resposta
descreve o que o modelo devolveu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

LETTERS = "ABCDEFGHIJ"


# --- Itens (o que um benchmark produz) --------------------------------------


@dataclass(frozen=True)
class Item:
    """Base de todos os itens. Novos tipos de avaliação definem subclasses."""

    id: str


@dataclass(frozen=True)
class MCQItem(Item):
    """Questão de múltipla escolha."""

    question: str
    options: tuple[str, ...]
    gold: int  # índice 0-based em `options`
    subject: str | None = None
    context: str | None = None  # texto de apoio mostrado antes da pergunta

    def __post_init__(self) -> None:
        if not isinstance(self.options, tuple):
            object.__setattr__(self, "options", tuple(self.options))
        if not 2 <= len(self.options) <= len(LETTERS):
            raise ValueError(f"item {self.id}: precisa de 2 a {len(LETTERS)} opções")
        if any(not str(o).strip() for o in self.options):
            raise ValueError(f"item {self.id}: há opção vazia")
        if not 0 <= self.gold < len(self.options):
            raise ValueError(
                f"item {self.id}: gold={self.gold} fora do intervalo de {len(self.options)} opções"
            )

    @property
    def letters(self) -> str:
        return LETTERS[: len(self.options)]


@dataclass(frozen=True)
class InstructionItem(Item):
    """Prompt com instruções verificáveis (IFEval): cada id tem seus parâmetros em `kwargs`."""

    prompt: str
    instruction_ids: tuple[str, ...]
    kwargs: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instruction_ids", tuple(self.instruction_ids))
        object.__setattr__(self, "kwargs", tuple(dict(k or {}) for k in self.kwargs))
        if not self.instruction_ids:
            raise ValueError(f"item {self.id}: sem instruções")
        if len(self.kwargs) != len(self.instruction_ids):
            raise ValueError(
                f"item {self.id}: {len(self.instruction_ids)} instruções e {len(self.kwargs)} kwargs"
            )


# --- Pedidos (o que uma avaliação pede ao backend) --------------------------


@dataclass(frozen=True)
class LoglikelihoodRequest:
    """log P(continuation | context). Contexto vazio = probabilidade incondicional."""

    kind: ClassVar[str] = "loglikelihood"
    context: str
    continuation: str


@dataclass(frozen=True)
class GenerateRequest:
    """Geração livre. O backend decide se aplica chat template, conforme o modelo."""

    kind: ClassVar[str] = "generate"
    prompt: str
    max_tokens: int | None = None  # None = padrão do modelo
    stop: tuple[str, ...] = ()  # aplicado só sem chat template (modelos base)
    # Conversa pronta (lista de {"role", "content"}). Se presente, modelos de chat a
    # recebem no lugar de `prompt`, que fica como versão em texto para o registro.
    messages: tuple[dict[str, str], ...] | None = None


Request = LoglikelihoodRequest | GenerateRequest


# --- Respostas (o que o backend devolve) ------------------------------------


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


# --- Pontuação (o que uma avaliação devolve por item) -----------------------


@dataclass(frozen=True)
class Scored:
    metrics: dict[str, float]  # agregadas como média ± stderr
    details: dict[str, Any] = field(default_factory=dict)  # vai para samples.jsonl
    # Peso do item na média de uma métrica (padrão 1). Ex.: inst_level_acc do IFEval pesa
    # pelo número de instruções, para a média ser sobre instruções e não sobre prompts.
    weights: dict[str, float] = field(default_factory=dict)
