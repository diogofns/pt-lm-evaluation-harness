from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class Item:
    id: str


@dataclass(frozen=True)
class MCQItem(Item):
    """Questão de múltipla escolha."""

    question: str
    options: tuple[str, ...]
    gold: int  # índice 0-based em `options`
    subject: str | None = None
    context: str | None = None  # texto de apoio mostrado antes da pergunta
    protocol: str = "pt"  # protocolo de prompt específico do benchmark
    prompt_description: str | None = None  # prefixo opcional do protocolo de avaliação
    task_group: str | None = None  # subtarefa de avaliação; INCLUDE agrupa por domínio

    def __post_init__(self) -> None:
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
class CompletionItem(Item):
    """Prompt e resposta textual de referência (GSM8K, MMLU-Pro e MMLU-Redux)."""

    prompt: str
    target: str
    protocol: str
    subject: str | None = None
    description: str = ""


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
