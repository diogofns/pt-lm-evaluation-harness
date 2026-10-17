"""Tipos que circulam entre benchmarks, avaliações e o cliente, e os registros por nome.

Um item descreve uma questão, um pedido descreve o que perguntar ao modelo, uma
resposta descreve o que ele devolveu. Benchmarks e avaliações se registram por nome
em BENCHMARKS e EVALUATIONS ao serem importados (ver benchmarks.py, mcq.py, ifeval.py).
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Protocol, TypeVar

LETTERS = "ABCDEFGHIJ"
CF_NORMALIZATIONS = ("char", "token")


# --- Itens (o que um benchmark produz) --------------------------------------------


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


# --- Pedidos e respostas ----------------------------------------------------------
# Os nomes e padrões dos campos dos pedidos entram no hash da trava de prompts.


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


# --- Benchmarks e avaliações ------------------------------------------------------


@dataclass(frozen=True)
class Benchmark:
    """Dados e conversão de cada linha em item.

    - `to_item` pode devolver None para descartar uma linha inválida.
    - `cf_normalization` fixa qual normalização do CF vira `acc_norm`, igual para todos
      os modelos: "char" (padrão, não depende do tokenizer) ou "token".
    - `card`: nome do cartão em cards/ (padrão: o nome do benchmark).
    """

    version: str
    source: Any  # data.HFDataset
    to_item: Callable[[dict[str, Any]], Item | None]
    evaluations: tuple[str, ...]
    fewshot_source: Any = None  # None = exemplos do próprio conjunto, sem o item
    description: str = ""
    card: str | None = None
    cf_normalization: str = "char"

    def __post_init__(self) -> None:
        if self.cf_normalization not in CF_NORMALIZATIONS:
            raise ValueError(
                f"cf_normalization deve ser uma de {CF_NORMALIZATIONS}, veio {self.cf_normalization!r}"
            )


class Evaluation(Protocol):
    """Transforma itens em pedidos e pontua as respostas.

    - `needs`: tipos de pedido usados; decidem em que cenário a avaliação roda
      (base: log-likelihood; instruct/api: geração).
    - `supports_multiturn`: aceita `build(..., multiturn=True)` (few-shot como turnos).
    - `scoring_version`: entra na identidade das amostras; aumente ao mudar a pontuação.
    """

    name: str
    item_type: type[Item]
    needs: frozenset[str]

    def build(self, item: Item, shots: Sequence[Item]) -> list[Request]: ...
    def score(self, item: Item, responses: Sequence[Response]) -> Scored: ...


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, obj: T) -> T:
        if name in self._items:
            raise ValueError(f"{self.kind} '{name}' já registrado")
        self._items[name] = obj
        return obj

    def get(self, name: str) -> T:
        if name not in self._items:
            close = difflib.get_close_matches(name, self._items, n=3)
            hint = f" Quis dizer: {', '.join(close)}?" if close else ""
            raise KeyError(f"{self.kind} '{name}' não existe.{hint}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


BENCHMARKS: Registry[Benchmark] = Registry("benchmark")
EVALUATIONS: Registry[Evaluation] = Registry("avaliação")
