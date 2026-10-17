"""Classe base dos benchmarks: de onde vêm os dados e como cada linha vira um item.

Cada benchmark é uma subclasse de Benchmark numa pasta própria (benchmarks/<nome>/),
com `benchmark.py` e o cartão `card.md`; subclasses concretas com `name` se registram em
BENCHMARKS ao serem definidas.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..core import LETTERS, Item, MCQItem, Registry, non_default

if TYPE_CHECKING:
    from ..evaluations import Evaluation

CF_NORMALIZATIONS = ("char", "token")
ANSWER_FORMATS = ("auto", "letter", "index", "text")


@dataclass(frozen=True)
class HFDataset:
    """Dataset do Hub. `split="*"` carrega todos os splits; `subset` pode ser uma tupla
    (ex.: um config por ano), e cada linha recebe `_subset`, `_split` e `_index`."""

    repo: str
    split: str = "test"
    subset: str | tuple[str, ...] | None = None
    revision: str | None = None  # fixe um commit para resultados reproduzíveis

    @property
    def subsets(self) -> tuple[str | None, ...]:
        return self.subset if isinstance(self.subset, tuple) else (self.subset,)

    def load(self) -> list[dict[str, Any]]:
        from datasets import get_dataset_split_names, load_dataset

        rows: list[dict[str, Any]] = []
        for subset in self.subsets:
            splits = (
                get_dataset_split_names(self.repo, subset, revision=self.revision)
                if self.split == "*"
                else [self.split]
            )
            for split in splits:
                ds = load_dataset(self.repo, subset, split=split, revision=self.revision)
                rows.extend(
                    {**row, "_subset": subset, "_split": split, "_index": i} for i, row in enumerate(ds)
                )
        return rows

    def describe(self) -> str:
        subsets = "+".join(s for s in self.subsets if s)
        name = f"{self.repo}/{subsets}" if subsets else self.repo
        return f"hf:{name}:{self.split}@{_resolve_revision(self.repo, self.revision)}"


@cache
def _resolve_revision(repo: str, revision: str | None) -> str:
    """Commit exato do dataset, para registrar junto dos resultados."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo, revision=revision).sha or (revision or "main")
    except Exception:  # sem rede ou sem permissão: registra o que foi pedido
        return revision or "main"


@dataclass(frozen=True)
class Loaded:
    items: list[Item]
    pool: list[Item]  # candidatos a few-shot
    discarded: int  # linhas para as quais to_item devolveu None


class Benchmark(ABC):
    """Um benchmark: dados (revisão fixada), conversão de linhas e avaliações declaradas.

    - `evaluations`: as que fazem sentido para os itens; quais rodam depende do tipo do
      modelo (base: log-likelihood; instruct/api: geração).
    - `cf_normalization`: qual normalização do CF vira `acc_norm`, igual para todos os
      modelos: "char" (padrão, não depende do tokenizer) ou "token".
    - `fewshot_source`: de onde vêm os exemplos (None = o próprio conjunto, sem o item).
    """

    name: ClassVar[str]
    version: ClassVar[str]
    source: ClassVar[HFDataset]
    description: ClassVar[str] = ""
    evaluations: ClassVar[tuple[str, ...]] = ("mcf", "cf", "gen")
    fewshot_source: ClassVar[HFDataset | None] = None
    cf_normalization: ClassVar[str] = "char"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls) or "name" not in cls.__dict__:
            return
        if cls.cf_normalization not in CF_NORMALIZATIONS:
            raise ValueError(f"{cls.name}: cf_normalization deve ser uma de {CF_NORMALIZATIONS}")
        BENCHMARKS.register(cls.name, cls())

    @abstractmethod
    def to_item(self, row: dict[str, Any]) -> Item | None:
        """Converte uma linha do dataset num item; None descarta a linha."""

    @property
    def card(self) -> Path:
        return Path(inspect.getfile(type(self))).with_name("card.md")

    @cached_property
    def data(self) -> Loaded:
        """Itens e candidatos a few-shot, carregados uma vez por processo."""
        rows = self.source.load()
        items = [it for it in map(self.to_item, rows) if it is not None]
        if len({it.id for it in items}) != len(items):
            raise ValueError(f"{self.name}: há IDs de item repetidos")
        pool = items
        if self.fewshot_source:
            pool = [it for it in map(self.to_item, self.fewshot_source.load()) if it is not None]
        return Loaded(items, pool, discarded=len(rows) - len(items))

    def select_shots(self, item: Item, k: int, seed: int) -> list[Item]:
        """`k` exemplos sorteados com (seed, item.id): estáveis entre execuções e independentes
        da ordem. Prefere exemplos da mesma matéria quando há suficientes."""
        if k <= 0:
            return []
        exclude_self = self.fewshot_source is None
        candidates = [p for p in self.data.pool if not (exclude_self and p.id == item.id)]
        subject = getattr(item, "subject", None)
        if subject is not None:
            same = [p for p in candidates if getattr(p, "subject", None) == subject]
            if len(same) >= k:
                candidates = same
        return random.Random(f"{seed}:{item.id}").sample(candidates, min(k, len(candidates)))

    def prompt_signature(self, evaluation: Evaluation) -> str:
        """Hash dos pedidos de um item de referência com um exemplo few-shot: muda quando o
        texto dos prompts muda, pela avaliação ou pelos dados."""
        probe = self.data.items[0]
        shot = next((p for p in self.data.pool if p.id != probe.id), probe)
        payload = json.dumps([non_default(r) for r in evaluation.build(probe, [shot])], ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --- Benchmarks de colunas comuns -------------------------------------------------


@dataclass(frozen=True)
class Columns:
    """Como ler uma linha de colunas comuns (caminhos com ponto acessam campos aninhados).

    - `options`: caminho para uma lista ("choices.text") ou lista de colunas (["A", "B", "C", "D"]).
    - `answer_is`: como ler o gabarito: "letter" ("B", "(b)"), "index" (0-based), "text"
      (o texto da opção) ou "auto". Em "auto", um gabarito que vale como letra e como
      texto de outra opção (ex.: "B" com as opções "A", "~A", "B", "~B") é erro.
    """

    question: str
    options: str | Sequence[str]
    answer: str
    answer_is: str = "auto"
    id: str | None = None
    subject: str | None = None
    context: str | None = None

    def __post_init__(self) -> None:
        if self.answer_is not in ANSWER_FORMATS:
            raise ValueError(f"answer_is deve ser um de {ANSWER_FORMATS}, veio {self.answer_is!r}")

    def to_item(self, row: dict[str, Any]) -> MCQItem:
        if isinstance(self.options, str):
            opts = [str(o) for o in get_path(row, self.options)]
        else:
            opts = [str(get_path(row, col)) for col in self.options]
        return MCQItem(
            id=str(get_path(row, self.id)) if self.id else default_id(row),
            question=str(get_path(row, self.question)),
            options=tuple(opts),
            gold=answer_index(get_path(row, self.answer), opts, self.answer_is),
            subject=self._optional(row, self.subject),
            context=self._optional(row, self.context),
        )

    @staticmethod
    def _optional(row: dict[str, Any], path: str | None) -> str | None:
        value = get_path(row, path) if path else None
        return (str(value).strip() or None) if value is not None else None


class ColumnBenchmark(Benchmark):
    """Benchmark cujas linhas se convertem só com nomes de colunas (`columns`)."""

    columns: ClassVar[Columns]

    def to_item(self, row: dict[str, Any]) -> MCQItem:
        return self.columns.to_item(row)


def get_path(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for key in path.split("."):
        value = value[key]
    return value


def default_id(row: dict[str, Any]) -> str:
    return f"{row['_split']}-{row['_index']}"


def letter_index(value: str) -> int:
    """'B', ' b ', '(B)', 'B)' → 1."""
    letter = str(value).strip().strip("()").strip().upper()
    if len(letter) != 1 or letter not in LETTERS:
        raise ValueError(f"não é uma letra de alternativa: {value!r}")
    return LETTERS.index(letter)


def answer_index(value: Any, options: Sequence[str], answer_is: str = "auto") -> int:
    """Índice da opção correta, lendo `value` como `answer_is` manda (ver `Columns`)."""
    if answer_is == "index" or (
        answer_is == "auto" and isinstance(value, int) and not isinstance(value, bool)
    ):
        return int(value)
    text = str(value).strip()
    if answer_is == "letter":
        return letter_index(text)
    by_text = list(options).index(text) if text in options else None
    if answer_is == "text":
        if by_text is None:
            raise ValueError(f"gabarito {text!r} não é o texto de nenhuma opção")
        return by_text
    try:
        by_letter: int | None = letter_index(text)
    except ValueError:
        by_letter = None
    if by_text is not None and by_letter is not None and by_text != by_letter:
        raise ValueError(
            f"gabarito {text!r} ambíguo: é a letra da opção {by_letter} e o texto da opção {by_text}; "
            "declare answer_is em Columns"
        )
    if by_text is None and by_letter is None:
        raise ValueError(f"gabarito {text!r} não é letra, índice nem texto de uma opção")
    return by_text if by_text is not None else by_letter  # type: ignore[return-value]


BENCHMARKS: Registry[Benchmark] = Registry("benchmark")
