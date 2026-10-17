"""Dados dos benchmarks: fonte no Hugging Face, conversão de colunas, few-shot e assinatura dos prompts."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import MISSING, dataclass, fields
from functools import cache
from typing import Any

from .core import BENCHMARKS, LETTERS, Evaluation, Item, MCQItem


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


# --- Colunas → MCQItem ------------------------------------------------------------

ANSWER_FORMATS = ("auto", "letter", "index", "text")


def columns(
    *,
    question: str,
    options: str | Sequence[str],
    answer: str,
    id: str | None = None,
    subject: str | None = None,
    context: str | None = None,
    answer_is: str = "auto",
) -> Callable[[dict[str, Any]], MCQItem]:
    """Monta um `to_item` a partir de nomes de colunas (caminhos com ponto acessam campos aninhados).

    - `options`: caminho para uma lista ("choices.text") ou lista de colunas (["A", "B", "C", "D"]).
    - `answer_is`: como ler o gabarito: "letter" ("B", "(b)"), "index" (0-based), "text"
      (o texto da opção) ou "auto". Em "auto", um gabarito que vale como letra e como
      texto de outra opção (ex.: "B" com as opções "A", "~A", "B", "~B") é erro.
    """
    if answer_is not in ANSWER_FORMATS:
        raise ValueError(f"answer_is deve ser um de {ANSWER_FORMATS}, veio {answer_is!r}")

    def optional(row: dict[str, Any], path: str | None) -> str | None:
        value = get_path(row, path) if path else None
        return (str(value).strip() or None) if value is not None else None

    def to_item(row: dict[str, Any]) -> MCQItem:
        if isinstance(options, str):
            opts = [str(o) for o in get_path(row, options)]
        else:
            opts = [str(get_path(row, col)) for col in options]
        return MCQItem(
            id=str(get_path(row, id)) if id else default_id(row),
            question=str(get_path(row, question)),
            options=tuple(opts),
            gold=answer_index(get_path(row, answer), opts, answer_is),
            subject=optional(row, subject),
            context=optional(row, context),
        )

    return to_item


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
    """Índice da opção correta, lendo `value` como `answer_is` manda (ver `columns`)."""
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
            "declare answer_is em columns()"
        )
    if by_text is None and by_letter is None:
        raise ValueError(f"gabarito {text!r} não é letra, índice nem texto de uma opção")
    return by_text if by_text is not None else by_letter  # type: ignore[return-value]


# --- Itens, few-shot e assinatura -------------------------------------------------


@dataclass(frozen=True)
class Loaded:
    items: list[Item]
    pool: list[Item]  # candidatos a few-shot
    discarded: int  # linhas para as quais to_item devolveu None


@cache
def load_items(name: str) -> Loaded:
    """Itens de um benchmark, carregados uma vez por processo."""
    bench = BENCHMARKS.get(name)
    rows = bench.source.load()
    items = [it for it in map(bench.to_item, rows) if it is not None]
    if len({it.id for it in items}) != len(items):
        raise ValueError(f"{name}: há IDs de item repetidos")
    pool = items
    if bench.fewshot_source:
        pool = [it for it in map(bench.to_item, bench.fewshot_source.load()) if it is not None]
    return Loaded(items, pool, discarded=len(rows) - len(items))


def select_shots(
    item: Item, pool: Sequence[Item], k: int, *, seed: int, exclude_self: bool = True
) -> list[Item]:
    """`k` exemplos sorteados com (seed, item.id): estáveis entre execuções e independentes
    da ordem. Prefere exemplos da mesma matéria quando há suficientes."""
    if k <= 0:
        return []
    candidates = [p for p in pool if not (exclude_self and p.id == item.id)]
    subject = getattr(item, "subject", None)
    if subject is not None:
        same = [p for p in candidates if getattr(p, "subject", None) == subject]
        if len(same) >= k:
            candidates = same
    return random.Random(f"{seed}:{item.id}").sample(candidates, min(k, len(candidates)))


def prompt_signature(evaluation: Evaluation, items: Sequence[Item], pool: Sequence[Item]) -> str:
    """Hash dos pedidos de um item de referência com um exemplo few-shot: muda quando o
    texto dos prompts muda, pela avaliação ou pelos dados."""
    probe = items[0]
    shot = next((p for p in pool if p.id != probe.id), probe)
    payload = json.dumps([non_default(r) for r in evaluation.build(probe, [shot])], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def non_default(obj: object) -> dict[str, Any]:
    """Campos de um dataclass fora do valor padrão: um campo novo não muda hashes antigos."""
    out = {}
    for f in fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, f.name)
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if value != default:
            out[f.name] = value
    return out
