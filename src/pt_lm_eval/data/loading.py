"""Carga dos itens de um benchmark e assinatura dos prompts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import MISSING, dataclass, fields
from functools import cache

from ..core.protocols import BENCHMARKS, Evaluation
from ..core.types import Item


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
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: há IDs de item repetidos")
    if bench.fewshot_source:
        pool = [it for it in map(bench.to_item, bench.fewshot_source.load()) if it is not None]
    else:
        pool = items
    return Loaded(items, pool, discarded=len(rows) - len(items))


def prompt_signature(evaluation: Evaluation, items: Sequence[Item], pool: Sequence[Item]) -> str:
    """Hash dos pedidos gerados para um item de referência com um exemplo few-shot.

    Muda sempre que o texto dos prompts muda, seja pela avaliação, seja pelos dados.
    """
    probe = items[0]
    shot = next((p for p in pool if p.id != probe.id), probe)
    payload = json.dumps([_non_default(r) for r in evaluation.build(probe, [shot])], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _non_default(request: object) -> dict:
    """Campos do pedido fora do valor padrão: um campo opcional novo não muda a assinatura."""
    out = {}
    for f in fields(request):  # type: ignore[arg-type]
        value = getattr(request, f.name)
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        if value != default:
            out[f.name] = value
    return out
