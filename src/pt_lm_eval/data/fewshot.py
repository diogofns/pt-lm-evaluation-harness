"""Seleção determinística de exemplos few-shot."""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..core.types import Item


def select_shots(
    item: Item,
    pool: Sequence[Item],
    k: int,
    *,
    seed: int = 1234,
    strategy: str = "random",
    exclude_self: bool = True,
) -> list[Item]:
    """Escolhe `k` exemplos para `item`.

    Prefere exemplos da mesma matéria quando há suficientes. Com `random`, a
    escolha depende só de (seed, item.id), então é estável entre execuções e
    independente da ordem de processamento. `first_n` pega os primeiros do pool.
    """
    if k <= 0:
        return []
    candidates = [p for p in pool if not (exclude_self and p.id == item.id)]
    subject = getattr(item, "subject", None)
    if subject is not None:
        same = [p for p in candidates if getattr(p, "subject", None) == subject]
        if len(same) >= k:
            candidates = same
    if strategy == "first_n":
        return list(candidates[:k])
    if strategy != "random":
        raise ValueError(f"estratégia de few-shot desconhecida: {strategy}")
    return random.Random(f"{seed}:{item.id}").sample(candidates, min(k, len(candidates)))
