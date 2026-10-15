"""Agregação: média, erro-padrão e média entre matérias."""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean_stderr(values: Sequence[float]) -> tuple[float, float]:
    """Média e erro-padrão da média (desvio-padrão amostral / √n), como no lm-eval."""
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    mean = sum(values) / n
    if n == 1:
        return mean, math.nan
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def macro_mean_stderr(groups: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Média simples entre grupos (ex.: matérias) e seu erro-padrão.

    Var(média) = Σ s²/nᵢ / k², com s² a variância de todos os itens juntos. Usar a
    variância agrupada evita que grupos com 1 item contem como variância zero.
    """
    groups = [g for g in groups if g]
    k = len(groups)
    if k == 0:
        return math.nan, math.nan
    mean = sum(sum(g) / len(g) for g in groups) / k
    values = [v for g in groups for v in g]
    if len(values) < 2:
        return mean, math.nan
    overall = sum(values) / len(values)
    pooled_var = sum((v - overall) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(pooled_var * sum(1 / len(g) for g in groups)) / k
