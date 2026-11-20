# Portions copyright (c) 2020 EleutherAI, MIT. See NOTICE and LICENSE-MIT.

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from ..core import Item, MCQItem
from .base import ColumnBenchmark


class GroupedMCQMixin:
    """Seleciona itens e exemplos por subtarefa de múltipla escolha."""

    grouped_metrics = True
    fewshot_sampler = "first_n"
    exclude_eval_from_shots = False  # usado pelo sampler first_n

    @staticmethod
    def task_key(item: MCQItem) -> str:
        return item.task_group or item.subject or "(sem matéria)"

    def select_items(self, items: list[Item], limit: int | None, seed: int) -> list[Item]:
        if limit is None:
            return items
        # Apply the limit inside each leaf task, taking its first documents.
        counts: dict[str, int] = defaultdict(int)
        selected = []
        for item in items:
            key = self.task_key(item)
            if counts[key] < limit:
                selected.append(item)
                counts[key] += 1
        return selected

    def select_shots_for_items(self, items: Sequence[Item], k: int, seed: int) -> dict[str, list[Item]]:
        if k <= 0:
            return {item.id: [] for item in items}
        if self.fewshot_holdout:
            remaining, reserved = self._partition_fewshot(self.data.items, k)
            allowed = {item.id for item in remaining}
            if any(item.id not in allowed for item in items):
                raise ValueError(f"{self.name}: questão reservada para few-shot não pode entrar no eval")
            return {item.id: reserved[self.task_key(item)] for item in items}
        pools: dict[str, list[MCQItem]] = defaultdict(list)
        for item in self.data.pool:
            pools[self.task_key(item)].append(item)
        rngs: dict[str, random.Random] = {}
        result = {}
        for item in items:
            key = self.task_key(item)
            pool = pools[key]
            if self.fewshot_sampler == "first_n":
                candidates = [p for p in pool if not self.exclude_eval_from_shots or not _same_doc(p, item)]
                if k > len(candidates):
                    raise ValueError(f"{self.name}/{key}: pediu {k} few-shot, mas há só {len(candidates)}")
                shots = candidates[:k]
            else:
                rnd = rngs.setdefault(key, random.Random(seed))
                shots = rnd.sample(pool, k)
            result[item.id] = shots
        return result

    def select_shots(self, item: Item, k: int, seed: int) -> list[Item]:
        if self.fewshot_holdout:
            return self.select_shots_for_items([item], k, seed)[item.id]
        # Mantém a mesma posição do sorteio numa execução completa.
        prefix = []
        for doc in self.data.items:
            if self.task_key(doc) == self.task_key(item):
                prefix.append(doc)
            if doc.id == item.id:
                break
        if not any(doc.id == item.id for doc in prefix):
            raise ValueError(f"{self.name}: item {item.id} ausente do conjunto de avaliação")
        return self.select_shots_for_items(prefix, k, seed)[item.id]

    def _partition_fewshot(
        self, items: list[MCQItem], k: int
    ) -> tuple[list[MCQItem], dict[str, list[MCQItem]]]:
        areas: dict[str, list[MCQItem]] = defaultdict(list)
        for item in items:
            areas[self.task_key(item)].append(item)
        reserved = {}
        for area, pool in areas.items():
            if len(pool) <= k:
                raise ValueError(f"{self.name}/{area}: {k} few-shot não deixam questões para avaliar")
            reserved[area] = pool[:k]
        # Exclude duplicate questions too, even if they have another ID/area/answer.
        keys = {_question_key(item) for shots in reserved.values() for item in shots}
        remaining = [item for item in items if _question_key(item) not in keys]
        remaining_areas = {self.task_key(item) for item in remaining}
        if missing := areas.keys() - remaining_areas:
            raise ValueError(
                f"{self.name}: não restam questões após reservar few-shot nas áreas {sorted(missing)}"
            )
        return remaining, reserved

    def select_for_evaluation(
        self, items: list[Item], k: int, limit: int | None, seed: int, *, holdout: bool = False
    ) -> tuple[list[Item], dict[str, list[Item]]]:
        holdout = holdout or self.fewshot_holdout
        if holdout and self.fewshot_source is not None:
            raise ValueError(f"{self.name}: fewshot_holdout exige exemplos do próprio teste")
        if not holdout or k == 0:
            return super().select_for_evaluation(items, k, limit, seed)
        remaining, reserved = self._partition_fewshot(items, k)
        selected = self.select_items(remaining, limit, seed)
        return selected, {item.id: reserved[self.task_key(item)] for item in selected}


class GroupedMCQBenchmark(GroupedMCQMixin, ColumnBenchmark):
    """Grouped evaluation protocol for column datasets."""


def _same_doc(left: MCQItem, right: MCQItem) -> bool:
    # Dataset documents have no synthetic split/index IDs in the original document records.
    return (left.question, left.options, left.gold, left.subject, left.task_group) == (
        right.question,
        right.options,
        right.gold,
        right.subject,
        right.task_group,
    )


def _question_key(item: MCQItem) -> tuple[str, tuple[str, ...]]:
    return item.question.strip(), tuple(option.strip() for option in item.options)
