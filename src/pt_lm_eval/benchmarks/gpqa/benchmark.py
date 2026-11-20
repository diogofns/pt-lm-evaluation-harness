import random
from functools import cached_property

from ...core import MCQItem
from ..base import Benchmark, HFDataset, Loaded

REPO = "Idavidrein/gpqa"
REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"


def preprocess(text):
    if text is None:
        return " "
    return text.strip().replace(" [title]", ". ").replace("  ", " ")


class _GPQA(Benchmark):
    evaluations = ("gpqa_mcf",)
    version = "1.0"
    default_num_fewshot = 0
    fixed_num_fewshot = 0
    description = "GPQA em inglês: likelihood de (A)/(B)/(C)/(D), protocolo zero-shot."

    @cached_property
    def data(self):
        # Shuffle options deterministically with Python random seed 0.
        # A private generator makes the standalone task invariant to the model/task order.
        rng = random.Random(0)
        items = []
        for row in self.source.load():
            choices = [preprocess(row[f"Incorrect Answer {i}"]) for i in (1, 2, 3)]
            correct = preprocess(row["Correct Answer"])
            choices.append(correct)
            rng.shuffle(choices)
            items.append(self.to_item({**row, "_choices": choices, "_gold": choices.index(correct)}))
        return Loaded(items, items, 0)

    def to_item(self, row):
        return MCQItem(
            id=f"{row['_subset']}/{row['_index']}",
            question=row["Question"],
            options=tuple(row["_choices"]),
            gold=row["_gold"],
            protocol="gpqa",
        )


class GPQAMain(_GPQA):
    name = "gpqa_main_zeroshot"
    source = HFDataset(REPO, "train", "gpqa_main", REVISION)


class GPQADiamond(_GPQA):
    name = "gpqa_diamond_zeroshot"
    source = HFDataset(REPO, "train", "gpqa_diamond", REVISION)


class GPQAExtended(_GPQA):
    name = "gpqa_extended_zeroshot"
    source = HFDataset(REPO, "train", "gpqa_extended", REVISION)
