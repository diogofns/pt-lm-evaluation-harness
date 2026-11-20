import random

from ...core import CompletionItem
from ..base import Benchmark, HFDataset

REPO = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"


class GSM8K(Benchmark):
    name = "gsm8k"
    version = "1.0"
    description = "GSM8K original: geração 5-shot, strict-match/flexible-extract; sem gsm8k_cot."
    source = HFDataset(REPO, "test", "main", REVISION)
    fewshot_source = HFDataset(REPO, "train", "main", REVISION)
    evaluations = ("exact_match",)
    default_num_fewshot = 5

    def to_item(self, row):
        return CompletionItem(
            id=f"{row['_split']}/{row['_index']}",
            prompt=f"Question: {row['question']}\nAnswer:",
            target=row["answer"],
            protocol="gsm8k",
        )

    def select_shots(self, item, k, seed):
        if k == 0:
            return []
        # Advance the same random.Random(seed) across test docs as ContextSampler.
        # Cache by k/seed; replaying full canonical order also makes resume/limit stable.
        if not hasattr(self, "_shot_cache"):
            self._shot_cache = {}
        key = (k, seed)
        if key not in self._shot_cache:
            rng = random.Random(seed)
            pool = self.data.pool
            self._shot_cache[key] = {it.id: rng.sample(pool, k) for it in self.data.items}
        return self._shot_cache[key][item.id]
