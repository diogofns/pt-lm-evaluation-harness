from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset
from pt_lm_eval.core.types import InstructionItem


def to_item(row: dict) -> InstructionItem:
    return InstructionItem(
        id=str(row["key"]),
        prompt=row["prompt"],
        instruction_ids=row["instruction_id_list"],
        kwargs=row["kwargs"],
    )


BENCHMARKS.register(
    "ifeval_pt",
    Benchmark(
        version="1.0",
        description="Portuguese-IFEval: 535 prompts, 1.238 instruções verificáveis (6 do português).",
        source=HFDataset(
            "gap-ufg/Portuguese-IFEval", split="test", revision="0ff3fe8d82a5cdb43153f86b301b2c9d5c6ea098"
        ),
        to_item=to_item,
        evaluations=("ifeval",),
        card=Path(__file__).with_name("README.md"),
    ),
)
