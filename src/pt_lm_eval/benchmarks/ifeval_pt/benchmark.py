from ...core import InstructionItem
from ..base import Benchmark, HFDataset


class IFEvalPt(Benchmark):
    name = "ifeval_pt"
    version = "1.0"
    description = "Portuguese-IFEval: 535 prompts, 1.238 instruções verificáveis (6 do português)."
    source = HFDataset(
        "gap-ufg/Portuguese-IFEval", split="test", revision="0ff3fe8d82a5cdb43153f86b301b2c9d5c6ea098"
    )
    evaluations = ("ifeval",)

    def to_item(self, row: dict) -> InstructionItem:
        return InstructionItem(
            id=str(row["key"]),
            prompt=row["prompt"],
            instruction_ids=row["instruction_id_list"],
            kwargs=row["kwargs"],
        )
