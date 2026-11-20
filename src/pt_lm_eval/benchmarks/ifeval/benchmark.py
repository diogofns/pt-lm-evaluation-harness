from ...core import InstructionItem
from ..base import Benchmark, HFDataset


class IFEval(Benchmark):
    name = "ifeval"
    version = "1.0"
    description = "Google IFEval original em inglês; 0-shot e 1280 tokens."
    source = HFDataset("google/IFEval", "train", revision="966cd89545d6b6acfd7638bc708b98261ca58e84")
    evaluations = ("ifeval_en",)
    fixed_num_fewshot = 0

    def to_item(self, row):
        return InstructionItem(
            id=str(row["key"]),
            prompt=row["prompt"],
            instruction_ids=row["instruction_id_list"],
            kwargs=row["kwargs"],
        )
