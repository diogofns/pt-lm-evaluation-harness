from ..base import Columns, HFDataset
from ..mcq_protocol import GroupedMCQBenchmark


class MathPtBr(GroupedMCQBenchmark):
    fewshot_holdout = True
    name = "math_pt_br"
    version = "1.2"
    evaluations = ("mcf", "gen")
    description = "MATH-PT: 638 questões brasileiras de matemática, subset ptbr_multiple_choice."
    source = HFDataset(
        "tiagoteixeira03/MATH-PT",
        subset="ptbr_multiple_choice",
        split="test",
        revision="11c44f85c32d5d6b3dce76d163201ae4c6abd4b2",
    )
    columns = Columns(
        question="problem",
        options=[f"choices.{letter}" for letter in "ABCDE"],
        answer="answer",
        answer_is="letter",
        subject="level",
        protocol="global_mmlu",
    )
