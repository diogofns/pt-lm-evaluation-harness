from ..base import Columns, HFDataset
from ..mcq_protocol import GroupedMCQBenchmark


class GlobalPIQANonparallelPtBr(GroupedMCQBenchmark):
    name = "global_piqa_nonparallel_pt_br"
    version = "2.1"
    evaluations = ("cf", "gen")
    fewshot_sampler = "default"
    base_generation_evaluations = ("gen",)
    description = "Global PIQA nonparallel em português brasileiro, 100 questões de senso comum cultural."
    source = HFDataset(
        "mrlbenchmarks/global-piqa-nonparallel",
        split="test",
        subset="por_latn_braz",
        revision="6777742fa3634c0583cda3b7f8a482ea7b1b0937",
    )
    columns = Columns(
        id="example_id",
        question="prompt",
        options=["solution0", "solution1"],
        answer="label",
        answer_is="index",
        protocol="global_piqa",
    )
