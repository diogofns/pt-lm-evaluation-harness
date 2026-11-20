from ..base import Columns, HFDataset
from ..mcq_protocol import GroupedMCQBenchmark


class MMMLUPtBr(GroupedMCQBenchmark):
    name = "mmmlu_pt_br"
    version = "2.3"
    evaluations = ("mcf", "gen")
    description = "MMLU traduzido profissionalmente para o português do Brasil (OpenAI MMMLU)."
    exclude_eval_from_shots = True
    source = HFDataset(
        "openai/MMMLU", subset="PT_BR", split="test", revision="325a01dc3e173cac1578df94120499aaca2e2504"
    )
    columns = Columns(
        question="Question",
        options=["A", "B", "C", "D"],
        answer="Answer",
        answer_is="letter",
        subject="Subject",
        protocol="mmmlu",
    )
