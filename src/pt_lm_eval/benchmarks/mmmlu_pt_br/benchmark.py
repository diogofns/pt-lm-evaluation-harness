from ..base import ColumnBenchmark, Columns, HFDataset


class MMMLUPtBr(ColumnBenchmark):
    name = "mmmlu_pt_br"
    version = "1.1"
    description = "MMLU traduzido profissionalmente para o português do Brasil (OpenAI MMMLU)."
    source = HFDataset(
        "openai/MMMLU", subset="PT_BR", split="test", revision="325a01dc3e173cac1578df94120499aaca2e2504"
    )
    columns = Columns(
        question="Question",
        options=["A", "B", "C", "D"],
        answer="Answer",
        answer_is="letter",
        subject="Subject",
    )
