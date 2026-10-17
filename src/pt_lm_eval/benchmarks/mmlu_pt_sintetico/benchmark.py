from ..base import ColumnBenchmark, Columns, HFDataset


class MMLUPtSintetico(ColumnBenchmark):
    name = "mmlu_pt_sintetico"
    version = "1.1"
    description = "Questões no estilo MMLU geradas por Qwen3-235B-A22B em português."
    source = HFDataset(
        "cemig-ceia/benchmark_Qwen3-235B-A22B_MMLU_v2",
        split="train",
        revision="831ba6bbae5b6057fbc3cae40e13eae7b55ebb26",
    )
    columns = Columns(
        question="question", options="choices.text", answer="answer", answer_is="letter", subject="subject"
    )
