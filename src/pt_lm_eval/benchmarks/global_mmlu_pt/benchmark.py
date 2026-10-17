from ..base import ColumnBenchmark, Columns, HFDataset

REPO, REVISION = "CohereLabs/Global-MMLU", "0e619dbeb34206cd48705a1a0ea7fb21cae09993"


class GlobalMMLUPt(ColumnBenchmark):
    name = "global_mmlu_pt"
    version = "1.0"
    description = "Global-MMLU em português (MMLU traduzido e revisado); exemplos few-shot do split dev."
    source = HFDataset(REPO, split="test", subset="pt", revision=REVISION)
    fewshot_source = HFDataset(REPO, split="dev", subset="pt", revision=REVISION)
    columns = Columns(
        id="sample_id",
        question="question",
        options=["option_a", "option_b", "option_c", "option_d"],
        answer="answer",
        answer_is="letter",
        subject="subject",
    )
