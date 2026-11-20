from ...core import MCQItem
from ..base import Benchmark, HFDataset, letter_index

REPO, REVISION = "li-lab/MMLU-ProX", "8e6106a6c6ce1c5027e66cc338143cf997b2aa09"


class MMLUProXPt(Benchmark):
    name = "mmlu_prox_pt"
    version = "1.0"
    description = "MMLU-ProX completo em português, 3–10 alternativas; few-shot do split validation."
    source = HFDataset(REPO, split="test", subset="pt", revision=REVISION)
    fewshot_source = HFDataset(REPO, split="validation", subset="pt", revision=REVISION)

    def to_item(self, row):
        # Nulos são apenas o padding final nesta revisão; N/A literal é preservado.
        options = tuple(row[f"option_{i}"] for i in range(10) if row[f"option_{i}"] is not None)
        return MCQItem(
            id=str(row["question_id"]),
            question=row["question"],
            options=options,
            gold=letter_index(row["answer"]),  # gabarito explícito em forma de letra
            subject=row["category"],
        )
