from ...core import MCQItem
from ..base import Benchmark, HFDataset, letter_index


class Enem(Benchmark):
    """ENEM 2022–2024, como no lighteval (enem_adapter): anuladas saem e a descrição de cada
    figura entra no lugar do `[[placeholder]]` do enunciado."""

    name = "enem"
    version = "1.0"
    description = "ENEM 2022–2024 (5 alternativas); matéria = ano da prova."
    source = HFDataset(
        "maritaca-ai/enem",
        split="train",
        subset=("2022", "2023", "2024"),
        revision="b2a7257356b0f56b740d57f23f8a575542ff6c0c",
    )

    def to_item(self, row: dict) -> MCQItem | None:
        if row["label"] == "Anulado":
            return None
        question = row["question"]
        for description in row["description"]:
            question = question.replace("[[placeholder]]", description, 1)
        return MCQItem(
            id=f"{row['exam']}-{row['id']}",
            question=question,
            options=tuple(row["alternatives"]),
            gold=letter_index(row["label"]),
            subject=row["exam"],
        )
