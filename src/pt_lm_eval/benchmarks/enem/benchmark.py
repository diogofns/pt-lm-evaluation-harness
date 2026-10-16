"""ENEM 2022–2024, como no lighteval (enem_adapter): questões anuladas saem e as
descrições das figuras entram no lugar de cada `[[placeholder]]` do enunciado."""

from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset, MCQItem, letter_index

YEARS = ("2022", "2023", "2024")


def to_item(row: dict) -> MCQItem | None:
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


BENCHMARKS.register(
    "enem",
    Benchmark(
        version="1.0",
        description="ENEM 2022–2024 (5 alternativas); matéria = ano da prova.",
        source=HFDataset(
            "maritaca-ai/enem",
            split="train",
            subset=YEARS,
            revision="b2a7257356b0f56b740d57f23f8a575542ff6c0c",
        ),
        to_item=to_item,
        evaluations=("mcf", "cf", "gen"),
        card=Path(__file__).with_name("README.md"),
    ),
)
