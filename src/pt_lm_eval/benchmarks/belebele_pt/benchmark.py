"""Belebele em português: compreensão de leitura; a passagem do FLORES vai como contexto."""

from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset, MCQItem
from pt_lm_eval.data.columns import default_id


def to_item(row: dict) -> MCQItem:
    return MCQItem(
        id=default_id(row),
        context=row["flores_passage"],
        question=row["question"],
        options=tuple(row[f"mc_answer{i}"] for i in range(1, 5)),
        gold=int(row["correct_answer_num"]) - 1,
    )


BENCHMARKS.register(
    "belebele_pt",
    Benchmark(
        version="1.0",
        description="Belebele (por_Latn): compreensão de leitura sobre passagens do FLORES, 4 alternativas.",
        source=HFDataset(
            "facebook/belebele",
            split="test",
            subset="por_Latn",
            revision="7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
        ),
        to_item=to_item,
        evaluations=("mcf", "cf", "gen"),
        card=Path(__file__).with_name("README.md"),
    ),
)
