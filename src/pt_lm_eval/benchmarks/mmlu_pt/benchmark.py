"""MMLU-PT: questões de vestibulares, concursos e olimpíadas brasileiras.

Quatro variantes (graduação e ensino médio, completo e difícil). Cada split do
dataset é uma área (ex.: law, stem) e vira a matéria do item.
"""

from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset, MCQItem, letter_index

from .parsing import parse_alternatives

REVISIONS = {
    "mmlu-pt/mmlu-pt-undergraduate-full": "98cc293cd60c3779c2b1da1dd5abb3b47d9dee42",
    "mmlu-pt/mmlu-pt-undergraduate-hard": "7346b172a05c0286b57282fdf541ccdeba28df6a",
    "mmlu-pt/mmlu-pt-high-school-full": "b29d6ca3887d250114b543974fa8845a3143e229",
    "mmlu-pt/mmlu-pt-high-school-hard": "c40e827798003b02a7b563f2afa4f8413fa6c9f6",
}


def to_item(row: dict) -> MCQItem | None:
    alternatives = parse_alternatives(row["alternatives"])
    # ~0,5% dos itens têm alternativas vazias (em geral, questões com imagem): descartados.
    if any(not str(a).strip() for a in alternatives["text"]):
        return None
    return MCQItem(
        id=f"{row['_split']}-{row['_index']}",
        question=row["statement"],
        options=tuple(alternatives["text"]),
        gold=letter_index(row["answer"]),
        subject=row["_split"],
    )


for name, repo in [
    ("mmlu_pt_undergraduate", "mmlu-pt/mmlu-pt-undergraduate-full"),
    ("mmlu_pt_undergraduate_hard", "mmlu-pt/mmlu-pt-undergraduate-hard"),
    ("mmlu_pt_high_school", "mmlu-pt/mmlu-pt-high-school-full"),
    ("mmlu_pt_high_school_hard", "mmlu-pt/mmlu-pt-high-school-hard"),
]:
    BENCHMARKS.register(
        name,
        Benchmark(
            version="1.0",
            description=f"MMLU-PT ({repo}); matérias = splits do dataset.",
            source=HFDataset(repo, split="*", revision=REVISIONS[repo]),
            to_item=to_item,
            evaluations=("mcf", "cf", "gen"),
            card=Path(__file__).with_name("README.md"),
        ),
    )
