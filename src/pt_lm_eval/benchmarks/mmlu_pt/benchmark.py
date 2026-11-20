from ...core import MCQItem
from ..base import Benchmark, HFDataset, letter_index
from ..mcq_protocol import GroupedMCQMixin
from .parsing import parse_alternatives


class MMLUPt(GroupedMCQMixin, Benchmark):
    fewshot_holdout = True
    """Conversão comum às quatro variantes (sem `name`, não é registrada)."""

    evaluations = ("mcf", "gen")

    def to_item(self, row: dict) -> MCQItem | None:
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
            protocol="global_mmlu",
        )


class MMLUPtUndergraduate(MMLUPt):
    name = "mmlu_pt_undergraduate"
    version = "2.3"
    description = "MMLU-PT (mmlu-pt/mmlu-pt-undergraduate-full); matérias = splits do dataset."
    source = HFDataset(
        "mmlu-pt/mmlu-pt-undergraduate-full", split="*", revision="98cc293cd60c3779c2b1da1dd5abb3b47d9dee42"
    )


class MMLUPtUndergraduateHard(MMLUPt):
    name = "mmlu_pt_undergraduate_hard"
    version = "2.3"
    description = "MMLU-PT (mmlu-pt/mmlu-pt-undergraduate-hard); matérias = splits do dataset."
    source = HFDataset(
        "mmlu-pt/mmlu-pt-undergraduate-hard", split="*", revision="7346b172a05c0286b57282fdf541ccdeba28df6a"
    )


class MMLUPtHighSchool(MMLUPt):
    name = "mmlu_pt_high_school"
    version = "2.3"
    description = "MMLU-PT (mmlu-pt/mmlu-pt-high-school-full); matérias = splits do dataset."
    source = HFDataset(
        "mmlu-pt/mmlu-pt-high-school-full", split="*", revision="b29d6ca3887d250114b543974fa8845a3143e229"
    )


class MMLUPtHighSchoolHard(MMLUPt):
    name = "mmlu_pt_high_school_hard"
    version = "2.3"
    description = "MMLU-PT (mmlu-pt/mmlu-pt-high-school-hard); matérias = splits do dataset."
    source = HFDataset(
        "mmlu-pt/mmlu-pt-high-school-hard", split="*", revision="c40e827798003b02a7b563f2afa4f8413fa6c9f6"
    )
