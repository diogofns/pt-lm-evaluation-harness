from ...core import MCQItem
from ..base import Benchmark, HFDataset, default_id


class BelebelePt(Benchmark):
    """Compreensão de leitura: a passagem do FLORES vai como contexto do item."""

    name = "belebele_pt"
    version = "1.0"
    description = "Belebele (por_Latn): compreensão de leitura sobre passagens do FLORES, 4 alternativas."
    source = HFDataset(
        "facebook/belebele",
        split="test",
        subset="por_Latn",
        revision="7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
    )

    def to_item(self, row: dict) -> MCQItem:
        return MCQItem(
            id=default_id(row),
            context=row["flores_passage"],
            question=row["question"],
            options=tuple(row[f"mc_answer{i}"] for i in range(1, 5)),
            gold=int(row["correct_answer_num"]) - 1,
        )
