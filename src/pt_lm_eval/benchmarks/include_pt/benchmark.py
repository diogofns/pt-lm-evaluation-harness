from dataclasses import replace
from typing import ClassVar

from ...core import MCQItem
from ..base import Columns, HFDataset
from ..mcq_protocol import GroupedMCQBenchmark


class IncludePt(GroupedMCQBenchmark):
    name = "include_pt"
    version = "2.3"
    evaluations = ("mcf", "gen")
    fewshot_sampler = "default"
    description = "INCLUDE em português: questões acadêmicas e profissionais de múltipla escolha."
    source = HFDataset(
        "CohereLabs/include-base-44",
        subset="Portuguese",
        split="test",
        revision="d2e1f6015f67a43c02a9a68db98e2298e2d6a660",
    )
    columns = Columns(
        question="question",
        options=["option_a", "option_b", "option_c", "option_d"],
        answer="answer",
        answer_is="index",
        subject="subject",
        protocol="include",
    )

    descriptions: ClassVar[dict[str, str]] = {
        domain: f"A questão de múltipla escolha a seguir é sobre {label}."
        for domain, label in {
            "Applied Science": "ciências aplicadas",
            "Arts & Humanities": "artes e humanidades",
            "Business & Commerce": "negócios e comércio",
            "Health oriented education": "formação em saúde",
            "Social Science": "ciências sociais",
            "STEM": "ciências, tecnologia, engenharia e matemática",
        }.items()
    }

    def to_item(self, row: dict) -> MCQItem:
        item = self.columns.to_item(row)
        try:
            description = self.descriptions[row["domain"]]
        except KeyError as exc:
            raise ValueError(f"INCLUDE Portuguese: domínio desconhecido {row['domain']!r}") from exc
        return replace(item, prompt_description=description, task_group=row["domain"])
