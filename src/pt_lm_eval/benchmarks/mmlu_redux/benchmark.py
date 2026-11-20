from ...core import CompletionItem
from ..base import Benchmark, HFDataset

REPO = "fxmarty/mmlu-redux-2.0-ok"
REVISION = "8dd06e599d20d57e83b590d497c95869a2d18781"
SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


class MMLURedux(Benchmark):
    name = "mmlu_redux_generative"
    version = "1.0"
    description = "MMLU-Redux 2.0: apenas error_type=ok, 57 matérias, geração zero-shot."
    source = HFDataset(REPO, "test", SUBJECTS, REVISION)
    evaluations = ("exact_match",)
    fixed_num_fewshot = 0

    def to_item(self, row):
        subject = row["_subset"]
        prompt = row["question"].strip() + "\n"
        prompt += "\n".join(f"{c}. {o}" for c, o in zip("ABCD", row["choices"], strict=True))
        prompt += (
            "\nPlease respond with the correct letter (A, B, C or D) without any additional comments, "
            "only the correct letter:"
        )
        description = (
            "The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n"
        )
        return CompletionItem(
            id=f"{subject}/{row['_index']}",
            prompt=prompt,
            target="ABCD"[row["answer"]],
            protocol="mmlu_redux",
            subject=subject,
            description=description,
        )
