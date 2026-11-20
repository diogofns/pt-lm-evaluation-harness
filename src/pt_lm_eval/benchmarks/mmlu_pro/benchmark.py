from ...core import CompletionItem
from ..base import Benchmark, HFDataset

REPO = "TIGER-Lab/MMLU-Pro"
REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"


class MMLUPro(Benchmark):
    name = "mmlu_pro"
    version = "1.0"
    description = "MMLU-Pro original: 5-shot CoT por matéria, geração e exact_match/custom-extract."
    source = HFDataset(REPO, "test", revision=REVISION)
    fewshot_source = HFDataset(REPO, "validation", revision=REVISION)
    evaluations = ("exact_match",)
    default_num_fewshot = 5
    fewshot_strategy = "first_n"

    def to_item(self, row):
        prompt = f"Question:\n{row['question']}\nOptions:\n"
        prompt += "".join(
            f"{letter}. {option.strip()}\n"
            for letter, option in zip("ABCDEFGHIJ", row["options"], strict=False)
        )
        prompt += "Answer: Let's think step by step."
        target = (
            row["cot_content"].replace("A: Let's think step by step. ", "")
            if row["_split"] == "validation"
            else row["answer"]
        )
        return CompletionItem(
            id=f"{row['_split']}/{row.get('question_id', row['_index'])}",
            prompt=prompt,
            target=target,
            protocol="mmlu_pro",
            subject=row["category"],
            description=(
                "The following are multiple choice questions (with answers) about "
                f"{row['category']}. Think step by step and then finish your answer with "
                '"the answer is (X)" where X is the correct letter choice.\n'
            ),
        )
