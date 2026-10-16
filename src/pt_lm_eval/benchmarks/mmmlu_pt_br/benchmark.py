from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset, columns

BENCHMARKS.register(
    "mmmlu_pt_br",
    Benchmark(
        version="1.1",
        description="MMLU traduzido profissionalmente para o português do Brasil (OpenAI MMMLU).",
        source=HFDataset(
            "openai/MMMLU",
            subset="PT_BR",
            split="test",
            revision="325a01dc3e173cac1578df94120499aaca2e2504",
        ),
        to_item=columns(
            question="Question",
            options=["A", "B", "C", "D"],
            answer="Answer",
            answer_is="letter",
            subject="Subject",
        ),
        evaluations=("mcf", "cf", "gen"),
        card=Path(__file__).with_name("README.md"),
    ),
)
