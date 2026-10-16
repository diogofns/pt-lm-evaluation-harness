from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset, columns

REPO, REVISION = "CohereLabs/Global-MMLU", "0e619dbeb34206cd48705a1a0ea7fb21cae09993"

BENCHMARKS.register(
    "global_mmlu_pt",
    Benchmark(
        version="1.0",
        description="Global-MMLU em português (MMLU traduzido e revisado); exemplos few-shot do split dev.",
        source=HFDataset(REPO, split="test", subset="pt", revision=REVISION),
        fewshot_source=HFDataset(REPO, split="dev", subset="pt", revision=REVISION),
        to_item=columns(
            id="sample_id",
            question="question",
            options=["option_a", "option_b", "option_c", "option_d"],
            answer="answer",
            answer_is="letter",
            subject="subject",
        ),
        evaluations=("mcf", "cf", "gen"),
        card=Path(__file__).with_name("README.md"),
    ),
)
