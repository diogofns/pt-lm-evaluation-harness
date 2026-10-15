from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset, columns

BENCHMARKS.register(
    "mmlu_pt_sintetico",
    Benchmark(
        version="1.0",
        description="Questões no estilo MMLU geradas por Qwen3-235B-A22B em português.",
        source=HFDataset(
            "cemig-ceia/benchmark_Qwen3-235B-A22B_MMLU_v2",
            split="train",
            revision="831ba6bbae5b6057fbc3cae40e13eae7b55ebb26",
        ),
        to_item=columns(question="question", options="choices.text", answer="answer", subject="subject"),
        evaluations=("mcf", "cf", "gen"),
        card=Path(__file__).with_name("README.md"),
    ),
)
