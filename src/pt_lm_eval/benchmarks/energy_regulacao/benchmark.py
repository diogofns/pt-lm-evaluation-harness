from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset, columns

BENCHMARKS.register(
    "energy_regulacao",
    Benchmark(
        version="1.0",
        description="Regulação do setor elétrico brasileiro (PRORET/ANEEL), 5 alternativas.",
        source=HFDataset(
            "cemig-ceia/energy-eval", split="train", revision="cac76ae9456fc3c9354d761880bc3c3a5b54d1e1"
        ),
        to_item=columns(
            id="id", question="question", options="choices.text", answer="answerKey", answer_is="letter"
        ),
        evaluations=("mcf", "cf", "gen"),
        card=Path(__file__).with_name("README.md"),
    ),
)
