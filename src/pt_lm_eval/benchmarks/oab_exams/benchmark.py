from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset, columns

BENCHMARKS.register(
    "oab_exams",
    Benchmark(
        version="1.0",
        description="Exame da OAB, 2010–2018 (4 alternativas); matéria = área do direito, quando há.",
        source=HFDataset(
            "eduagarcia/oab_exams", split="train", revision="fe810531d30e7799dd83c2e5be302ca2f2bdf930"
        ),
        to_item=columns(
            id="id",
            question="question",
            options="choices.text",
            answer="answerKey",
            answer_is="letter",
            subject="question_type",
        ),
        evaluations=("mcf", "cf", "gen"),
        card=Path(__file__).with_name("README.md"),
    ),
)
