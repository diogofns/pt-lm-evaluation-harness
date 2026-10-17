"""Toda avaliação e todo benchmark registrados cumprem o contrato."""

from __future__ import annotations

from pathlib import Path

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS, InstructionItem, MCQItem
from pt_lm_eval.cli import card_problems, check_benchmarks
from pt_lm_eval.core import GenerateResponse, LoglikelihoodResponse

# Um exemplo por tipo de item; cada avaliação é testada com o tipo que declara.
EXAMPLES = {
    MCQItem: MCQItem(id="q", question="Quanto é 2 + 2?", options=("3", "4", "5", "6"), gold=1, subject="mat"),
    InstructionItem: InstructionItem(
        id="i", prompt="Escreva sem vírgulas.", instruction_ids=("pt:punctuation:no_comma",), kwargs=({},)
    ),
}


@pytest.mark.parametrize("name", EVALUATIONS.names())
def test_evaluation_builds_declared_requests_and_scores_floats(name):
    ev = EVALUATIONS.get(name)
    item = EXAMPLES[ev.item_type]
    for shots in ([], [item]):
        requests = ev.build(item, shots)
        assert requests and {r.kind for r in requests} <= ev.needs
        fake = [
            LoglikelihoodResponse(-1.0, False, 1)
            if r.kind == "loglikelihood"
            else GenerateResponse("Resposta: A")
            for r in requests
        ]
        metrics = ev.score(item, fake).metrics
        assert metrics and all(isinstance(v, float) for v in metrics.values())


@pytest.mark.parametrize("name", BENCHMARKS.names())
def test_benchmark_declares_known_evaluations_and_a_complete_card(name):
    bench = BENCHMARKS.get(name)
    assert bench.version and set(bench.evaluations) <= set(EVALUATIONS.names())
    assert not card_problems(bench.card)


@pytest.mark.network
@pytest.mark.parametrize("name", BENCHMARKS.names())
def test_benchmark_matches_the_prompt_lock(name):
    # Prompt mudou sem a versão mudar = falha (como o CI dos benchmarks).
    assert check_benchmarks([name], Path(__file__).parents[1] / "benchmarks.lock.json")
