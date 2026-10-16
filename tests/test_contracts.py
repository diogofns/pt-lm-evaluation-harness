"""Todo plugin registrado passa pelos testes de contrato do seu protocolo."""

from __future__ import annotations

from pathlib import Path

import pytest

from pt_lm_eval import BACKENDS, BENCHMARKS, EVALUATIONS, REPORTERS, InstructionItem, MCQItem
from pt_lm_eval.runner.config import ModelConfig
from pt_lm_eval.testing import check_benchmark, check_evaluation

ITEM = MCQItem(id="q", question="Quanto é 2 + 2?", options=("3", "4", "5", "6"), gold=1, subject="mat")
# Um exemplo por tipo de item; cada avaliação é testada com o tipo que declara.
EXAMPLES = {
    MCQItem: ITEM,
    InstructionItem: InstructionItem(
        id="i", prompt="Escreva sem vírgulas.", instruction_ids=("pt:punctuation:no_comma",), kwargs=({},)
    ),
}


@pytest.mark.parametrize("name", EVALUATIONS.names())
def test_evaluation_contract(name):
    check_evaluation(name, EXAMPLES[EVALUATIONS.get(name).item_type])


@pytest.mark.parametrize("name", BACKENDS.names())
def test_backend_factory_contract(name):
    factory = BACKENDS.get(name)
    for model_type in ("base", "instruct", "api"):
        model = ModelConfig(
            name="m", type=model_type, base_url="http://x/v1" if model_type == "api" else None
        )
        capabilities = factory.capabilities(model)
        backend = factory.create(model, "http://x/v1", max_length=None)
        assert backend.capabilities == capabilities


@pytest.mark.parametrize("name", REPORTERS.names())
def test_reporter_contract(name, tmp_path):
    result = {
        "model": {"name": "m"},
        "benchmark": {"name": "b", "version": "1.0"},
        "evaluation": {"name": "mcf"},
        "metrics": {"acc": 0.5, "acc_stderr": None},
    }
    path = REPORTERS.get(name).write([result], tmp_path, "exp")
    assert path.exists()


@pytest.mark.parametrize("name", BENCHMARKS.names())
def test_benchmark_declares_known_evaluations_and_card(name):
    bench = BENCHMARKS.get(name)
    assert bench.version
    assert set(bench.evaluations) <= set(EVALUATIONS.names())
    assert bench.card is not None and bench.card.exists()


@pytest.mark.network
@pytest.mark.parametrize("name", BENCHMARKS.names())
def test_benchmark_contract(name):
    # Com a trava: prompt mudou sem a versão mudar = falha (como o CI dos benchmarks).
    check_benchmark(name, lock=Path(__file__).parents[1] / "benchmarks.lock.json")
