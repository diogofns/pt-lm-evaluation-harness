import asyncio
from dataclasses import replace

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS, MCQItem
from pt_lm_eval.benchmarks.base import HFDataset, Loaded
from pt_lm_eval.core import GenerateResponse
from pt_lm_eval.evaluations.prompts import GEN_INSTRUCTION
from pt_lm_eval.experiment import Experiment, parse_config

NAMES = [
    "mmlu_pt_high_school",
    "mmlu_pt_undergraduate",
    "global_mmlu_pt",
    "mmmlu_pt_br",
    "include_pt",
    "math_pt_br",
]


def fixture_items(bench):
    protocol = (
        "mmmlu" if bench.name == "mmmlu_pt_br" else "include" if bench.name == "include_pt" else "global_mmlu"
    )
    return [
        MCQItem(
            id=f"{subject}-{i}",
            question=f"Questão {subject}/{i}?",
            options=("primeira opção", "segunda opção", "terceira opção", "quarta opção"),
            gold=i % 4,
            subject=subject,
            protocol=protocol,
            prompt_description=(
                "A questão de múltipla escolha a seguir é sobre ciências aplicadas."
                if protocol == "include"
                else None
            ),
        )
        for subject in ("anatomy", "abstract_algebra")
        for i in range(4)
    ]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shots", [0, 1])
def test_generation_preserves_mcf_item_and_fewshot_selection(monkeypatch, name, shots):
    bench = BENCHMARKS.get(name)
    items = fixture_items(bench)
    pool = [replace(item, id="dev-" + item.id) for item in items] if bench.fewshot_source else items
    monkeypatch.setattr(bench, "data", Loaded(items, pool, 0))
    cfg = parse_config(
        {
            "experiment": "gen",
            "num_fewshot": shots,
            "limit": 2,
            "models": [{"name": "m", "type": "instruct"}],
            "benchmarks": [{"name": name, "evaluations": ["mcf", "gen"]}],
        }
    )
    experiment = Experiment(cfg)
    mcf_step, gen_step = cfg.plan()
    mcf_items, mcf_shots = experiment.select(mcf_step)
    gen_items, gen_shots = experiment.select(gen_step)
    assert gen_items == mcf_items and gen_shots == mcf_shots
    for item in gen_items:
        (request,) = gen_step.build(item, gen_shots[item.id])
        assert request.kind == "generate"
        assert GEN_INSTRUCTION in request.prompt
        assert request.prompt.endswith("\nResposta:")
        assert "The following" not in request.prompt and "\nAnswer:" not in request.prompt
        if name == "mmmlu_pt_br":
            assert request.prompt.startswith("A seguir estão questões de múltipla escolha")
            assert request.prompt.count("em português brasileiro.") == 1
        if name == "include_pt":
            assert request.prompt.startswith(
                "A questão de múltipla escolha a seguir é sobre ciências aplicadas.\n\n"
            )
        for shot in gen_shots[item.id]:
            assert f"{shot.question}\n" in request.prompt
            assert f"Resposta: {shot.letters[shot.gold]}" in request.prompt


@pytest.mark.parametrize("name", ["mmmlu_pt_br", "include_pt"])
def test_generation_multiturn_preserves_translated_header(name):
    bench = BENCHMARKS.get(name)
    shot, item = fixture_items(bench)[:2]
    (request,) = EVALUATIONS.get("gen").build(item, [shot], multiturn=True)
    assert [m["role"] for m in request.messages] == ["system", "user", "assistant", "user"]
    assert request.messages[0]["content"].startswith("A seguir" if name == "mmmlu_pt_br" else "A questão")
    assert request.messages[2]["content"] == "Resposta: A"
    assert GEN_INSTRUCTION in request.messages[-1]["content"]


def test_generation_only_run_never_requests_loglikelihood(monkeypatch, tmp_path):
    bench = BENCHMARKS.get("mmlu_pt_high_school")
    items = fixture_items(bench)
    monkeypatch.setattr(bench, "data", Loaded(items, items, 0))
    monkeypatch.setattr(HFDataset, "describe", lambda self: "offline-fixture")
    cfg = parse_config(
        {
            "experiment": "generation",
            "output_dir": str(tmp_path),
            "num_fewshot": 1,
            "models": [{"name": "m", "type": "instruct"}],
            "benchmarks": [{"name": bench.name, "evaluations": ["gen"]}],
        }
    )

    class Client:
        async def generate(self, requests):
            return [GenerateResponse("Resposta: B", finish_reason="stop") for request in requests]

        async def loglikelihood(self, requests):
            raise AssertionError("A execução generativa não deve repetir o MCF")

    result = asyncio.run(Experiment(cfg).run_step(cfg.plan()[0], Client()))
    assert result["evaluation"]["name"] == "gen"
    assert result["benchmark"]["n_evaluated"] == 6
    assert result["metrics"]["extracted"] == 1
    assert result["metrics"]["acc"] == pytest.approx(1 / 3)
    assert set(result["evaluation"]["reserved_fewshot_ids"]) == {"anatomy-0", "abstract_algebra-0"}
