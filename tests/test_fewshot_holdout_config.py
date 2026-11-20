import asyncio
from dataclasses import replace

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS
from pt_lm_eval.benchmarks.base import HFDataset, Loaded
from pt_lm_eval.core import GenerateResponse, MCQItem
from pt_lm_eval.experiment.config import parse_config
from pt_lm_eval.experiment.runner import Experiment


@pytest.mark.parametrize("name", ["mmmlu_pt_br", "include_pt", "global_piqa_nonparallel_pt_br"])
def test_config_reserves_five_shots_and_duplicates_outside_scoring(monkeypatch, tmp_path, name):
    bench = BENCHMARKS.get(name)
    items = [
        MCQItem(
            id=f"{group}-{i}",
            question=f"Questão {group}/{i}",
            options=("Sim", "Não"),
            gold=0,
            subject=group,
            task_group=group,
        )
        for group in ["a", "b"]
        for i in range(7)
    ]
    items.append(replace(items[0], id="duplicate", subject="b", task_group="b"))
    monkeypatch.setattr(bench, "data", Loaded(items, items, 0))
    monkeypatch.setattr(HFDataset, "describe", lambda self: "offline-fixture")
    cfg = parse_config(
        {
            "experiment": "holdout",
            "output_dir": str(tmp_path),
            "num_fewshot": 5,
            "models": [{"name": "m", "type": "instruct"}],
            "benchmarks": [
                {"name": name, "evaluations": ["gen"], "fewshot_holdout": True},
                {"name": "ifeval_pt", "num_fewshot": 0},
            ],
        }
    )
    step, ifeval = cfg.plan()
    assert step.num_fewshot == 5 and ifeval.num_fewshot == 0
    exp = Experiment(cfg)
    selected, shots = exp.select(step)
    assert {x.id for x in selected} == {"a-5", "a-6", "b-5", "b-6"}
    reserved = {x.id for examples in shots.values() for x in examples}
    assert reserved == {f"{g}-{i}" for g in ["a", "b"] for i in range(5)}
    assert not reserved & {x.id for x in selected}
    assert all(len(examples) == 5 for examples in shots.values())
    ev = EVALUATIONS.get("gen")
    assert exp._fingerprint(step, bench.version, "prompt", ev) != exp._fingerprint(
        replace(step, fewshot_holdout=False), bench.version, "prompt", ev
    )

    class Client:
        async def loglikelihood(self, requests):
            raise AssertionError("Esta avaliação usa somente geração")

        async def generate(self, requests):
            return [GenerateResponse("Resposta: A", finish_reason="stop") for _ in requests]

    result = asyncio.run(exp.run_step(step, Client()))
    assert result["benchmark"]["n_evaluated"] == 4
    assert result["evaluation"]["num_fewshot"] == 5
    assert result["evaluation"]["fewshot_policy"] == "heldout_first_n_per_area"
    assert set(result["evaluation"]["reserved_fewshot_ids"]) == reserved
    assert result["metrics"]["acc"] == 1.0


def test_holdout_config_rejects_string_boolean():
    with pytest.raises(ValueError, match="fewshot_holdout deve ser booleano"):
        parse_config(
            {
                "experiment": "bad",
                "models": [{"name": "m", "type": "instruct"}],
                "benchmarks": [{"name": "include_pt", "fewshot_holdout": "false"}],
            }
        )
