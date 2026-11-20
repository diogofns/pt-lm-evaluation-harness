import asyncio
import json

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS
from pt_lm_eval.benchmarks.base import HFDataset, Loaded
from pt_lm_eval.core import LoglikelihoodResponse
from pt_lm_eval.experiment.config import parse_config
from pt_lm_eval.experiment.runner import Experiment


def row(index=0, level=4):
    return {
        "_split": "test",
        "_index": index,
        "problem": f"Calcule $x_{index}^2$.",
        # Deliberately shuffled mapping: option order must follow labels A–E.
        "choices": {"D": "12", "A": "16", "E": "11", "C": "13", "B": "14"},
        "answer": "D",
        "level": level,
        "solution": "DO_NOT_INCLUDE_SOLUTION",
    }


def test_math_pt_br_selects_only_requested_subset_and_maps_letter_target():
    bench = BENCHMARKS.get("math_pt_br")
    assert bench.source.repo == "tiagoteixeira03/MATH-PT"
    assert bench.source.subset == "ptbr_multiple_choice" and bench.source.split == "test"
    assert bench.source.revision == "11c44f85c32d5d6b3dce76d163201ae4c6abd4b2"
    item = bench.to_item(row())
    assert (item.id, item.subject, item.gold) == ("test-0", "4", 3)
    assert item.options == ("16", "14", "13", "12", "11")
    requests = EVALUATIONS.get("mcf").build(item, [])
    assert [r.continuation for r in requests] == [" A", " B", " C", " D", " E"]
    assert requests[0].context == "Calcule $x_0^2$.\nA. 16\nB. 14\nC. 13\nD. 12\nE. 11\nResposta:"
    assert "DO_NOT_INCLUDE_SOLUTION" not in requests[0].context
    responses = [LoglikelihoodResponse(-1.0 if i == 3 else -5.0, False, 1) for i in range(5)]
    metrics = EVALUATIONS.get("mcf").score(item, responses).metrics
    assert metrics == {"acc": 1.0, "random_baseline": 0.2}


def test_math_pt_br_rejects_invalid_answers():
    with pytest.raises(ValueError):
        BENCHMARKS.get("math_pt_br").to_item({**row(), "answer": "F"})


@pytest.mark.parametrize("model_type", ["base", "instruct"])
def test_math_pt_br_reserves_first_examples_per_level(monkeypatch, model_type):
    bench = BENCHMARKS.get("math_pt_br")
    items = [bench.to_item(row(i, level)) for i, level in enumerate([2, 2, 2, 4, 4, 4])]
    monkeypatch.setattr(bench, "data", Loaded(items, items, 0))
    cfg = parse_config(
        {
            "experiment": "math",
            "limit": 1,
            "num_fewshot": 1,
            "models": [{"name": "m", "type": model_type}],
            "benchmarks": [bench.name],
        }
    )
    assert [s.evaluation for s in cfg.plan()] == (["mcf"] if model_type == "base" else ["mcf", "gen"])
    selected, shots = Experiment(cfg).select(cfg.plan()[0])
    assert [item.id for item in selected] == ["test-1", "test-4"]
    assert [item.id for item in shots["test-1"]] == ["test-0"]
    assert [item.id for item in shots["test-4"]] == ["test-3"]
    assert not {item.id for item in selected} & {s.id for examples in shots.values() for s in examples}


def test_math_pt_br_end_to_end_without_inference_server(monkeypatch, tmp_path):
    bench = BENCHMARKS.get("math_pt_br")
    items = [bench.to_item(row(i, level)) for i, level in enumerate([2, 2, 2, 4, 4, 4])]
    monkeypatch.setattr(bench, "data", Loaded(items, items, 0))
    monkeypatch.setattr(HFDataset, "describe", lambda self: "offline-fixture")
    cfg = parse_config(
        {
            "experiment": "math",
            "output_dir": str(tmp_path),
            "num_fewshot": 1,
            "models": [{"name": "m", "type": "instruct"}],
            "benchmarks": [bench.name],
        }
    )

    class Client:
        async def loglikelihood(self, requests):
            return [LoglikelihoodResponse(-1 if r.continuation == " D" else -5, False, 1) for r in requests]

        async def generate(self, requests):
            raise AssertionError("MATH-PT deve usar apenas MCF")

    result = asyncio.run(Experiment(cfg).run_step(cfg.plan()[0], Client()))
    assert result["benchmark"]["n_evaluated"] == 4
    assert result["metrics"]["acc"] == 1.0
    assert set(result["by_task"]) == {"2", "4"}
    assert result["evaluation"]["reserved_fewshot_ids"] == ["test-0", "test-3"]
    output = cfg.out_dir / "m/math_pt_br@1.2.mcf.results.json"
    assert json.loads(output.read_text())["metrics"] == result["metrics"]
