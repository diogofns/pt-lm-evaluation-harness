from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS
from pt_lm_eval.benchmarks.base import Loaded
from pt_lm_eval.core import GenerateResponse, LoglikelihoodResponse
from pt_lm_eval.experiment.config import parse_config
from pt_lm_eval.experiment.report import Results, aggregate
from pt_lm_eval.experiment.runner import Experiment

FIXTURES = json.loads((Path(__file__).parent / "fixtures/mcq_protocols.json").read_text())
METHODS = {
    "global_mmlu_pt": "mcf",
    "mmmlu_pt_br": "mcf",
    "include_pt": "mcf",
    "global_piqa_nonparallel_pt_br": "cf",
}


def documents(name):
    return next(case["rows"] for case in FIXTURES["cases"] if case["benchmark"] == name)


def native_benchmark(monkeypatch, name, rows):
    bench = BENCHMARKS.get(name)
    items = [bench.to_item({**row, "_split": "test", "_index": i}) for i, row in enumerate(rows)]
    pool = [replace(item, id=f"dev-{i}") for i, item in enumerate(items)] if bench.fewshot_source else items
    monkeypatch.setattr(bench, "data", Loaded(items, pool, 0))
    return bench, items


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: f"{c['benchmark']}-{c['shots']}shot")
def test_mcq_requests_and_scores_match_frozen_cases(monkeypatch, case):
    name = case["benchmark"]
    bench, items = native_benchmark(monkeypatch, name, case["rows"])
    shots = bench.select_shots_for_items(items, case["shots"], 1234)
    for item, expected in zip(items, case["expected"], strict=True):
        assert item.id == expected["id"]
        requests = EVALUATIONS.get(METHODS[name]).build(item, shots[item.id])
        assert [[r.context, r.continuation] for r in requests] == expected["pairs"]
        assert bench.select_shots(item, case["shots"], 1234) == shots[item.id]
        for score in expected["scores"]:
            responses = [
                LoglikelihoodResponse(value, False, i + 1) for i, value in enumerate(score["logprobs"])
            ]
            metrics = EVALUATIONS.get(METHODS[name]).score(item, responses).metrics
            if name == "global_piqa_nonparallel_pt_br":
                metrics["acc_norm"] = metrics["acc_char"]
            for metric, value in score["metrics"].items():
                assert metrics[metric] == value


@pytest.mark.parametrize("case", FIXTURES["generation"], ids=lambda c: f"{c['shots']}shot")
def test_piqa_generation_matches_frozen_cases(monkeypatch, case):
    bench, items = native_benchmark(monkeypatch, "global_piqa_nonparallel_pt_br", case["rows"])
    shots = bench.select_shots_for_items(items, case["shots"], 1234)
    for item, expected in zip(items, case["expected"], strict=True):
        request = EVALUATIONS.get("gen").build(item, shots[item.id])[0]
        assert request.prompt == expected["prompt"]
        parameters = expected["parameters"]
        assert request.max_tokens == parameters["max_gen_toks"] == 2048
        assert request.params == {"temperature": parameters["temperature"], "top_p": parameters["top_p"]}
        assert not request.stop and parameters["until"] == []
        assert request.enforce_generation_config


@pytest.mark.parametrize("case", FIXTURES["extraction"])
def test_piqa_filter_matches_frozen_cases(monkeypatch, case):
    _, items = native_benchmark(
        monkeypatch, "global_piqa_nonparallel_pt_br", documents("global_piqa_nonparallel_pt_br")
    )
    score = EVALUATIONS.get("gen").score(items[0], [GenerateResponse(case["text"])])
    assert (score.details["pred"] or "[invalid]") == case["expected"]
    assert score.metrics["exact_match"] == float(case["expected"] == "A")


def test_grouped_statistics_match_frozen_case():
    case = FIXTURES["group_stats"]
    records = [
        {"subject": "same", "task_group": str(i), "metrics": {"acc": value}}
        for i, values in enumerate(case["groups"])
        for value in values
    ]
    result = aggregate(records, group_by_task=True)
    assert result["metrics"]["acc"] == pytest.approx(case["mean"])
    assert result["metrics"]["acc_stderr"] == pytest.approx(case["stderr"])
    assert len(result["by_task"]) == 3
    assert result["metrics"]["acc_stderr"] != aggregate(records)["metrics"]["acc_stderr"]
    assert math.isnan(aggregate(records[:1], group_by_task=True)["metrics"]["acc_stderr"])


def test_limit_is_per_subtask_and_piqa_generation_accepts_base(monkeypatch):
    rows = documents("include_pt")
    bench, _ = native_benchmark(monkeypatch, "include_pt", rows)
    cfg = parse_config(
        {
            "experiment": "port",
            "limit": 2,
            "num_fewshot": 2,
            "models": [{"name": "m", "type": "base"}],
            "benchmarks": ["include_pt", "global_piqa_nonparallel_pt_br"],
        }
    )
    experiment = Experiment(cfg)
    step = next(s for s in cfg.plan() if s.benchmark == "include_pt")
    selected, shots = experiment.select(step)
    assert [item.id for item in selected] == ["test-0", "test-1", "test-5", "test-6"]
    assert shots == bench.select_shots_for_items(selected, 2, cfg.seed)
    assert any(s.evaluation == "gen" for s in cfg.plan() if s.benchmark.startswith("global_piqa"))


def test_new_protocol_paths_preserve_old_results(tmp_path):
    cfg = parse_config(
        {"experiment": "port", "models": [{"name": "m", "type": "base"}], "benchmarks": ["mmmlu_pt_br"]}
    )
    old = tmp_path / "m/mmmlu_pt_br.mcf.results.json"
    old.parent.mkdir()
    old.write_text('{"historical": true}\n')
    previous = old.with_name("mmmlu_pt_br@2.1.mcf.results.json")
    previous.write_text('{"historical_version": "2.1"}\n')
    samples, result = Results(tmp_path).paths(cfg.plan()[0])
    assert result.name == "mmmlu_pt_br@2.3.mcf.results.json"
    assert samples.name == "mmmlu_pt_br@2.3.mcf.samples.jsonl"
    assert old.read_text() == '{"historical": true}\n'
    assert previous.read_text() == '{"historical_version": "2.1"}\n'


@pytest.mark.parametrize("model_type", ["base", "instruct"])
def test_pt_collection_uses_only_declared_reference_evaluations(model_type):
    names = [
        "global_mmlu_pt",
        "mmmlu_pt_br",
        "include_pt",
        "mmlu_pt_high_school",
        "mmlu_pt_undergraduate",
        "mmlu_pt_high_school_hard",
        "mmlu_pt_undergraduate_hard",
    ]
    cfg = parse_config(
        {
            "experiment": "reference",
            "models": [{"name": "m", "type": model_type}],
            "benchmarks": [*names, "global_piqa_nonparallel_pt_br", "ifeval_pt"],
        }
    )
    for name in names:
        assert [s.evaluation for s in cfg.plan() if s.benchmark == name] == (
            ["mcf"] if model_type == "base" else ["mcf", "gen"]
        )
    assert [s.evaluation for s in cfg.plan() if s.benchmark.startswith("global_piqa")] == ["cf", "gen"]
    ifeval = next(s for s in cfg.plan() if s.benchmark == "ifeval_pt")
    assert (ifeval.evaluation == "ifeval") == (model_type == "instruct")
    rejected = parse_config(
        {
            "experiment": "reference",
            "models": [{"name": "m", "type": model_type}],
            "benchmarks": [{"name": "mmmlu_pt_br", "evaluations": ["cf"]}],
        }
    )
    with pytest.raises(ValueError, match="não declaradas"):
        rejected.plan()
