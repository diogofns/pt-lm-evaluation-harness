import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS
from pt_lm_eval.benchmarks.base import HFDataset, Loaded
from pt_lm_eval.core import LoglikelihoodResponse
from pt_lm_eval.experiment.config import parse_config
from pt_lm_eval.experiment.runner import Experiment

VARIANTS = [
    "mmlu_pt_undergraduate",
    "mmlu_pt_undergraduate_hard",
    "mmlu_pt_high_school",
    "mmlu_pt_high_school_hard",
]


def row(index=0, subject="stem", choices=4):
    letters = "ABCDEF"[:choices]
    return {
        "_split": subject,
        "_index": index,
        "statement": f"  Questão {subject}/{index}?  ",
        "alternatives": {"text": [f" opção {letter} " for letter in letters]},
        "answer": letters[-1],
    }


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("choices", [3, 4, 5, 6])
def test_mmlu_pt_matches_global_mmlu_rendering_and_scoring(name, choices):
    item = BENCHMARKS.get(name).to_item(row(choices=choices))
    requests = EVALUATIONS.get("mcf").build(item, [])
    letters = "ABCDEF"[:choices]
    expected = "Questão stem/0?\n" + "\n".join(f"{letter}.  opção {letter} " for letter in letters)
    expected += "\nResposta:"
    assert len(requests) == choices
    assert [request.context for request in requests] == [expected] * choices
    assert [request.continuation for request in requests] == [f" {letter}" for letter in letters]
    responses = [LoglikelihoodResponse(float(i - choices), False, 1) for i in range(choices)]
    result = EVALUATIONS.get("mcf").score(item, responses)
    assert result.metrics["acc"] == 1.0 and result.details["pred"] == letters[-1]
    if choices == 4:
        global_item = BENCHMARKS.get("global_mmlu_pt").to_item(
            {
                "sample_id": "global",
                "question": item.question,
                "subject": item.subject,
                "answer": item.letters[item.gold],
                **dict(zip([f"option_{letter}" for letter in "abcd"], item.options, strict=True)),
            }
        )
        assert requests == EVALUATIONS.get("mcf").build(global_item, [])
        assert result == EVALUATIONS.get("mcf").score(global_item, responses)


@pytest.mark.parametrize("name", VARIANTS)
def test_mmlu_pt_uses_first_examples_and_limit_per_area(monkeypatch, name):
    bench = BENCHMARKS.get(name)
    items = [bench.to_item(row(i, subject)) for subject in ["stem", "law"] for i in range(4)]
    monkeypatch.setattr(bench, "data", Loaded(items, items, 0))
    selected, shots = bench.select_for_evaluation(items, k=2, limit=1, seed=1234)
    assert [item.id for item in selected] == ["stem-2", "law-2"]
    assert [item.id for item in shots["stem-2"]] == ["stem-0", "stem-1"]
    assert [item.id for item in shots["law-2"]] == ["law-0", "law-1"]
    assert shots == bench.select_shots_for_items(selected, k=2, seed=9999)
    assert all(bench.select_shots(item, 2, 1234) == shots[item.id] for item in selected)
    all_items, all_shots = bench.select_for_evaluation(items, k=2, limit=None, seed=1234)
    assert {item.id for item in all_items} == {"stem-2", "stem-3", "law-2", "law-3"}
    reserved_ids = {shot.id for examples in all_shots.values() for shot in examples}
    assert not reserved_ids & {item.id for item in all_items}
    assert all_shots["stem-2"] == all_shots["stem-3"]
    with pytest.raises(ValueError, match="reservada"):
        bench.select_shots(items[0], 2, 1234)
    for model_type in ["base", "instruct"]:
        config = parse_config(
            {"experiment": "mmlu", "models": [{"name": "m", "type": model_type}], "benchmarks": [name]}
        )
        assert [step.evaluation for step in config.plan()] == (
            ["mcf"] if model_type == "base" else ["mcf", "gen"]
        )


@pytest.mark.parametrize("num_shots", [0, 1])
def test_mmlu_pt_result_uses_global_mmlu_grouping_and_versioned_files(monkeypatch, tmp_path, num_shots):
    bench = BENCHMARKS.get("mmlu_pt_undergraduate")
    items = [
        bench.to_item({**row(i, subject), "answer": "AB"[i % 2]})
        for subject in ["stem", "law"]
        for i in range(3)
    ]
    monkeypatch.setattr(bench, "data", Loaded(items, items, 0))
    monkeypatch.setattr(HFDataset, "describe", lambda self: "offline-fixture")
    config = parse_config(
        {
            "experiment": "mmlu",
            "output_dir": str(tmp_path),
            "limit": 2,
            "num_fewshot": num_shots,
            "models": [{"name": "m", "type": "instruct"}],
            "benchmarks": ["mmlu_pt_undergraduate"],
        }
    )

    class Client:
        async def generate(self, requests):
            raise AssertionError("MMLU-PT deve usar apenas loglikelihood")

        async def loglikelihood(self, requests):
            return [
                LoglikelihoodResponse(-1.0 if r.continuation == " A" else -2.0, False, 1) for r in requests
            ]

    result = asyncio.run(Experiment(config).run_step(config.plan()[0], Client()))
    assert result["evaluation"]["aggregation"] == "grouped_tasks"
    assert set(result["evaluation"]["reserved_fewshot_ids"]) == ({"stem-0", "law-0"} if num_shots else set())
    assert result["benchmark"]["n_evaluated"] == 4
    assert set(result["by_task"]) == {"law", "stem"}
    assert result["metrics"]["acc"] == 0.5
    assert result["metrics"]["acc_stderr"] == pytest.approx(0.125**0.5)
    path = Path(config.out_dir) / "m/mmlu_pt_undergraduate@2.3.mcf.results.json"
    assert json.loads(path.read_text())["metrics"] == result["metrics"]


def test_holdout_excludes_duplicates_and_zero_shot_keeps_every_item(monkeypatch):
    bench = BENCHMARKS.get("mmlu_pt_high_school")
    items = [bench.to_item(row(i)) for i in range(5)]
    # Same question with another ID, whitespace and even a contradictory label.
    duplicate = replace(items[0], id="stem-duplicate", question=items[0].question.strip(), gold=0)
    items.append(duplicate)
    monkeypatch.setattr(bench, "data", Loaded(items, items, 0))
    selected, shots = bench.select_for_evaluation(items, k=1, limit=None, seed=1234)
    assert [item.id for item in selected] == ["stem-1", "stem-2", "stem-3", "stem-4"]
    assert all(examples == [items[0]] for examples in shots.values())
    selected, shots = bench.select_for_evaluation(items, k=0, limit=None, seed=1234)
    assert selected == items and all(not examples for examples in shots.values())
    with pytest.raises(ValueError, match="não deixam questões"):
        bench.select_for_evaluation(items[:2], k=2, limit=None, seed=1234)


def test_runner_reserves_examples_after_explicit_exclusions_before_limit(monkeypatch):
    bench = BENCHMARKS.get("mmlu_pt_undergraduate")
    items = [bench.to_item(row(i)) for i in range(5)]
    monkeypatch.setattr(bench, "data", Loaded(items, items, 0))
    cfg = parse_config(
        {
            "experiment": "holdout",
            "num_fewshot": 1,
            "limit": 1,
            "models": [{"name": "m", "type": "base"}],
            "benchmarks": [{"name": bench.name, "exclude_ids": ["stem-0"]}],
        }
    )
    selected, shots = Experiment(cfg).select(cfg.plan()[0])
    assert [item.id for item in selected] == ["stem-2"]
    assert [item.id for item in shots["stem-2"]] == ["stem-1"]


def test_preview_uses_only_heldout_eval_items(monkeypatch, capsys):
    from pt_lm_eval.cli import show_prompts

    bench = BENCHMARKS.get("mmlu_pt_high_school")
    items = [bench.to_item(row(i)) for i in range(4)]
    monkeypatch.setattr(bench, "data", Loaded(items, items, 0))
    show_prompts(bench.name, n=1, num_fewshot=2)
    output = capsys.readouterr().out
    assert "item stem-2" in output
    assert "Questão stem/0?" in output and "Questão stem/1?" in output
