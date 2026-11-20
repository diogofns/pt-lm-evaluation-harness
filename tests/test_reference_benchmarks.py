from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS, CompletionItem, InstructionItem
from pt_lm_eval.core import GenerateResponse, LoglikelihoodResponse
from pt_lm_eval.experiment.config import parse_config
from pt_lm_eval.experiment.report import aggregate
from pt_lm_eval.experiment.runner import _resume
from pt_lm_eval.inference.client import Client

ROOT = Path(__file__).parents[1]
FIXTURES = json.loads((ROOT / "tests/fixtures/english_protocols.json").read_text())


def test_default_shots_and_model_capabilities():
    cfg = parse_config(
        {
            "experiment": "ref",
            "models": [{"name": "m", "type": "base"}],
            "benchmarks": [
                "gsm8k",
                "mmlu_pro",
                "ifeval",
                "gpqa_diamond_zeroshot",
                "mmlu_redux_generative",
                "mmmlu_pt_br",
            ],
        }
    )
    plan = cfg.plan()
    assert {s.benchmark: s.num_fewshot for s in plan} == {
        "gsm8k": 5,
        "mmlu_pro": 5,
        "ifeval": 0,
        "gpqa_diamond_zeroshot": 0,
        "mmlu_redux_generative": 0,
        "mmmlu_pt_br": 0,
    }
    assert all(not s.skip_reason for s in plan)
    assert not any(s.benchmark == "mmmlu_pt_br" and s.evaluation == "gen" for s in plan)
    override = parse_config(
        {
            "experiment": "x",
            "models": [{"name": "m", "type": "instruct"}],
            "num_fewshot": 3,
            "benchmarks": ["gsm8k", "ifeval"],
        }
    )
    assert [(s.benchmark, s.num_fewshot) for s in override.plan()] == [("gsm8k", 3), ("ifeval", 0)]
    with pytest.raises(ValueError, match="exige num_fewshot=0"):
        parse_config(
            {
                "experiment": "x",
                "models": [{"name": "m", "type": "instruct"}],
                "benchmarks": [{"name": "gpqa_main_zeroshot", "num_fewshot": 2}],
            }
        ).plan()


def test_mmlu_pro_prompt_and_fewshot_target_match_frozen_case():
    expected = FIXTURES["mmlu_pro"]
    row = expected["row"]
    item = BENCHMARKS.get("mmlu_pro").to_item(row)
    assert item.prompt == expected["example_prompt"]
    assert item.target == expected["example_target"]
    query = BENCHMARKS.get("mmlu_pro").to_item({**row, "_split": "test", "_index": 1})
    (request,) = EVALUATIONS.get("exact_match").build(query, [item])
    assert request.prompt == (
        query.description
        + expected["example_prompt"]
        + " "
        + expected["example_target"]
        + "\n\n"
        + expected["query_prompt"]
    )
    assert query.description == expected["description"]
    assert request.max_tokens == 2048
    assert request.stop == ("Question:",)


def test_gpqa_shuffling_prompt_targets_and_metrics_match_frozen_cases(monkeypatch):
    case = FIXTURES["gpqa"]
    bench = type(BENCHMARKS.get("gpqa_diamond_zeroshot"))()
    monkeypatch.setattr(type(bench.source), "load", lambda source: case["rows"])
    for item, expected in zip(bench.data.items, case["expected"], strict=True):
        assert item.options == tuple(expected["choices"])
        assert f"({item.letters[item.gold]})" == expected["answer"]
        requests = EVALUATIONS.get("gpqa_mcf").build(item, [])
        assert [r.continuation for r in requests] == [" (A)", " (B)", " (C)", " (D)"]
        assert requests[0].context == (
            f"What is the correct answer to this question:{expected['question']}\nChoices:\n"
            + "\n".join(f"({c}) {option}" for c, option in zip("ABCD", expected["choices"], strict=True))
            + "\nAnswer:"
        )
        responses = [LoglikelihoodResponse(-1 if i == item.gold else -10, False, 1) for i in range(4)]
        assert EVALUATIONS.get("gpqa_mcf").score(item, responses).metrics == {"acc": 1.0, "acc_norm": 1.0}


@pytest.mark.parametrize(
    "response",
    ["#### 1,200", "I get $1,200.", "first 3, then 1200", "#### -5.5", "No number", "#### 1\n#### 1200"],
)
def test_gsm8k_numeric_filters(response):
    expected = {
        "#### 1,200": (1, 1),
        "I get $1,200.": (0, 1),
        "first 3, then 1200": (0, 1),
        "#### -5.5": (0, 0),
        "No number": (0, 0),
        "#### 1\n#### 1200": (0, 1),
    }
    item = CompletionItem("g", "Question: Test?\nAnswer:", "Solution\n#### 1200", "gsm8k")
    metrics = EVALUATIONS.get("exact_match").score(item, [GenerateResponse(response)]).metrics
    strict, flexible = expected[response]
    assert metrics == {"exact_match,strict-match": strict, "exact_match,flexible-extract": flexible}


def test_reference_client_enforces_task_limits_stops_and_keeps_whitespace():
    item = CompletionItem("g", "Question: Test?\nAnswer:", "#### 4", "gsm8k")
    (request,) = EVALUATIONS.get("exact_match").build(item, [])
    captured = []
    client = Client(
        model="m",
        base_url="http://unused",
        chat=True,
        generate_params={"max_tokens": 8192, "temperature": 0.9},
    )

    async def post(path, payload):
        captured.append((path, payload))
        return {"choices": [{"message": {"content": "  #### 4\n"}, "finish_reason": "stop"}]}

    client._post = post
    response = asyncio.run(client._generate_one(request))
    assert captured[0][1]["max_tokens"] == 256
    assert captured[0][1]["temperature"] == 0
    assert captured[0][1]["stop"] == ["Question:", "</s>", "<|im_end|>"]
    assert response.text == "  #### 4\n"


@pytest.mark.parametrize("override,temperature,top_p", [(False, 0.8, 0.95), (True, 0.2, 0.9)])
def test_sampling_sweep_can_override_piqa_without_changing_token_limit(override, temperature, top_p):
    from pt_lm_eval.core import MCQItem, non_default
    from pt_lm_eval.experiment.config import ModelConfig
    from pt_lm_eval.experiment.runner import Experiment

    model = ModelConfig(
        name="m",
        type="instruct",
        thinking=False,
        override_task_sampling=override,
        generate={"temperature": 0.2, "top_p": 0.9, "max_tokens": 8192},
    )
    item = MCQItem("p", "situation", ("one", "two"), 0, protocol="global_piqa")
    (request,) = EVALUATIONS.get("gen").build(item, [])
    captured = []

    async def post(path, payload):
        captured.append(payload)
        return {"choices": [{"message": {"content": "Resposta: A"}}]}

    client = Client.for_model(model, "http://unused", None)
    client._post = post
    asyncio.run(client.generate([request]))
    assert captured[0]["temperature"] == temperature
    assert captured[0]["top_p"] == top_p
    assert captured[0]["max_tokens"] == 2048
    assert captured[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert ("override_task_sampling" in non_default(model)) == override
    cfg = parse_config(
        {
            "experiment": "sweep",
            "models": [{"name": "m", "type": "instruct", "override_task_sampling": override}],
            "benchmarks": ["global_piqa_nonparallel_pt_br"],
        }
    )
    original = parse_config(
        {
            "experiment": "sweep",
            "models": [{"name": "m", "type": "instruct"}],
            "benchmarks": ["global_piqa_nonparallel_pt_br"],
        }
    )
    step, other = cfg.plan()[0], original.plan()[0]
    fp = Experiment(cfg)._fingerprint(step, "2.0", "prompt", EVALUATIONS.get("cf"))
    before = Experiment(original)._fingerprint(other, "2.0", "prompt", EVALUATIONS.get("cf"))
    assert (fp != before) == override


def test_english_ifeval_uses_reference_and_instruction_weighting():
    ev = EVALUATIONS.get("ifeval_en")
    a = InstructionItem(
        "i",
        "No commas. Include hello.",
        ("punctuation:no_comma", "keywords:existence"),
        ({}, {"keywords": ["hello"]}),
    )
    b = InstructionItem("j", "No commas.", ("punctuation:no_comma",), ({},))
    one = ev.score(a, [GenerateResponse("hello, world")])
    two = ev.score(b, [GenerateResponse("fine")])
    assert one.metrics["prompt_level_strict_acc"] == 0
    assert one.metrics["inst_level_strict_acc"] == 0.5
    result = aggregate([{"metrics": s.metrics, "weights": s.weights} for s in [one, two]])
    assert result["metrics"]["inst_level_strict_acc"] == pytest.approx(2 / 3)
    assert ev.build(a, [])[0].max_tokens == 1280
    assert ev.build(a, [])[0].stop == ()


def test_english_ifeval_loose_ignores_intro_line():
    item = InstructionItem(
        "i",
        "No commas. Include hello.",
        ("punctuation:no_comma", "keywords:existence"),
        ({}, {"keywords": ["hello"]}),
    )
    score = EVALUATIONS.get("ifeval_en").score(item, [GenerateResponse("Intro, with a comma.\nhello")])
    assert score.metrics["prompt_level_strict_acc"] == 0.0
    assert score.metrics["inst_level_strict_acc"] == 0.5
    assert score.metrics["prompt_level_loose_acc"] == 1.0
    assert score.metrics["inst_level_loose_acc"] == 1.0


def test_resume_keeps_unicode_next_line_in_json_strings(tmp_path):
    path = tmp_path / "samples.jsonl"
    record = {"id": "last", "prompt": "first\u0085last"}
    path.write_text(json.dumps({"_fingerprint": "f"}) + "\n" + json.dumps(record, ensure_ascii=False) + "\n")
    assert _resume(path, "f") == {"last": record}


def test_resume_repairs_empty_headers_and_truncated_utf8(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.touch()
    assert _resume(path, "f") == {}
    with path.open("ab") as fh:
        fh.write(b'{"id":"ok"}\n{"id":"unfinished","text":"\xc3')
    assert _resume(path, "f") == {"ok": {"id": "ok"}}
    assert path.read_text().splitlines()[-1] == '{"id": "ok"}'


def test_gsm8k_sampler_matches_reference_rng_in_canonical_order():
    from pt_lm_eval.benchmarks.base import Loaded

    bench = type(BENCHMARKS.get("gsm8k"))()
    docs = [CompletionItem(f"test/{i}", f"Q{i}", "#### 4", "gsm8k") for i in range(4)]
    pool = [CompletionItem(f"train/{i}", f"Shot{i}", "#### 4", "gsm8k") for i in range(20)]
    bench.data = Loaded(docs, pool, 0)
    rng = random.Random(1234)
    expected = {d.id: rng.sample(pool, 5) for d in docs}
    for doc in reversed(docs):
        assert bench.select_shots(doc, 5, 1234) == expected[doc.id]
    assert bench.select_shots(docs[0], 5, 1234) == expected[docs[0].id]


@pytest.mark.parametrize(
    "protocol,text,expected",
    [
        ("mmlu_pro", "The answer is (B).", 1.0),
        ("mmlu_pro", "The Answer is (B).", 0.0),
        ("mmlu_pro", "answer is (C), but answer is (B)", 0.0),
        ("mmlu_redux", "B", 1.0),
        ("mmlu_redux", "Answer: B", 0.0),
        ("mmlu_redux", "b", 0.0),
    ],
)
def test_letter_extraction_preserves_protocol_case_and_first_match(protocol, text, expected):
    item = CompletionItem("m", "question", "B", protocol)
    score = EVALUATIONS.get("exact_match").score(item, [GenerateResponse(text)])
    assert list(score.metrics.values()) == [expected]


def test_reference_stops_do_not_cut_reasoning_and_are_applied_to_final_answer():
    (request,) = EVALUATIONS.get("exact_match").build(
        CompletionItem("g", "Question: Test?\nAnswer:", "#### 4", "gsm8k"), []
    )
    payloads = []
    client = Client(model="m", base_url="http://unused", chat=True, reasoning_enabled=True)

    async def post(path, payload):
        payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "Question: do not stop here",
                        "content": "  #### 4\nQuestion: next question",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    client._post = post
    result = asyncio.run(client._generate_one(request))
    assert "stop" not in payloads[0]
    assert result.text == "#### 4\n"
    assert result.reasoning == "Question: do not stop here"
