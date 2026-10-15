"""Testes do núcleo que não precisam de rede nem GPU."""

from __future__ import annotations

import math

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS, MCQItem, columns, letter_index
from pt_lm_eval.backends.openai_compat import _continuation_logprob
from pt_lm_eval.backends.tokenization import truncate_left
from pt_lm_eval.core.registry import Registry
from pt_lm_eval.core.types import LoglikelihoodResponse
from pt_lm_eval.data.fewshot import select_shots
from pt_lm_eval.metrics.accuracy import argmax, choice_accuracy
from pt_lm_eval.metrics.stats import macro_mean_stderr, mean_stderr
from pt_lm_eval.report.results import aggregate
from pt_lm_eval.runner.config import parse_config

ITEM = MCQItem(id="q1", question="Quanto é 2 + 2?", options=("3", "4", "5"), gold=1, subject="mat")


# --- tipos ----------------------------------------------------------------------


def test_mcq_item_validates_gold_and_options():
    with pytest.raises(ValueError, match="fora do intervalo"):
        MCQItem(id="x", question="?", options=("a", "b"), gold=2)
    with pytest.raises(ValueError, match="opção vazia"):
        MCQItem(id="x", question="?", options=("a", " "), gold=0)
    assert ITEM.letters == "ABC"


# --- registro -------------------------------------------------------------------


def test_registry_rejects_duplicates_and_suggests_names():
    reg: Registry[int] = Registry("coisa", "grupo.inexistente")
    reg.register("energia", 1)
    with pytest.raises(ValueError, match="já registrado"):
        reg.register("energia", 2)
    with pytest.raises(KeyError, match="Quis dizer: energia"):
        reg.get("enrgia")


def test_builtin_plugins_are_discovered():
    assert {"mcf", "cf", "cf_pmi", "gen"} <= set(EVALUATIONS.names())
    assert "energy_regulacao" in BENCHMARKS.names()
    assert BENCHMARKS.origin("energy_regulacao") == "pt-lm-eval"


# --- dados ----------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [("B", 1), (" b ", 1), ("(C)", 2), ("A)", 0)])
def test_letter_index(value, expected):
    assert letter_index(value) == expected


def test_columns_accepts_letter_index_or_text():
    row = {"_split": "test", "_index": 7, "q": "?", "opts": {"text": ["x", "y"]}, "a": "y"}
    to_item = columns(question="q", options="opts.text", answer="a")
    item = to_item(row)
    assert (item.id, item.gold, item.options) == ("test-7", 1, ("x", "y"))
    assert to_item({**row, "a": 0}).gold == 0
    assert to_item({**row, "a": "B"}).gold == 1


def test_select_shots_is_deterministic_and_excludes_self():
    pool = [MCQItem(id=f"p{i}", question="?", options=("a", "b"), gold=0) for i in range(20)]
    shots = select_shots(pool[0], pool, 3, seed=1)
    assert shots == select_shots(pool[0], pool, 3, seed=1)
    assert pool[0] not in shots and len(shots) == 3
    assert select_shots(pool[0], pool, 2, strategy="first_n") == pool[1:3]


# --- tokenização e logprobs -------------------------------------------------------


def test_truncate_left_keeps_continuation():
    ids, start = truncate_left([1, 2, 3, 4], [5, 6], max_length=4)
    assert ids == [3, 4, 5, 6] and start == 2
    with pytest.raises(ValueError):
        truncate_left([1], [2, 3, 4], max_length=3)


def test_continuation_logprob_checks_token_alignment():
    ids = [10, 11, 12]
    prompt_logprobs = [
        None,
        {"11": {"logprob": -1.0, "rank": 1}},
        {"12": {"logprob": -2.0, "rank": 3}, "7": {"logprob": -0.5, "rank": 1}},
    ]
    r = _continuation_logprob(prompt_logprobs, ids, start=1)
    assert (r.logprob, r.is_greedy, r.n_tokens) == (-3.0, False, 2)
    # Empate exato com o mais provável conta como guloso, mesmo com rank 2.
    tie = [None, {"11": {"logprob": -1.0, "rank": 2}, "5": {"logprob": -1.0, "rank": 1}}]
    assert _continuation_logprob(tie, [10, 11], start=1).is_greedy
    with pytest.raises(KeyError):  # servidor tokenizou diferente: falha alta, nunca silenciosa
        _continuation_logprob(prompt_logprobs, [10, 99, 12], start=1)


# --- métricas -------------------------------------------------------------------


def test_argmax_ties_pick_first():
    assert argmax([1.0, 3.0, 3.0]) == 1


def test_choice_accuracy_normalizations():
    # Opção 0 tem logprob total maior, mas é longa: perde nas versões normalizadas.
    responses = [LoglikelihoodResponse(-6.0, False, 6), LoglikelihoodResponse(-4.0, False, 1)]
    texts = ["uma opção bem comprida", "x"]
    unconditional = [LoglikelihoodResponse(-1.0, False, 6), LoglikelihoodResponse(-10.0, False, 1)]
    got = choice_accuracy(responses, gold=0, choice_texts=texts, unconditional=unconditional)
    assert got == {"acc": 0.0, "acc_char": 1.0, "acc_token": 1.0, "acc_pmi": 0.0}


def test_mean_stderr_matches_formula():
    mean, se = mean_stderr([1.0, 0.0, 1.0, 1.0])
    assert mean == 0.75 and se == pytest.approx(math.sqrt(0.25 / 4))
    assert math.isnan(mean_stderr([1.0])[1])
    # Grupos [1, 0] e [1]: média das médias = 0.75; variância agrupada de [1, 0, 1] = 1/3.
    mean, se = macro_mean_stderr([[1.0, 0.0], [1.0]])
    assert mean == 0.75 and se == pytest.approx(math.sqrt((1 / 3) * (1 / 2 + 1)) / 2)


def test_aggregate_uses_macro_average_across_subjects():
    records = [{"subject": "a", "metrics": {"acc": 1.0}}] * 3 + [{"subject": "b", "metrics": {"acc": 0.0}}]
    agg = aggregate(records)
    assert agg["metrics"]["acc"] == 0.5  # média das matérias
    assert agg["metrics_micro"]["acc"] == 0.75  # média dos itens
    assert agg["by_subject"]["a"]["n"] == 3


# --- configuração -------------------------------------------------------------------


def test_config_rejects_unknown_keys():
    base = {"experiment": "x", "models": [{"name": "m", "type": "base"}], "benchmarks": ["energy_regulacao"]}
    assert parse_config(base).benchmarks[0].name == "energy_regulacao"
    with pytest.raises(ValueError, match="chaves desconhecidas"):
        parse_config({**base, "modles": []})
    with pytest.raises(ValueError, match="type deve ser"):
        parse_config({**base, "models": [{"name": "m", "type": "chat"}]})
