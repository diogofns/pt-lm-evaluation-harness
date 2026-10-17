"""Prompts, pedidos e pontuação das avaliações de múltipla escolha."""

from __future__ import annotations

import pytest

from pt_lm_eval import EVALUATIONS, MCQItem
from pt_lm_eval.core import GenerateResponse, LoglikelihoodResponse
from pt_lm_eval.evaluations.multiple_choice import extract_letter
from pt_lm_eval.inference.client import split_reasoning

ITEM = MCQItem(id="q1", question="Quanto é 2 + 2?", options=("3", "4", "5"), gold=1)
SHOT = MCQItem(id="s1", question="Quanto é 1 + 1?", options=("2", "3", "4"), gold=0)


def test_mcf_prompt_and_continuations():
    reqs = EVALUATIONS.get("mcf").build(ITEM, [SHOT])
    assert reqs[0].context == (
        "Pergunta: Quanto é 1 + 1?\nA. 2\nB. 3\nC. 4\nResposta: A\n\n"
        "Pergunta: Quanto é 2 + 2?\nA. 3\nB. 4\nC. 5\nResposta:"
    )
    assert [r.continuation for r in reqs] == [" A", " B", " C"]


def test_cf_prompt_hides_options_and_scores_text():
    reqs = EVALUATIONS.get("cf").build(ITEM, [SHOT])
    assert reqs[0].context == "Pergunta: Quanto é 1 + 1?\nResposta: 2\n\nPergunta: Quanto é 2 + 2?\nResposta:"
    assert [r.continuation for r in reqs] == [" 3", " 4", " 5"]


def test_cf_normalizations_can_disagree_with_raw_accuracy():
    # Opção 0 tem logprob total maior, mas é longa: perde nas versões normalizadas.
    item = MCQItem(id="x", question="?", options=("uma opção bem comprida", "x"), gold=0)
    responses = [LoglikelihoodResponse(-6.0, False, 6), LoglikelihoodResponse(-4.0, False, 1)]
    metrics = EVALUATIONS.get("cf").score(item, responses).metrics
    assert metrics == {"acc": 0.0, "acc_char": 1.0, "acc_token": 1.0, "random_baseline": 0.5}


def test_gen_prompt_and_score():
    ev = EVALUATIONS.get("gen")
    (req,) = ev.build(ITEM, [SHOT])
    assert req.prompt.endswith(
        'Responda com a letra da alternativa correta no formato "Resposta: X".\nResposta:'
    )
    assert "Resposta: A\n\nPergunta: Quanto é 2 + 2?" in req.prompt
    ok = ev.score(ITEM, [GenerateResponse("Resposta: B", finish_reason="stop")])
    assert ok.metrics == {"acc": 1.0, "extracted": 1.0, "truncated": 0.0, "random_baseline": 1 / 3}
    cut = ev.score(ITEM, [GenerateResponse("Vamos pensar passo a passo", finish_reason="length")])
    assert cut.metrics == {"acc": 0.0, "extracted": 0.0, "truncated": 1.0, "random_baseline": 1 / 3}


def test_gen_multiturn_puts_each_shot_in_its_own_turns():
    ev = EVALUATIONS.get("gen")
    (req,) = ev.build(ITEM, [SHOT], multiturn=True)
    assert [m["role"] for m in req.messages] == ["user", "assistant", "user"]
    assert req.messages[0]["content"].startswith("Pergunta: Quanto é 1 + 1?\nA. 2")
    assert req.messages[1]["content"] == "Resposta: A"
    assert req.messages[2]["content"].endswith('no formato "Resposta: X".')
    assert req.stop == ()
    assert ev.build(ITEM, [], multiturn=True) == ev.build(ITEM, [])  # sem exemplos, nada muda


def test_loglikelihood_records_random_baseline():
    responses = [LoglikelihoodResponse(lp, False, 1) for lp in (-3.0, -1.0, -2.0)]
    assert EVALUATIONS.get("mcf").score(ITEM, responses).metrics == {"acc": 1.0, "random_baseline": 1 / 3}


@pytest.mark.parametrize(
    "text,expected",
    [
        # No formato pedido, normalizado (maiúsculas, markdown, parênteses, pontuação):
        ("Resposta: B", "B"),
        ("resposta: (C)", "C"),
        ("Resposta: b", "B"),
        ("**Resposta: D**", "D"),
        ("**Resposta:** B", "B"),
        ("Resposta: D.", "D"),
        ("**Resposta: A**.", "A"),
        ("B", "B"),  # completa o "Resposta:" do prompt
        ("C) 42", "C"),
        ("A. A parcela do encargo referente às instalações da distribuidora.", "A"),
        # Outras letras mencionadas não contam; vale a última resposta no formato:
        ("A alternativa correta é C, pois a alternativa B está errada.\n\nResposta: C", "C"),
        ("Resposta: C\n\nA letra B estaria certa se a norma fosse outra.", "C"),
        ("Resposta: A\n\nCorrigindo o cálculo… Resposta: C", "C"),
        # Fora do formato pedido não conta:
        ("A resposta correta é a letra E.", None),
        ("Analisando: a opção A está errada. Resposta final: B", None),
        ("Resposta: a alternativa B", None),
        ("\\boxed{A}", None),
        # Nenhuma alternativa:
        ("Resposta: Brasília", None),  # palavra que começa com maiúscula não é alternativa
        ("Não sei responder.", None),
        ("Resposta: F", None),  # fora das alternativas do item
    ],
)
def test_extract_letter(text, expected):
    assert extract_letter(text, "ABCDE") == expected


def test_split_reasoning():
    assert split_reasoning("<think>calculando…</think>\nResposta: B") == ("Resposta: B", "calculando…")
    assert split_reasoning("Resposta: B") == ("Resposta: B", None)
