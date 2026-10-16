"""Prompts, pedidos e pontuação das avaliações de múltipla escolha."""

from __future__ import annotations

import pytest

from pt_lm_eval import EVALUATIONS, MCQItem
from pt_lm_eval.core.types import GenerateResponse, LoglikelihoodResponse
from pt_lm_eval.metrics.extraction import extract_letter, extract_letter_flexible, split_reasoning

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


def test_cf_pmi_adds_unconditional_requests():
    ev = EVALUATIONS.get("cf_pmi")
    reqs = ev.build(ITEM, [])
    assert len(reqs) == 6 and reqs[3].context == "Resposta:"
    responses = [LoglikelihoodResponse(lp, False, 1) for lp in (-3.0, -2.0, -1.0, -4.0, -0.5, -0.1)]
    scored = ev.score(ITEM, responses)
    assert scored.metrics["acc"] == 0.0  # " 5" tem o maior logprob
    assert scored.metrics["acc_pmi"] == 0.0  # -3+4=1, -2+0.5=-1.5, -1+0.1=-0.9 → opção A
    assert scored.details["pred"] == "C"


def test_gen_prompt_and_score():
    ev = EVALUATIONS.get("gen")
    (req,) = ev.build(ITEM, [SHOT])
    assert req.prompt.endswith(
        'Responda com a letra da alternativa correta no formato "Resposta: X".\nResposta:'
    )
    assert "Resposta: A\n\nPergunta: Quanto é 2 + 2?" in req.prompt
    ok = ev.score(ITEM, [GenerateResponse("Resposta: B", finish_reason="stop")])
    assert ok.metrics == {
        "acc": 1.0,
        "extracted": 1.0,
        "acc_flex": 1.0,
        "truncated": 0.0,
        "random_baseline": 1 / 3,
    }
    cut = ev.score(ITEM, [GenerateResponse("Vamos pensar passo a passo", finish_reason="length")])
    assert cut.metrics == {
        "acc": 0.0,
        "extracted": 0.0,
        "acc_flex": 0.0,
        "truncated": 1.0,
        "random_baseline": 1 / 3,
    }


def test_gen_multiturn_puts_each_shot_in_its_own_turns():
    ev = EVALUATIONS.get("gen")
    (req,) = ev.build(ITEM, [SHOT], multiturn=True)
    assert [m["role"] for m in req.messages] == ["user", "assistant", "user"]
    assert req.messages[0]["content"].startswith("Pergunta: Quanto é 1 + 1?\nA. 2")
    assert req.messages[1]["content"] == "Resposta: A"
    assert req.messages[2]["content"].endswith('no formato "Resposta: X".')
    assert req.stop == ()
    # Sem exemplos, multi-turn não muda nada.
    assert ev.build(ITEM, [], multiturn=True) == ev.build(ITEM, [])


def test_loglikelihood_records_random_baseline():
    responses = [LoglikelihoodResponse(lp, False, 1) for lp in (-3.0, -1.0, -2.0)]
    assert EVALUATIONS.get("mcf").score(ITEM, responses).metrics == {"acc": 1.0, "random_baseline": 1 / 3}


@pytest.mark.parametrize(
    "text,strict,flexible",
    [
        # No formato pedido, normalizado (maiúsculas, markdown, parênteses, pontuação):
        ("Resposta: B", "B", "B"),
        ("resposta: (C)", "C", "C"),
        ("Resposta: b", "B", "B"),
        ("**Resposta: D**", "D", "D"),
        ("**Resposta:** B", "B", "B"),
        ("Resposta: D.", "D", "D"),
        ("**Resposta: A**.", "A", "A"),
        ("B", "B", "B"),  # completa o "Resposta:" do prompt
        ("C) 42", "C", "C"),
        ("A. A parcela do encargo referente às instalações da distribuidora.", "A", "A"),
        # Outras letras mencionadas não contam; vale a última resposta no formato:
        ("A alternativa correta é C, pois a alternativa B está errada.\n\nResposta: C", "C", "C"),
        ("Resposta: C\n\nA letra B estaria certa se a norma fosse outra.", "C", "C"),
        ("Resposta: A\n\nCorrigindo o cálculo… Resposta: C", "C", "C"),
        # Fora do formato: só o modo flexível (acc_flex) aceita.
        ("A resposta correta é a letra E.", None, "E"),
        ("A alternativa correta é a C, pois…", None, "C"),
        ("Analisando: a opção A está errada. Resposta final: B", None, "B"),
        ("Resposta: a alternativa B", None, "B"),
        ("The answer is (D)", None, "D"),
        ("\\boxed{A}", None, "A"),
        ("A alternativa correta é: C. Furthermore.", None, "C"),
        ("A alternativa B é 5,0%, que é a resposta correta. ⏎ Portanto, a resposta correta é B.", None, "B"),
        ("Receita líquida: $195.000\n\nA resposta correta é: C.", None, "C"),
        # Nenhuma alternativa:
        ("Resposta: Brasília", None, None),  # palavra que começa com maiúscula não é alternativa
        ("Não sei responder.", None, None),
        ("Resposta: F", None, None),  # fora das alternativas do item
    ],
)
def test_extract_letter(text, strict, flexible):
    assert extract_letter(text, "ABCDE") == strict
    assert extract_letter_flexible(text, "ABCDE") == flexible


def test_split_reasoning():
    assert split_reasoning("<think>calculando…</think>\nResposta: B") == ("Resposta: B", "calculando…")
    assert split_reasoning("Resposta: B") == ("Resposta: B", None)
