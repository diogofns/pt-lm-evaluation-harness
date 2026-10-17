"""Verificadores do IFEval PT, strict/loose e agregação por instrução."""

from __future__ import annotations

import math

import pytest

from pt_lm_eval import EVALUATIONS, InstructionItem
from pt_lm_eval.core import GenerateResponse
from pt_lm_eval.ifeval import CHECKERS, follows_loose, follows_strict
from pt_lm_eval.report import aggregate, mean_stderr, weighted_mean_stderr


def ok(instruction: str, text: str, **kw) -> bool:
    return follows_strict(text, f"pt:{instruction}", kw, prompt="")


def test_registry_covers_every_instruction_in_the_dataset():
    used = """change_case:capital_word_frequency change_case:english_capital change_case:english_lowercase
    combination:repeat_prompt combination:two_responses detectable_content:mesoclise_verification
    detectable_content:number_placeholders detectable_content:postscript detectable_content:specific_porque
    detectable_content:vos_address_verification detectable_format:constrained_response
    detectable_format:json_format detectable_format:multiple_sections detectable_format:number_bullet_lists
    detectable_format:number_highlighted_sections detectable_format:title keywords:cedilha_frequency
    keywords:existence keywords:forbidden_words keywords:frequency keywords:letter_frequency
    language:response_language length_constraints:nth_paragraph_first_word
    length_constraints:number_paragraphs length_constraints:number_sentences length_constraints:number_words
    punctuation:crase_presence punctuation:no_comma punctuation:no_tilde
    startend:end_checker startend:quotation"""
    assert {f"pt:{i}" for i in used.split()} <= set(CHECKERS)


@pytest.mark.parametrize(
    "instruction,text,kw,expected",
    [
        ("punctuation:no_comma", "Sem vírgulas aqui.", {}, True),
        ("punctuation:no_comma", "Uma, duas.", {}, False),
        ("punctuation:no_tilde", "Sem acento til.", {}, True),
        ("punctuation:no_tilde", "Não pode.", {}, False),
        ("punctuation:crase_presence", "Fui à praia.", {}, True),
        ("punctuation:crase_presence", "Fui a praia.", {}, False),
        ("keywords:cedilha_frequency", "Ação e coração.", {"count": 2}, True),
        ("keywords:cedilha_frequency", "Ação.", {"count": 2}, False),
        ("keywords:existence", "Saudade e SAÚDE.", {"keywords": ["saudade", "saúde"]}, True),
        ("keywords:forbidden_words", "O gato dorme.", {"forbidden_words": ["gato"]}, False),
        ("keywords:forbidden_words", "Os gatos dormem.", {"forbidden_words": ["gato"]}, True),
        (
            "keywords:frequency",
            "rio rio Rio",
            {"keyword": "rio", "frequency": 3, "relation": "ao menos"},
            True,
        ),
        (
            "keywords:letter_frequency",
            "banana",
            {"letter": "a", "let_frequency": 3, "let_relation": "menos que"},
            False,
        ),
        ("detectable_content:mesoclise_verification", "Dir-lhe-ei a verdade.", {}, True),
        ("detectable_content:vos_address_verification", "Vós sois bem-vindos.", {}, True),
        ("detectable_content:vos_address_verification", "Você é bem-vindo.", {}, False),
        ("detectable_content:postscript", "Texto.\nP.S. até logo", {"postscript_marker": "P.S."}, True),
        ("detectable_content:postscript", "Texto.\nOBS: até logo", {"postscript_marker": "OBS."}, True),
        ("detectable_content:number_placeholders", "Olá [nome], de [cidade].", {"num_placeholders": 2}, True),
        ("detectable_format:title", "<<Poema>>\nVersos.", {}, True),
        ("detectable_format:title", "<< >>\nVersos.", {}, False),
        ("detectable_format:json_format", '```json\n{"a": 1}\n```', {}, True),
        ("detectable_format:json_format", "{a: 1}", {}, False),
        ("detectable_format:number_bullet_lists", "* um\n* dois\n- três", {"num_bullets": 3}, True),
        ("detectable_format:number_highlighted_sections", "*um* e **dois**", {"num_highlights": 2}, True),
        (
            "detectable_format:multiple_sections",
            "SEÇÃO 1\na\nSEÇÃO 2\nb",
            {"section_spliter": "SEÇÃO", "num_sections": 2},
            True,
        ),
        ("detectable_format:constrained_response", "Minha resposta é Sim.", {}, True),
        ("length_constraints:number_paragraphs", "um\n***\ndois", {"num_paragraphs": 2}, True),
        ("length_constraints:number_paragraphs", "um\n***\n***\ndois", {"num_paragraphs": 2}, False),
        (
            "length_constraints:nth_paragraph_first_word",
            'Primeiro.\n\n"Bem, segundo.',
            {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "Bem"},
            True,
        ),
        ("combination:two_responses", "Uma.\n******\nOutra.", {}, True),
        ("combination:two_responses", "Igual.\n******\nIgual.", {}, False),
        ("combination:repeat_prompt", "Repita isto. Resposta.", {"prompt_to_repeat": "repita isto."}, True),
        ("startend:end_checker", 'Texto final. Até mais!"', {"end_phrase": "até mais!"}, True),
        ("startend:quotation", '"Tudo entre aspas."', {}, True),
        ("startend:quotation", 'Tudo "entre" aspas.', {}, False),
    ],
)
def test_checkers(instruction, text, kw, expected):
    assert ok(instruction, text, **kw) is expected


@pytest.mark.parametrize(
    "kind,text,expected",
    [
        ("porque", "Saí cedo porque choveu.", True),
        ("porque", "Saí cedo, o porque não sei.", False),  # depois de determinante é substantivo
        ("porquê", "Não entendi o porquê da decisão.", True),
        ("por que", "Por que você saiu?", True),
        ("por que", "Não sei por que ele saiu.", True),  # pergunta indireta
        ("por quê", "Você saiu por quê?", True),
        ("por quê", "Por quê você saiu", False),
        ("porque", "Porque sim, e porque não.", False),  # mais de uma ocorrência
    ],
)
def test_specific_porque(kind, text, expected):
    assert ok("detectable_content:specific_porque", text, porque_type=kind) is expected


def test_empty_response_never_follows_and_loose_drops_first_line_and_asterisks():
    assert not follows_strict("   ", "pt:punctuation:no_comma", {}, "")
    preamble = "Claro, aqui está:\n<<Título>>\nTexto sem vírgulas."
    assert not follows_strict(preamble, "pt:punctuation:no_comma", {}, "")
    assert follows_loose(preamble, "pt:punctuation:no_comma", {}, "")
    assert follows_loose('**"Entre aspas."**', "pt:startend:quotation", {}, "")


def test_weighted_mean_stderr_reduces_to_plain_mean_with_equal_weights():
    values = [1.0, 0.0, 0.5, 1.0]
    assert weighted_mean_stderr(values, [1, 1, 1, 1]) == pytest.approx(mean_stderr(values))
    mean, se = weighted_mean_stderr([1.0, 0.0], [3, 1])
    assert mean == 0.75 and se > 0
    assert math.isnan(weighted_mean_stderr([1.0], [2])[1])


def test_ifeval_scores_prompt_and_instruction_levels():
    ev = EVALUATIONS.get("ifeval")
    item = InstructionItem(
        id="1",
        prompt="Escreva sem vírgulas e sem til.",
        instruction_ids=("pt:punctuation:no_comma", "pt:punctuation:no_tilde"),
        kwargs=({}, None),
    )
    (request,) = ev.build(item, [])
    assert request.prompt == item.prompt and request.max_tokens == 2048
    half = ev.score(item, [GenerateResponse("Texto sem vírgulas mas com til: não.", finish_reason="stop")])
    assert half.metrics["prompt_level_strict_acc"] == 0.0
    assert half.metrics["inst_level_strict_acc"] == 0.5
    assert half.weights == {"inst_level_strict_acc": 2, "inst_level_loose_acc": 2}
    # A média por instrução pesa cada prompt pelo número de instruções: (1 + 2) / (1 + 2·1) ≠ média simples.
    records = [
        {"metrics": {"inst_level_strict_acc": 1.0}, "weights": {"inst_level_strict_acc": 1}},
        {"metrics": {"inst_level_strict_acc": 0.5}, "weights": {"inst_level_strict_acc": 2}},
    ]
    assert aggregate(records)["metrics"]["inst_level_strict_acc"] == pytest.approx(2 / 3)


def test_ifeval_rejects_unknown_instructions():
    item = InstructionItem(id="x", prompt="?", instruction_ids=("pt:nao:existe",), kwargs=({},))
    with pytest.raises(KeyError, match="sem verificador"):
        EVALUATIONS.get("ifeval").build(item, [])
