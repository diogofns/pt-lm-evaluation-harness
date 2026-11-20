from pt_lm_eval import BENCHMARKS, EVALUATIONS
from pt_lm_eval.core import LoglikelihoodResponse


def test_mmlu_prox_uses_reference_letter_and_omits_null_padding():
    row = {
        "question_id": 3787,
        "question": "Em qual das seguintes ligações a extremidade negativa não está escrita por último?",
        "answer": "C",
        "answer_index": 1,
        "category": "chemistry",
        "cot_content": "NAO_INCLUIR_RACIOCINIO_DO_GABARITO",
        **{
            f"option_{i}": v
            for i, v in enumerate(["H—O", "P—Cl", "N—H", "C—H", "S—O", "C—O", "Si—Cl", None, None, None])
        },
    }
    item = BENCHMARKS.get("mmlu_prox_pt").to_item(row)
    assert item.gold == 2 and item.options[item.gold] == "N—H"
    requests = EVALUATIONS.get("mcf").build(item, [])
    assert [r.continuation for r in requests] == [f" {x}" for x in "ABCDEFG"]
    assert all("NAO_INCLUIR" not in r.context and "None" not in r.context for r in requests)
    responses = [LoglikelihoodResponse(-1.0 if i == 2 else -5.0, False, 1) for i in range(7)]
    assert EVALUATIONS.get("mcf").score(item, responses).metrics["acc"] == 1.0


def test_global_piqa_keeps_brazilian_text_and_zero_based_answer():
    item = BENCHMARKS.get("global_piqa_nonparallel_pt_br").to_item(
        {
            "example_id": "por-test",
            "prompt": "Qual ingrediente é usado na paçoca?",
            "solution0": "gergelim",
            "solution1": "amendoim",
            "label": 1,
            "eng_translated0": "DO_NOT_INCLUDE_ENGLISH",
            "eng_translated1": "DO_NOT_INCLUDE_ENGLISH",
        }
    )
    requests = EVALUATIONS.get("cf").build(item, [])
    assert item.gold == 1 and item.options[item.gold] == "amendoim"
    assert all("DO_NOT_INCLUDE" not in r.context for r in requests)
    assert [r.continuation for r in requests] == [" gergelim", " amendoim"]
