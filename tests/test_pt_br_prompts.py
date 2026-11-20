from dataclasses import replace

import pytest

from pt_lm_eval import BENCHMARKS, EVALUATIONS
from pt_lm_eval.benchmarks.mmlu_redux.benchmark import SUBJECTS
from pt_lm_eval.evaluations.labels_pt_br import MMLU_SUBJECTS_PT_BR


def mmmlu_row(subject):
    return {
        "_split": "test",
        "_index": 0,
        "Subject": subject,
        "Question": "Quanto é 2 + 2?",
        "A": "3",
        "B": "4",
        "C": "5",
        "D": "6",
        "Answer": "B",
    }


def test_mmlu_labels_cover_the_complete_subject_taxonomy():
    assert set(MMLU_SUBJECTS_PT_BR) == set(SUBJECTS)
    assert len(MMLU_SUBJECTS_PT_BR) == 57


@pytest.mark.parametrize("subject", SUBJECTS)
def test_mmmlu_translates_display_label_and_preserves_subject_id(subject):
    item = BENCHMARKS.get("mmmlu_pt_br").to_item(mmmlu_row(subject))
    request = EVALUATIONS.get("mcf").build(item, [replace(item, id="shot")])[0]
    assert item.subject == subject
    assert request.context.startswith("A seguir estão questões de múltipla escolha, com respostas, sobre ")
    assert f"{MMLU_SUBJECTS_PT_BR[subject]}, em português brasileiro.\n\n" in request.context
    assert request.context.endswith("\nResposta:")
    assert "\nResposta: B\n\n" in request.context
    assert "Answer:" not in request.context and "The following" not in request.context


def test_mmmlu_rejects_subjects_without_translation():
    item = BENCHMARKS.get("mmmlu_pt_br").to_item(mmmlu_row("unknown_subject"))
    with pytest.raises(ValueError, match="matéria sem tradução PT-BR"):
        EVALUATIONS.get("mcf").build(item, [])


@pytest.mark.parametrize(
    "domain,label",
    [
        ("Applied Science", "ciências aplicadas"),
        ("Arts & Humanities", "artes e humanidades"),
        ("Business & Commerce", "negócios e comércio"),
        ("Health oriented education", "formação em saúde"),
        ("Social Science", "ciências sociais"),
        ("STEM", "ciências, tecnologia, engenharia e matemática"),
    ],
)
def test_include_translates_domain_header_and_preserves_task_group(domain, label):
    item = BENCHMARKS.get("include_pt").to_item(
        {
            "_split": "test",
            "_index": 0,
            "question": "Quanto é 2 + 2?",
            "option_a": "3",
            "option_b": "4",
            "option_c": "5",
            "option_d": "6",
            "answer": 1,
            "subject": "math",
            "domain": domain,
        }
    )
    request = EVALUATIONS.get("mcf").build(item, [])[0]
    assert item.task_group == domain and item.subject == "math"
    assert request.context.startswith(f"A questão de múltipla escolha a seguir é sobre {label}.\n\nQuanto")
    assert request.context.endswith("\nResposta:")
