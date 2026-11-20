from __future__ import annotations

from collections.abc import Sequence

from ..core import MCQItem
from .labels_pt_br import MMLU_SUBJECTS_PT_BR

FEWSHOT_DELIMITER = "\n\n"
TARGET_DELIMITER = " "
ANSWER_PREFIX = "Resposta:"
GEN_INSTRUCTION = 'Responda com a letra da alternativa correta no formato "Resposta: X".'
GEN_STOP = ("\n\nPergunta:",)


def question(item: MCQItem) -> str:
    if item.protocol in {"mmmlu", "include", "global_piqa"}:
        return item.question.strip()
    lines = [f"Contexto: {item.context.strip()}"] if item.context else []
    lines.append(f"Pergunta: {item.question.strip()}")
    return "\n".join(lines)


def _choice_question(item: MCQItem) -> str:
    return item.question.strip() if item.protocol == "global_mmlu" else question(item)


def options(item: MCQItem) -> str:
    if item.protocol in {"global_mmlu", "mmmlu", "include"}:
        # Preserve dataset whitespace, including the INCLUDE template's space before D.
        return "\n".join(
            f"{' ' if item.protocol == 'include' and letter == 'D' else ''}{letter}. {opt}"
            for letter, opt in zip(item.letters, item.options, strict=True)
        )
    return "\n".join(
        f"{letter}. {opt.strip()}" for letter, opt in zip(item.letters, item.options, strict=True)
    )


def mcf(item: MCQItem, *, with_answer: bool) -> str:
    text = f"{_choice_question(item)}\n{options(item)}\n{ANSWER_PREFIX}"
    return text + TARGET_DELIMITER + item.letters[item.gold] if with_answer else text


def description(item: MCQItem) -> str:
    if item.prompt_description:
        return item.prompt_description + "\n\n"
    if item.protocol == "mmmlu" and item.subject:
        try:
            subject = MMLU_SUBJECTS_PT_BR[item.subject]
        except KeyError as exc:
            raise ValueError(f"MMMLU: matéria sem tradução PT-BR: {item.subject!r}") from exc
        return (
            "A seguir estão questões de múltipla escolha, com respostas, sobre "
            f"{subject}, em português brasileiro.\n\n"
        )
    return ""


def cf(item: MCQItem, *, with_answer: bool) -> str:
    if item.protocol == "global_piqa":
        text = item.question
    else:
        text = f"{question(item)}\n{ANSWER_PREFIX}"
    if not with_answer:
        return text
    target = item.options[item.gold]
    if item.protocol != "global_piqa":
        target = target.strip()
    return text + TARGET_DELIMITER + target


def with_shots(shots: Sequence[str], query: str) -> str:
    return FEWSHOT_DELIMITER.join([*shots, query])


def gen_question(item: MCQItem) -> str:
    if item.protocol == "global_piqa":
        return (
            "Dada a situação a seguir, qual alternativa tem maior probabilidade de estar correta?\n\n"
            f"Situação:\n{item.question} ...\n\n"
            f"Alternativa A: {item.options[0]}\n\n"
            f"Alternativa B: {item.options[1]}\n\n"
            'Termine sua resposta no formato "Resposta: X", em que X deve ser A ou B.'
        )
    return f"{_choice_question(item)}\n{options(item)}\n\n{GEN_INSTRUCTION}"


def gen(item: MCQItem, shots: Sequence[MCQItem]) -> str:
    if item.protocol == "global_piqa":
        examples = [f"{gen_question(s)}\n{ANSWER_PREFIX} {s.letters[s.gold]}" for s in shots]
        return with_shots(examples, gen_question(item))
    # Termina em "Resposta:" para modelos base completarem com a letra.
    return description(item) + with_shots(
        [mcf(s, with_answer=True) for s in shots], f"{gen_question(item)}\n{ANSWER_PREFIX}"
    )


def gen_messages(item: MCQItem, shots: Sequence[MCQItem]) -> tuple[dict[str, str], ...]:
    """Few-shot multi-turn: cada exemplo é uma pergunta do usuário e a resposta do assistente."""
    messages: list[dict[str, str]] = []
    if prefix := description(item):
        messages.append({"role": "system", "content": prefix.rstrip()})
    for s in shots:
        messages.append({"role": "user", "content": gen_question(s)})
        messages.append({"role": "assistant", "content": f"{ANSWER_PREFIX} {s.letters[s.gold]}"})
    messages.append({"role": "user", "content": gen_question(item)})
    return tuple(messages)
