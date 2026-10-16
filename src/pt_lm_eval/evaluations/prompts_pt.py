"""Textos dos prompts em português. Toda mudança aqui muda o hash dos prompts.

Formato de um exemplo MCF:

    Pergunta: <pergunta>
    A. <opção>
    B. <opção>
    Resposta: B

Exemplos few-shot são separados por uma linha em branco.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.types import MCQItem

FEWSHOT_DELIMITER = "\n\n"
TARGET_DELIMITER = " "
ANSWER_PREFIX = "Resposta:"
GEN_INSTRUCTION = 'Responda com a letra da alternativa correta no formato "Resposta: X".'
GEN_STOP = ("\n\nPergunta:",)


def question(item: MCQItem) -> str:
    lines = []
    if item.context:
        lines.append(f"Contexto: {item.context.strip()}")
    lines.append(f"Pergunta: {item.question.strip()}")
    return "\n".join(lines)


def options(item: MCQItem) -> str:
    return "\n".join(
        f"{letter}. {opt.strip()}" for letter, opt in zip(item.letters, item.options, strict=True)
    )


def mcf(item: MCQItem, *, with_answer: bool) -> str:
    text = f"{question(item)}\n{options(item)}\n{ANSWER_PREFIX}"
    return text + TARGET_DELIMITER + item.letters[item.gold] if with_answer else text


def cf(item: MCQItem, *, with_answer: bool) -> str:
    text = f"{question(item)}\n{ANSWER_PREFIX}"
    return text + TARGET_DELIMITER + item.options[item.gold].strip() if with_answer else text


def with_shots(shots: Sequence[str], query: str) -> str:
    return FEWSHOT_DELIMITER.join([*shots, query])


def gen(item: MCQItem, shots: Sequence[MCQItem]) -> str:
    # Termina em "Resposta:" para modelos base completarem com a letra; modelos
    # de chat respondem normalmente.
    query = f"{gen_question(item)}\n{ANSWER_PREFIX}"
    return with_shots([mcf(s, with_answer=True) for s in shots], query)


def gen_question(item: MCQItem) -> str:
    return f"{question(item)}\n{options(item)}\n\n{GEN_INSTRUCTION}"


def gen_messages(item: MCQItem, shots: Sequence[MCQItem]) -> tuple[dict[str, str], ...]:
    """Few-shot multi-turn: cada exemplo é uma pergunta do usuário e a resposta do assistente."""
    messages: list[dict[str, str]] = []
    for s in shots:
        messages.append({"role": "user", "content": gen_question(s)})
        messages.append({"role": "assistant", "content": f"{ANSWER_PREFIX} {s.letters[s.gold]}"})
    messages.append({"role": "user", "content": gen_question(item)})
    return tuple(messages)
