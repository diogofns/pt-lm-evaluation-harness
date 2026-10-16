"""Conversão declarativa de linhas em `MCQItem`, para datasets com colunas comuns."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..core.types import LETTERS, MCQItem

ANSWER_FORMATS = ("auto", "letter", "index", "text")


def columns(
    *,
    question: str,
    options: str | Sequence[str],
    answer: str,
    id: str | None = None,
    subject: str | None = None,
    context: str | None = None,
    answer_is: str = "auto",
) -> Callable[[dict[str, Any]], MCQItem]:
    """Monta um `to_item` a partir de nomes de colunas.

    - `options`: caminho para uma lista (ex.: "choices.text") ou lista de colunas
      (ex.: ["A", "B", "C", "D"]).
    - `answer`: a coluna do gabarito; `answer_is` diz como lê-la: "letter" ("B", "(b)"),
      "index" (0-based), "text" (o texto da opção correta) ou "auto". Declare quando
      souber: em "auto", um gabarito que vale como letra e como texto de outra opção
      (ex.: "B" com as opções "A", "~A", "B", "~B") é erro, para nunca ser adivinhado.
    - Caminhos com ponto acessam campos aninhados.
    """
    if answer_is not in ANSWER_FORMATS:
        raise ValueError(f"answer_is deve ser um de {ANSWER_FORMATS}, veio {answer_is!r}")

    def to_item(row: dict[str, Any]) -> MCQItem:
        if isinstance(options, str):
            opts = [str(o) for o in get_path(row, options)]
        else:
            opts = [str(get_path(row, col)) for col in options]
        return MCQItem(
            id=str(get_path(row, id)) if id else default_id(row),
            question=str(get_path(row, question)),
            options=tuple(opts),
            gold=answer_index(get_path(row, answer), opts, answer_is),
            subject=_optional(row, subject),
            context=_optional(row, context),
        )

    return to_item


def _optional(row: dict[str, Any], path: str | None) -> str | None:
    """Coluna opcional: ausente, nula ou vazia vira None (e não a string "None")."""
    if not path:
        return None
    value = get_path(row, path)
    return str(value).strip() or None if value is not None else None


def get_path(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for key in path.split("."):
        value = value[key]
    return value


def default_id(row: dict[str, Any]) -> str:
    return f"{row['_split']}-{row['_index']}"


def letter_index(value: str) -> int:
    """'B', ' b ', '(B)', 'B)' → 1."""
    letter = str(value).strip().strip("()").strip().upper()
    if len(letter) != 1 or letter not in LETTERS:
        raise ValueError(f"não é uma letra de alternativa: {value!r}")
    return LETTERS.index(letter)


def answer_index(value: Any, options: Sequence[str], answer_is: str = "auto") -> int:
    """Índice da opção correta, lendo `value` como `answer_is` manda (ver `columns`)."""
    if answer_is == "index" or (
        answer_is == "auto" and isinstance(value, int) and not isinstance(value, bool)
    ):
        return int(value)
    text = str(value).strip()
    if answer_is == "letter":
        return letter_index(text)
    by_text = list(options).index(text) if text in options else None
    if answer_is == "text":
        if by_text is None:
            raise ValueError(f"gabarito {text!r} não é o texto de nenhuma opção")
        return by_text
    try:
        by_letter: int | None = letter_index(text)
    except ValueError:
        by_letter = None
    if by_text is not None and by_letter is not None and by_text != by_letter:
        raise ValueError(
            f"gabarito {text!r} ambíguo: é a letra da opção {by_letter} e o texto da opção {by_text}; "
            "declare answer_is em columns()"
        )
    if by_text is not None:
        return by_text
    if by_letter is not None:
        return by_letter
    raise ValueError(f"gabarito {text!r} não é letra, índice nem texto de uma opção")
