"""Extração da alternativa escolhida em respostas geradas."""

from __future__ import annotations

import re

_THINK_END = "</think>"

# Palavras-chave em minúsculas ou maiúsculas; a letra precisa ser maiúscula,
# para não confundir o artigo "a" com a alternativa A.
_KEYWORD = r"(?i:resposta|alternativa|op[çc][ãa]o|letra|answer(?:\s+is)?)"
_QUALIFIER = r"(?:\s+(?i:final|correta|certa|correct))*"
_LINK = r"\s*(?:(?i:é|e|is)|:)?\s*:?\s*"
_ARTICLE = r"(?:(?i:a|o)\s+)?(?:(?i:letra|alternativa|op[çc][ãa]o)\s+)?"
_LETTER = r"\**\(?([A-J])\)?\**(?![A-Za-zÀ-ÿ])"

_PATTERNS = [
    re.compile(_KEYWORD + _QUALIFIER + _LINK + _ARTICLE + _LETTER),
    re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-J])\)?\s*\}?\s*\}"),
]
_ONLY_LETTER = re.compile(r"^\s*\**\(?([A-J])\)?[.):]?\**\s*$")
_LEADING_LETTER = re.compile(r"^\s*\**\(?([A-J])[).:]\**\s")


def split_reasoning(text: str) -> tuple[str, str | None]:
    """Separa '<think>…</think>resposta' em (resposta, raciocínio)."""
    if _THINK_END not in text:
        return text, None
    reasoning, _, answer = text.rpartition(_THINK_END)
    return answer.strip(), reasoning.replace("<think>", "").strip()


def extract_letter(text: str, letters: str) -> str | None:
    """Alternativa escolhida, ou None se não houver uma inequívoca.

    Ordem: última menção explícita ("Resposta: B", "a alternativa correta é C",
    "answer is (D)", "\\boxed{A}"); resposta que é só a letra; resposta que
    começa com "B)" ou "B.".
    """
    for pattern in _PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return _valid(matches[-1], letters)
    for pattern in (_ONLY_LETTER, _LEADING_LETTER):
        match = pattern.search(text)
        if match:
            return _valid(match.group(1), letters)
    return None


def _valid(letter: str, letters: str) -> str | None:
    return letter if letter in letters else None
