"""Extração da alternativa escolhida em respostas geradas.

Segue o guia de avaliação do OpenEvals: o prompt pede um formato ("Resposta: X"),
o raciocínio é removido antes (`split_reasoning`) e a resposta é normalizada
(maiúsculas, markdown, parênteses, pontuação) antes da comparação exata.

- `extract_letter`: só o formato pedido. É a base de `acc`.
- `extract_letter_flexible`: o formato pedido ou, se ele faltar, menções como
  "a alternativa correta é C" e "\\boxed{C}". É a base de `acc_flex`, para
  separar quem não sabe de quem sabe mas ignora o formato.
"""

from __future__ import annotations

import re

_THINK_END = "</think>"

# "Resposta: X". A letra maiúscula não pode ser início de palavra ("Resposta: Brasília");
# minúscula só entre parênteses ou no fim da linha, para não confundir o artigo "a".
_FORMAT = re.compile(
    r"(?i:resposta)\**\s*:\s*\**\s*"
    r"(?:\(([A-Ja-j])\)|([A-J])(?![A-Za-zÀ-ÿ])|([a-j])(?=\s*[.)]?\**\s*$))",
    re.M,
)
# A resposta inteira é a letra, ou começa com "B)"/"B.": completa o "Resposta:" do prompt.
_ONLY_LETTER = re.compile(r"^\s*\**\(?([A-J])\)?[.):]?\**\s*$")
_LEADING_LETTER = re.compile(r"^\s*\**\(?([A-J])[).:]\**\s")

# Heurísticas do modo flexível.
_KEYWORD = r"(?i:resposta|alternativa|op[çc][ãa]o|letra|answer(?:\s+is)?)"
_QUALIFIER = r"(?:\s+(?i:final|correta|certa|correct))*"
_LINK = r"\s*(?:(?i:é|e|is)|:)?\s*:?\s*"
_ARTICLE = r"(?:(?i:a|o)\s+)?(?:(?i:letra|alternativa|op[çc][ãa]o)\s+)?"
_LETTER = r"\**\(?([A-J])\)?\**(?![A-Za-zÀ-ÿ])"
_FLEXIBLE = [
    re.compile(_KEYWORD + _QUALIFIER + _LINK + _ARTICLE + _LETTER),
    re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*\(?([A-J])\)?\s*\}?\s*\}"),
]


def split_reasoning(text: str) -> tuple[str, str | None]:
    """Separa '<think>…</think>resposta' em (resposta, raciocínio)."""
    if _THINK_END not in text:
        return text, None
    reasoning, _, answer = text.rpartition(_THINK_END)
    return answer.strip(), reasoning.replace("<think>", "").strip()


def extract_letter(text: str, letters: str) -> str | None:
    """Alternativa no formato pedido; a última, se o modelo se corrigir. None se não houver."""
    matches = _FORMAT.findall(text)
    if matches:
        return _valid(next(g for g in matches[-1] if g).upper(), letters)
    for pattern in (_ONLY_LETTER, _LEADING_LETTER):
        match = pattern.search(text)
        if match:
            return _valid(match.group(1), letters)
    return None


def extract_letter_flexible(text: str, letters: str) -> str | None:
    """O formato pedido ou, na falta dele, a última menção explícita a uma alternativa."""
    strict = extract_letter(text, letters)
    if strict is not None:
        return strict
    for pattern in _FLEXIBLE:
        matches = pattern.findall(text)
        if matches:
            return _valid(matches[-1], letters)
    return None


def _valid(letter: str, letters: str) -> str | None:
    return letter if letter in letters else None
