from __future__ import annotations

import ast
import re
import warnings
from typing import Any

_LATEX_START = re.compile(r"(\\+)([A-Za-z])")  # barras antes de uma letra (início de comando LaTeX)


def parse_alternatives(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise TypeError(f"alternativas: esperava str ou dict, veio {type(raw).__name__}")
    fixed = raw.replace("\r\n", "\\n").replace("\n", "\\n")
    # Número ímpar de barras antes de uma letra viraria escape inválido: completa para par.
    fixed = _LATEX_START.sub(lambda m: m.group(1) + ("\\" if len(m.group(1)) % 2 else "") + m.group(2), fixed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        parsed = ast.literal_eval(fixed)
    if not isinstance(parsed, dict) or "text" not in parsed:
        raise ValueError(f"alternativas sem o campo 'text': {raw[:120]!r}")
    return parsed
