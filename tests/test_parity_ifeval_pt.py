from __future__ import annotations

import collections
import glob
import json
import os
from pathlib import Path

import pytest

from pt_lm_eval.evaluations.ifeval import follows_loose, follows_strict

pytestmark = pytest.mark.parity

LANGDETECT = {
    "pt:change_case:english_capital",
    "pt:change_case:english_lowercase",
    "pt:language:response_language",
}


def _known_divergence(instruction: str, kwargs: dict, key: int) -> bool:
    if instruction in LANGDETECT:
        return True
    if instruction == "pt:detectable_content:postscript" and kwargs.get("postscript_marker") == "OBS.":
        return True
    return instruction == "pt:combination:repeat_prompt" and key == 50


@pytest.mark.parametrize("mode", ["strict", "loose"])
def test_checkers_match_published_verdicts(mode):
    root = os.environ.get("PT_LM_EVAL_IFEVAL_REF")
    if not root:
        pytest.skip("defina PT_LM_EVAL_IFEVAL_REF com o repositório de referência")
    root = Path(root)
    rows = {}
    for line in (root / "data" / "pt_input_data.jsonl").read_text().splitlines():
        row = json.loads(line)
        rows[row["prompt"]] = row
    runs = os.environ.get("PT_LM_EVAL_IFEVAL_RUNS", "eval_open_1")
    files = sorted(glob.glob(str(root / "experiments" / runs / "pt_*" / f"eval_results_{mode}.jsonl")))
    assert files, f"nenhum resultado em experiments/{runs}"
    follows = follows_strict if mode == "strict" else follows_loose

    checked, mismatches = 0, collections.Counter()
    for path in files:
        for line in Path(path).read_text().splitlines():
            result = json.loads(line)
            row = rows[result["prompt"]]
            response = result["response"] if isinstance(result["response"], str) else ""
            for instruction, kwargs, expected in zip(
                row["instruction_id_list"], row["kwargs"], result["follow_instruction_list"], strict=True
            ):
                kwargs = kwargs if isinstance(kwargs, dict) else {}
                if _known_divergence(instruction, kwargs, row["key"]):
                    continue
                checked += 1
                if follows(response, instruction, kwargs, row["prompt"]) != expected:
                    mismatches[instruction] += 1
    print(f"\n{mode}: {checked} vereditos de {len(files)} modelos; diferenças: {dict(mismatches)}")
    assert not mismatches
