"""Os benchmarks: de onde vêm os dados (revisão fixada) e como cada linha vira um item.

Cada um tem um cartão em cards/<nome>.md e uma entrada em benchmarks.lock.json. Para
adicionar um benchmark, registre-o aqui, escreva o cartão e rode
`pt-lm-eval benchmarks check <nome> --update-lock` (ver CONTRIBUTING.md).
"""

from __future__ import annotations

import ast
import re
import warnings
from typing import Any

from .core import BENCHMARKS, Benchmark, InstructionItem, MCQItem
from .data import HFDataset, columns, default_id, letter_index

MC = ("mcf", "cf", "gen")


def register(name: str, **fields: Any) -> None:
    BENCHMARKS.register(name, Benchmark(evaluations=fields.pop("evaluations", MC), **fields))


# --- Múltipla escolha com colunas comuns ------------------------------------------

register(
    "energy_regulacao",
    version="1.0",
    description="Regulação do setor elétrico brasileiro (PRORET/ANEEL), 5 alternativas.",
    source=HFDataset(
        "cemig-ceia/energy-eval", split="train", revision="cac76ae9456fc3c9354d761880bc3c3a5b54d1e1"
    ),
    to_item=columns(
        id="id", question="question", options="choices.text", answer="answerKey", answer_is="letter"
    ),
)
register(
    "mmmlu_pt_br",
    version="1.1",
    description="MMLU traduzido profissionalmente para o português do Brasil (OpenAI MMMLU).",
    source=HFDataset(
        "openai/MMMLU", subset="PT_BR", split="test", revision="325a01dc3e173cac1578df94120499aaca2e2504"
    ),
    to_item=columns(
        question="Question",
        options=["A", "B", "C", "D"],
        answer="Answer",
        answer_is="letter",
        subject="Subject",
    ),
)
register(
    "mmlu_pt_sintetico",
    version="1.1",
    description="Questões no estilo MMLU geradas por Qwen3-235B-A22B em português.",
    source=HFDataset(
        "cemig-ceia/benchmark_Qwen3-235B-A22B_MMLU_v2",
        split="train",
        revision="831ba6bbae5b6057fbc3cae40e13eae7b55ebb26",
    ),
    to_item=columns(
        question="question", options="choices.text", answer="answer", answer_is="letter", subject="subject"
    ),
)
register(
    "oab_exams",
    version="1.0",
    description="Exame da OAB, 2010–2018 (4 alternativas); matéria = área do direito, quando há.",
    source=HFDataset(
        "eduagarcia/oab_exams", split="train", revision="fe810531d30e7799dd83c2e5be302ca2f2bdf930"
    ),
    to_item=columns(
        id="id",
        question="question",
        options="choices.text",
        answer="answerKey",
        answer_is="letter",
        subject="question_type",
    ),
)
_GLOBAL_MMLU = ("CohereLabs/Global-MMLU", "0e619dbeb34206cd48705a1a0ea7fb21cae09993")
register(
    "global_mmlu_pt",
    version="1.0",
    description="Global-MMLU em português (MMLU traduzido e revisado); exemplos few-shot do split dev.",
    source=HFDataset(_GLOBAL_MMLU[0], split="test", subset="pt", revision=_GLOBAL_MMLU[1]),
    fewshot_source=HFDataset(_GLOBAL_MMLU[0], split="dev", subset="pt", revision=_GLOBAL_MMLU[1]),
    to_item=columns(
        id="sample_id",
        question="question",
        options=["option_a", "option_b", "option_c", "option_d"],
        answer="answer",
        answer_is="letter",
        subject="subject",
    ),
)


# --- Belebele: compreensão de leitura, com a passagem como contexto ---------------


def _belebele(row: dict) -> MCQItem:
    return MCQItem(
        id=default_id(row),
        context=row["flores_passage"],
        question=row["question"],
        options=tuple(row[f"mc_answer{i}"] for i in range(1, 5)),
        gold=int(row["correct_answer_num"]) - 1,
    )


register(
    "belebele_pt",
    version="1.0",
    description="Belebele (por_Latn): compreensão de leitura sobre passagens do FLORES, 4 alternativas.",
    source=HFDataset(
        "facebook/belebele",
        split="test",
        subset="por_Latn",
        revision="7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
    ),
    to_item=_belebele,
)


# --- ENEM 2022–2024, como no lighteval (enem_adapter) -----------------------------


def _enem(row: dict) -> MCQItem | None:
    """Questões anuladas saem; a descrição de cada figura entra no lugar do `[[placeholder]]`."""
    if row["label"] == "Anulado":
        return None
    question = row["question"]
    for description in row["description"]:
        question = question.replace("[[placeholder]]", description, 1)
    return MCQItem(
        id=f"{row['exam']}-{row['id']}",
        question=question,
        options=tuple(row["alternatives"]),
        gold=letter_index(row["label"]),
        subject=row["exam"],
    )


register(
    "enem",
    version="1.0",
    description="ENEM 2022–2024 (5 alternativas); matéria = ano da prova.",
    source=HFDataset(
        "maritaca-ai/enem",
        split="train",
        subset=("2022", "2023", "2024"),
        revision="b2a7257356b0f56b740d57f23f8a575542ff6c0c",
    ),
    to_item=_enem,
)


# --- MMLU-PT: vestibulares, concursos e olimpíadas; cada split é uma área ---------

_LATEX_START = re.compile(r"(\\+)([A-Za-z])")  # barras antes de uma letra (início de comando LaTeX)


def _parse_alternatives(raw: Any) -> dict[str, Any]:
    """As alternativas vêm como texto de um dict Python, com LaTeX e quebras de linha."""
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


def _mmlu_pt(row: dict) -> MCQItem | None:
    alternatives = _parse_alternatives(row["alternatives"])
    # ~0,5% dos itens têm alternativas vazias (em geral, questões com imagem): descartados.
    if any(not str(a).strip() for a in alternatives["text"]):
        return None
    return MCQItem(
        id=f"{row['_split']}-{row['_index']}",
        question=row["statement"],
        options=tuple(alternatives["text"]),
        gold=letter_index(row["answer"]),
        subject=row["_split"],
    )


for _name, _repo, _revision in [
    (
        "mmlu_pt_undergraduate",
        "mmlu-pt/mmlu-pt-undergraduate-full",
        "98cc293cd60c3779c2b1da1dd5abb3b47d9dee42",
    ),
    (
        "mmlu_pt_undergraduate_hard",
        "mmlu-pt/mmlu-pt-undergraduate-hard",
        "7346b172a05c0286b57282fdf541ccdeba28df6a",
    ),
    ("mmlu_pt_high_school", "mmlu-pt/mmlu-pt-high-school-full", "b29d6ca3887d250114b543974fa8845a3143e229"),
    (
        "mmlu_pt_high_school_hard",
        "mmlu-pt/mmlu-pt-high-school-hard",
        "c40e827798003b02a7b563f2afa4f8413fa6c9f6",
    ),
]:
    register(
        _name,
        version="1.0",
        description=f"MMLU-PT ({_repo}); matérias = splits do dataset.",
        source=HFDataset(_repo, split="*", revision=_revision),
        to_item=_mmlu_pt,
        card="mmlu_pt",
    )


# --- IFEval PT --------------------------------------------------------------------


def _ifeval(row: dict) -> InstructionItem:
    return InstructionItem(
        id=str(row["key"]),
        prompt=row["prompt"],
        instruction_ids=row["instruction_id_list"],
        kwargs=row["kwargs"],
    )


register(
    "ifeval_pt",
    version="1.0",
    description="Portuguese-IFEval: 535 prompts, 1.238 instruções verificáveis (6 do português).",
    source=HFDataset(
        "gap-ufg/Portuguese-IFEval", split="test", revision="0ff3fe8d82a5cdb43153f86b301b2c9d5c6ea098"
    ),
    to_item=_ifeval,
    evaluations=("ifeval",),
)
