"""Formato dos resultados (`*.results.json`) e das amostras (`*.samples.jsonl`).

Por amostra fica tudo o que é preciso para auditar um número: prompts,
continuações, few-shot usados, métricas e detalhes (logprobs, resposta, letra
extraída). O resultado agrega as amostras e registra como reproduzi-las.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.types import Request
from ..metrics.stats import macro_mean_stderr, mean_stderr

if TYPE_CHECKING:
    from ..core.protocols import Benchmark
    from ..runner.config import ExperimentConfig
    from ..runner.plan import Step

SCHEMA = "pt-lm-eval/results@1"


def describe_requests(requests: list[Request]) -> dict[str, Any]:
    """Prompts sem repetição (o mesmo contexto serve a todas as opções) e continuações."""
    prompts: list[str] = []
    for r in requests:
        text = getattr(r, "context", None)
        text = getattr(r, "prompt", "") if text is None else text
        if text not in prompts:
            prompts.append(text)
    out: dict[str, Any] = {"prompts": prompts}
    continuations = [r.continuation for r in requests if hasattr(r, "continuation")]
    if continuations:
        out["continuations"] = continuations
    return out


def aggregate(records: list[dict]) -> dict[str, Any]:
    """Média ± stderr por métrica; com várias matérias, a principal é a média entre matérias."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_subject[r.get("subject") or "(sem matéria)"].append(r)
    micro = _mean_metrics(records)
    if len(by_subject) == 1:
        return {"metrics": micro}
    subjects = {s: {"n": len(rs), **_mean_metrics(rs)} for s, rs in sorted(by_subject.items())}
    macro = {}
    for name in _metric_names(records):
        groups = [[r["metrics"][name] for r in rs if name in r["metrics"]] for rs in by_subject.values()]
        macro[name], macro[f"{name}_stderr"] = macro_mean_stderr(groups)
    return {"metrics": macro, "metrics_micro": micro, "by_subject": subjects}


def build_result(
    *,
    cfg: ExperimentConfig,
    step: Step,
    bench: Benchmark,
    n_items: int,
    n_discarded: int,
    prompt_sha: str,
    records: list[dict],
    samples_path: str,
) -> dict[str, Any]:
    model = step.model
    return {
        "schema": SCHEMA,
        "pt_lm_eval": package_version(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment": cfg.experiment,
        "benchmark": {
            "name": step.benchmark,
            "version": bench.version,
            "source": bench.source.describe(),
            "n_items": n_items,
            "n_discarded": n_discarded,
            "n_evaluated": len(records),
        },
        "evaluation": {
            "name": step.evaluation,
            "prompt_sha": prompt_sha,
            "num_fewshot": step.num_fewshot,
            "fewshot_strategy": step.fewshot_strategy,
            "seed": cfg.seed,
            "limit": step.limit,
        },
        "model": {
            **{k: v for k, v in asdict(model).items() if v not in (None, {}, "")},
            "server": None
            if model.base_url
            else {**asdict(cfg.server), "args": {**cfg.server.args, **model.server_args}},
        },
        **aggregate(records),
        "samples": samples_path,
    }


def write_result(result: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(_clean(result), ensure_ascii=False, indent=2, default=str) + "\n")


def _mean_metrics(records: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in _metric_names(records):
        mean, stderr = mean_stderr([r["metrics"][name] for r in records if name in r["metrics"]])
        out[name], out[f"{name}_stderr"] = mean, stderr
    return out


def _metric_names(records: list[dict]) -> list[str]:
    names: dict[str, None] = {}
    for r in records:
        names.update(dict.fromkeys(r["metrics"]))
    return list(names)


def package_version() -> str:
    try:
        return version("pt-lm-eval")
    except PackageNotFoundError:
        return "dev"


def _clean(value: Any) -> Any:
    """NaN (ex.: stderr com uma única amostra) vira null, para o JSON ser válido."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value
