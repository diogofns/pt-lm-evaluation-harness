"""Resultados (`*.results.json`), amostras (`*.samples.jsonl`), estatística e o CSV do experimento.

Cada amostra guarda o que é preciso para auditar um número (prompts, continuações,
few-shot, métricas, logprobs ou resposta); o resultado agrega as amostras e registra
como reproduzi-las. O CSV é reconstruído de todos os resultados do experimento.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core import Request
from ..inference.server import slug

if TYPE_CHECKING:
    from ..benchmarks import Benchmark
    from .config import ExperimentConfig, Step

SCHEMA = "pt-lm-eval/results@1"


def describe_requests(requests: Sequence[Request]) -> dict[str, Any]:
    """Prompts sem repetição (o mesmo contexto serve a todas as opções), continuações e conversas."""
    prompts: list[str] = []
    for r in requests:
        text = r.context if hasattr(r, "context") else r.prompt
        if text not in prompts:
            prompts.append(text)
    out: dict[str, Any] = {"prompts": prompts}
    if continuations := [r.continuation for r in requests if hasattr(r, "continuation")]:
        out["continuations"] = continuations
    if messages := [list(r.messages) for r in requests if getattr(r, "messages", None)]:
        out["messages"] = messages
    return out


# --- Estatística ------------------------------------------------------------------


def mean_stderr(values: Sequence[float]) -> tuple[float, float]:
    """Média e erro-padrão da média (desvio-padrão amostral / √n), como no lm-eval."""
    return weighted_mean_stderr(values, [1.0] * len(values))


def weighted_mean_stderr(values: Sequence[float], weights: Sequence[float]) -> tuple[float, float]:
    """Média ponderada Σwx/Σw e erro-padrão de razão, com os itens como unidades
    (com pesos iguais, é o erro-padrão usual da média)."""
    n, total = len(values), sum(weights)
    if n == 0 or total == 0:
        return math.nan, math.nan
    mean = sum(w * v for v, w in zip(values, weights, strict=True)) / total
    if n == 1:
        return mean, math.nan
    spread = sum((w * (v - mean)) ** 2 for v, w in zip(values, weights, strict=True))
    return mean, math.sqrt(spread * n / (n - 1)) / total


def macro_mean_stderr(groups: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Média simples entre grupos (matérias) e seu erro-padrão: Var = Σ s²/nᵢ / k², com s² a
    variância de todos os itens juntos (grupos com 1 item não contam como variância zero)."""
    groups = [g for g in groups if g]
    k = len(groups)
    if k == 0:
        return math.nan, math.nan
    mean = sum(sum(g) / len(g) for g in groups) / k
    values = [v for g in groups for v in g]
    if len(values) < 2:
        return mean, math.nan
    overall = sum(values) / len(values)
    pooled_var = sum((v - overall) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(pooled_var * sum(1 / len(g) for g in groups)) / k


def aggregate(records: list[dict], cf_normalization: str = "char") -> dict[str, Any]:
    """Média ± stderr de cada métrica sobre os itens (micro), como o `weight_by_size` do lm-eval.

    Com várias matérias, a média entre matérias fica em `metrics_macro` e cada uma em
    `by_subject`. No CF, `acc_norm` repete a normalização fixada pelo benchmark.
    """
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_subject[r.get("subject") or "(sem matéria)"].append(r)
    micro = _mean_metrics(records, cf_normalization)
    if len(by_subject) == 1:
        return {"metrics": micro}
    macro: dict[str, float] = {}
    for name in _metric_names(records):
        groups = [[r["metrics"][name] for r in rs if name in r["metrics"]] for rs in by_subject.values()]
        macro[name], macro[f"{name}_stderr"] = macro_mean_stderr(groups)
    subjects = {
        s: {"n": len(rs), **_mean_metrics(rs, cf_normalization)} for s, rs in sorted(by_subject.items())
    }
    return {
        "metrics": micro,
        "metrics_macro": _with_acc_norm(macro, cf_normalization),
        "by_subject": subjects,
    }


def _mean_metrics(records: list[dict], cf_normalization: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in _metric_names(records):
        rs = [r for r in records if name in r["metrics"]]
        weights = [r.get("weights", {}).get(name, 1.0) for r in rs]
        out[name], out[f"{name}_stderr"] = weighted_mean_stderr([r["metrics"][name] for r in rs], weights)
    return _with_acc_norm(out, cf_normalization)


def _with_acc_norm(metrics: dict[str, float], cf_normalization: str) -> dict[str, float]:
    source = f"acc_{cf_normalization}"
    if source not in metrics:
        return metrics
    return {**metrics, "acc_norm": metrics[source], "acc_norm_stderr": metrics[f"{source}_stderr"]}


def _metric_names(records: list[dict]) -> list[str]:
    return list(dict.fromkeys(name for r in records for name in r["metrics"]))


# --- Arquivos ---------------------------------------------------------------------


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
    fingerprint: str | None = None,
    timing: dict[str, Any] | None = None,
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
            "fewshot_as_multiturn": step.fewshot_as_multiturn,
            "cf_normalization": bench.cf_normalization,
            "seed": cfg.seed,
            "limit": step.limit,
            "exclude_ids": list(step.exclude_ids),
            "fingerprint": fingerprint,
        },
        "timing": timing,
        "model": {
            **{k: v for k, v in asdict(model).items() if v not in (None, {}, "")},
            "display_name": model.display_name,
            "server": None if model.base_url else {**asdict(cfg.server), "args": cfg.server.args_for(model)},
        },
        **aggregate(records, bench.cf_normalization),
        "samples": samples_path,
    }


class Results:
    """Os arquivos de um experimento: por passo, `*.samples.jsonl` e `*.results.json`; no
    fim, o CSV consolidado, reconstruído de todos os resultados."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)

    def paths(self, step: Step) -> tuple[Path, Path]:
        """(amostras, resultado) do passo, em <out_dir>/<modelo>/<benchmark>.<avaliação>.*"""
        stem = self.out_dir / slug(step.model.display_name) / f"{slug(step.benchmark)}.{step.evaluation}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        return Path(f"{stem}.samples.jsonl"), Path(f"{stem}.results.json")

    @staticmethod
    def write(result: dict[str, Any], path: Path) -> None:
        path.write_text(json.dumps(_clean(result), ensure_ascii=False, indent=2, default=str) + "\n")

    def collect(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.out_dir.glob("*/*.results.json"))]

    def write_csv(self, name: str) -> Path:
        """Uma linha por modelo, uma coluna por benchmark@versão|avaliação|métrica (mais o tempo
        em `duration_seconds`); versões diferentes de um benchmark nunca dividem coluna."""
        rows: dict[str, dict[str, Any]] = {}
        columns: dict[str, None] = {}
        for r in self.collect():
            model = r["model"].get("display_name") or r["model"]["name"]
            prefix = f"{r['benchmark']['name']}@{r['benchmark']['version']}|{r['evaluation']['name']}"
            values = dict(r["metrics"])
            if (r.get("timing") or {}).get("seconds") is not None:
                values["duration_seconds"] = r["timing"]["seconds"]
            row = rows.setdefault(model, {"model": model})
            for metric, value in values.items():
                columns[f"{prefix}|{metric}"] = None
                row[f"{prefix}|{metric}"] = "" if value is None or math.isnan(value) else round(value, 6)
        path = self.out_dir / f"{name}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["model", *sorted(columns)])
            writer.writeheader()
            writer.writerows(rows.values())
        return path


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
