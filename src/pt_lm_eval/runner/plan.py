"""Plano de execução: modelo × benchmark × avaliação, cruzando o que a avaliação
precisa com o que o backend oferece. Nada aqui conhece um plugin pelo nome."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.protocols import BACKENDS, BENCHMARKS, EVALUATIONS
from .config import BenchmarkRun, ExperimentConfig, ModelConfig


@dataclass(frozen=True)
class Step:
    model: ModelConfig
    benchmark: str
    evaluation: str
    num_fewshot: int
    fewshot_strategy: str
    limit: int | None
    skip_reason: str | None = None


def build_plan(cfg: ExperimentConfig) -> list[Step]:
    steps = []
    for model in cfg.models:
        capabilities = BACKENDS.get(model.backend).capabilities(model)
        for run in cfg.benchmarks:
            for evaluation in _evaluations(run):
                missing = EVALUATIONS.get(evaluation).needs - capabilities
                steps.append(
                    Step(
                        model=model,
                        benchmark=run.name,
                        evaluation=evaluation,
                        num_fewshot=cfg.num_fewshot if run.num_fewshot is None else run.num_fewshot,
                        fewshot_strategy=run.fewshot_strategy or cfg.fewshot_strategy,
                        limit=run.limit if run.limit is not None else cfg.limit,
                        skip_reason=f"backend sem {', '.join(sorted(missing))}" if missing else None,
                    )
                )
    return steps


def _evaluations(run: BenchmarkRun) -> tuple[str, ...]:
    declared = BENCHMARKS.get(run.name).evaluations
    if run.evaluations is None:
        return declared
    unknown = [e for e in run.evaluations if e not in declared]
    if unknown:
        raise ValueError(f"{run.name}: avaliações {unknown} não declaradas no benchmark ({declared})")
    return run.evaluations


def format_plan(steps: list[Step]) -> str:
    rows = [("modelo", "benchmark", "avaliação", "shots", "situação")]
    for s in steps:
        if s.skip_reason:
            status = f"pulado: {s.skip_reason}"
        else:
            kind = "/".join(sorted(EVALUATIONS.get(s.evaluation).needs))
            chat = " (chat)" if s.model.uses_chat and kind == "generate" else ""
            status = f"{kind}{chat}"
        rows.append((s.model.display_name, s.benchmark, s.evaluation, str(s.num_fewshot), status))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)
