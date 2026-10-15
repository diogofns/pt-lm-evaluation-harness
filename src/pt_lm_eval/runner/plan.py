"""Plano de execução: modelo × benchmark × avaliação, cruzando o que a avaliação
precisa com o que o backend oferece. Nada aqui conhece um plugin pelo nome.

Nomes com prefixo `lm_eval:` são tarefas do lm-evaluation-harness, executadas pela
ponte em `integrations/lm_eval.py`, com a avaliação "lm_eval"."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.protocols import BACKENDS, BENCHMARKS, EVALUATIONS
from ..integrations.lm_eval import is_lm_eval, task_needs
from .config import BenchmarkRun, ExperimentConfig, ModelConfig


@dataclass(frozen=True)
class Step:
    model: ModelConfig
    benchmark: str
    evaluation: str
    num_fewshot: int | None  # None = padrão da tarefa (só lm_eval)
    fewshot_strategy: str
    limit: int | None
    skip_reason: str | None = None


def build_plan(cfg: ExperimentConfig) -> list[Step]:
    steps = []
    for model in cfg.models:
        capabilities = BACKENDS.get(model.backend).capabilities(model)
        for run in cfg.benchmarks:
            for evaluation in _evaluations(run):
                missing = needs(run.name, evaluation) - capabilities
                if is_lm_eval(run.name):  # sem num_fewshot explícito, vale o padrão da tarefa
                    shots = run.num_fewshot
                else:
                    shots = cfg.num_fewshot if run.num_fewshot is None else run.num_fewshot
                steps.append(
                    Step(
                        model=model,
                        benchmark=run.name,
                        evaluation=evaluation,
                        num_fewshot=shots,
                        fewshot_strategy=run.fewshot_strategy or cfg.fewshot_strategy,
                        limit=run.limit if run.limit is not None else cfg.limit,
                        skip_reason=f"backend sem {', '.join(sorted(missing))}" if missing else None,
                    )
                )
    return steps


def needs(benchmark: str, evaluation: str) -> frozenset[str]:
    return task_needs(benchmark) if is_lm_eval(benchmark) else EVALUATIONS.get(evaluation).needs


def _evaluations(run: BenchmarkRun) -> tuple[str, ...]:
    if is_lm_eval(run.name):
        return ("lm_eval",)
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
            kind = "/".join(sorted(needs(s.benchmark, s.evaluation)))
            chat = " (chat)" if s.model.uses_chat and kind == "generate" else ""
            status = f"{kind}{chat}"
        rows.append(
            (
                s.model.display_name,
                s.benchmark,
                s.evaluation,
                "padrão" if s.num_fewshot is None else str(s.num_fewshot),
                status,
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)
