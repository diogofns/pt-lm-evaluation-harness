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
    # Efetivo: só vale com chat, geração e ao menos um exemplo (ou o padrão de uma tarefa lm_eval).
    fewshot_as_multiturn: bool = False


# Cenário de cada tipo de modelo (guia do OpenEvals): base por log-likelihood,
# instruct e API por geração. Cada benchmark roda as avaliações dele desse tipo.
SCENARIO = {
    "base": frozenset({"loglikelihood"}),
    "instruct": frozenset({"generate"}),
    "api": frozenset({"generate"}),
}


def build_plan(cfg: ExperimentConfig) -> list[Step]:
    steps = []
    for model in cfg.models:
        capabilities = BACKENDS.get(model.backend).capabilities(model)
        for run in cfg.benchmarks:
            evaluations = [e for e in _evaluations(run) if needs(run.name, e) <= SCENARIO[model.type]]
            if not evaluations:
                steps.append(
                    Step(
                        model=model,
                        benchmark=run.name,
                        evaluation="-",
                        num_fewshot=None,
                        fewshot_strategy=run.fewshot_strategy or cfg.fewshot_strategy,
                        limit=_limit(run, cfg),
                        skip_reason=f"não se aplica a modelos {model.type}",
                    )
                )
            for evaluation in evaluations:
                missing = needs(run.name, evaluation) - capabilities
                if is_lm_eval(run.name):  # sem num_fewshot explícito, vale o padrão da tarefa
                    shots = run.num_fewshot
                else:
                    shots = cfg.num_fewshot if run.num_fewshot is None else run.num_fewshot
                multiturn = _multiturn(cfg, run, model, evaluation, shots)
                steps.append(
                    Step(
                        model=model,
                        benchmark=run.name,
                        evaluation=evaluation,
                        num_fewshot=shots,
                        fewshot_strategy=run.fewshot_strategy or cfg.fewshot_strategy,
                        limit=_limit(run, cfg),
                        skip_reason=f"backend sem {', '.join(sorted(missing))}" if missing else None,
                        fewshot_as_multiturn=multiturn,
                    )
                )
    return steps


def _multiturn(
    cfg: ExperimentConfig, run: BenchmarkRun, model: ModelConfig, evaluation: str, shots: int | None
) -> bool:
    wanted = cfg.fewshot_as_multiturn if run.fewshot_as_multiturn is None else run.fewshot_as_multiturn
    if not (wanted and model.uses_chat and needs(run.name, evaluation) == {"generate"}):
        return False
    if is_lm_eval(run.name):
        return shots != 0
    if not getattr(EVALUATIONS.get(evaluation), "supports_multiturn", False):
        raise ValueError(f"{run.name}: a avaliação {evaluation} não suporta fewshot_as_multiturn")
    return bool(shots)


def _limit(run: BenchmarkRun, cfg: ExperimentConfig) -> int | None:
    if run.limit == "all":
        return None
    return run.limit if run.limit is not None else cfg.limit  # type: ignore[return-value]


def needs(benchmark: str, evaluation: str) -> frozenset[str]:
    return task_needs(benchmark) if is_lm_eval(benchmark) else EVALUATIONS.get(evaluation).needs


def _evaluations(run: BenchmarkRun) -> tuple[str, ...]:
    """Avaliações declaradas pelo benchmark; `evaluations` no YAML só restringe a lista."""
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
            if s.fewshot_as_multiturn:
                chat = " (chat, multi-turn)"
            status = f"{kind}{chat}"
        rows.append(
            (
                s.model.display_name,
                s.benchmark,
                s.evaluation,
                "-" if s.evaluation == "-" else "padrão" if s.num_fewshot is None else str(s.num_fewshot),
                status,
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)
