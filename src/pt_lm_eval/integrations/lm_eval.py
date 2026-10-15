"""Ponte com o lm-evaluation-harness: tarefas `lm_eval:<nome>` rodam pelo nosso backend.

O lm-eval cuida de dataset, prompt, filtros e métricas da tarefa; o pt-lm-eval
cuida do servidor, das requisições e do relatório. `PtLM` expõe nosso backend
como um `lm_eval.api.model.LM`, e os resultados viram o mesmo formato dos
benchmarks nativos. Requer o extra: `uv pip install -e ".[lm-eval]"`.

As métricas e o stderr são os calculados pelo próprio lm-eval (algumas tarefas
não agregam por média, então não os recalculamos).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.types import GenerateRequest, LoglikelihoodRequest
from ..report.results import SCHEMA, package_version, write_result

if TYPE_CHECKING:
    from ..core.protocols import Backend
    from ..runner.config import ExperimentConfig
    from ..runner.plan import Step

PREFIX = "lm_eval:"
_KIND = {
    "multiple_choice": "loglikelihood",
    "loglikelihood": "loglikelihood",
    "loglikelihood_rolling": "loglikelihood_rolling",  # nenhum backend oferece: fica pulado
    "generate_until": "generate",
}


def is_lm_eval(name: str) -> bool:
    return name.startswith(PREFIX)


@cache
def task_needs(name: str) -> frozenset[str]:
    """Tipos de pedido usados pela tarefa (ou grupo de tarefas) do lm-eval."""
    try:
        from lm_eval.tasks import TaskManager, get_task_dict
    except ImportError as exc:
        raise RuntimeError(f"{name} requer o extra do lm-eval: uv pip install -e '.[lm-eval]'") from exc

    tasks = _leaves(get_task_dict([name.removeprefix(PREFIX)], TaskManager()))
    return frozenset(_KIND[t.OUTPUT_TYPE] for t in tasks)


def _leaves(task_dict: dict) -> list[Any]:
    out = []
    for value in task_dict.values():
        out.extend(_leaves(value) if isinstance(value, dict) else [value])
    return out


def _make_lm(backend: Backend, loop: asyncio.AbstractEventLoop, chat: bool, model: str) -> Any:
    from lm_eval.api.model import LM

    class PtLM(LM):
        def _run(self, coro: Any) -> Any:  # o lm-eval roda numa thread; o backend, no loop principal
            return asyncio.run_coroutine_threadsafe(coro, loop).result()

        def loglikelihood(self, requests: list[Any], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
            reqs = [LoglikelihoodRequest(*r.args) for r in requests]
            return [(r.logprob, r.is_greedy) for r in self._run(backend.loglikelihood(reqs))]

        def loglikelihood_rolling(self, requests: list[Any], disable_tqdm: bool = False) -> list[float]:
            raise NotImplementedError("loglikelihood_rolling (perplexidade) ainda não é suportado")

        def generate_until(self, requests: list[Any], disable_tqdm: bool = False) -> list[str]:
            reqs = []
            for r in requests:
                context, kwargs = r.args
                messages = _as_messages(context) if chat else None
                reqs.append(
                    GenerateRequest(
                        prompt=context if messages is None else _flatten(messages),
                        max_tokens=kwargs.get("max_gen_toks"),
                        stop=tuple(kwargs.get("until") or ()),
                        messages=messages,
                    )
                )
            return [r.text for r in self._run(backend.generate(reqs))]

        def apply_chat_template(
            self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
        ) -> str:
            return json.dumps(chat_history, ensure_ascii=False)  # o backend aplica o template real

        @property
        def tokenizer_name(self) -> str:
            return model

        def chat_template(self, chat_template: bool | str = False) -> str:
            return ""

    return PtLM()


def _as_messages(context: str) -> tuple[dict[str, str], ...] | None:
    try:
        value = json.loads(context)
    except (json.JSONDecodeError, TypeError):
        return None
    return tuple(value) if isinstance(value, list) else None


def _flatten(messages: tuple[dict[str, str], ...]) -> str:
    return "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)


async def run_lm_eval_step(step: Step, backend: Backend, cfg: ExperimentConfig, stem: Path) -> dict:
    from lm_eval import __version__ as lm_eval_version
    from lm_eval.evaluator import simple_evaluate

    task = step.benchmark.removeprefix(PREFIX)
    # Chat template só em tarefas de geração: log-likelihood usa o endpoint de completions.
    chat = step.model.uses_chat and task_needs(step.benchmark) == {"generate"}
    lm = _make_lm(backend, asyncio.get_running_loop(), chat, step.model.name)
    out = await asyncio.to_thread(
        simple_evaluate,
        model=lm,
        tasks=[task],
        num_fewshot=step.num_fewshot,
        limit=step.limit,
        apply_chat_template=chat,
        log_samples=True,
        random_seed=cfg.seed,
        numpy_random_seed=cfg.seed,
        fewshot_random_seed=cfg.seed,
    )
    samples_path = Path(f"{stem}.samples.jsonl")
    with samples_path.open("w") as fh:
        for subtask, samples in out["samples"].items():
            for s in samples:
                record = {
                    "id": f"{subtask}-{s['doc_id']}",
                    "subject": subtask,
                    "target": s.get("target"),
                    "arguments": s.get("arguments"),
                    "filtered_resps": s.get("filtered_resps"),
                }
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    leaves = [t for t in out["results"] if t != task]
    result = {
        "schema": SCHEMA,
        "pt_lm_eval": package_version(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment": cfg.experiment,
        "benchmark": {
            "name": step.benchmark,
            "version": f"{out['versions'].get(task, '?')}",
            "source": f"lm-eval {lm_eval_version}",
            "n_evaluated": sum(n["effective"] for n in out["n-samples"].values()),
        },
        "evaluation": {
            "name": "lm_eval",
            "num_fewshot": step.num_fewshot,
            "chat_template": chat,
            "seed": cfg.seed,
            "limit": step.limit,
        },
        "model": {k: v for k, v in asdict(step.model).items() if v not in (None, {}, "")},
        "metrics": _metrics(out["results"].get(task) or out["results"][leaves[0]]),
        "by_subject": {t: _metrics(out["results"][t]) for t in leaves} if leaves else {},
        "samples": samples_path.name,
    }
    write_result(result, Path(f"{stem}.results.json"))
    return result


def _metrics(raw: dict[str, Any]) -> dict[str, float]:
    """'acc,none' → 'acc'; 'exact_match,strict-match' → 'exact_match[strict-match]'."""
    out = {}
    for key, value in raw.items():
        if "," not in key or not isinstance(value, int | float):
            continue
        metric, flt = key.split(",", 1)
        stderr = metric.endswith("_stderr")
        base = metric.removesuffix("_stderr")
        name = base if flt == "none" else f"{base}[{flt}]"
        out[f"{name}_stderr" if stderr else name] = float(value)
    return out
