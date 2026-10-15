"""Execução do plano: para cada modelo, sobe o endpoint e roda os passos em ordem."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import shutil
from collections.abc import Sequence
from dataclasses import asdict
from itertools import groupby
from pathlib import Path

from ..core.protocols import BACKENDS, BENCHMARKS, EVALUATIONS, REPORTERS, Backend
from ..core.types import Item, Request, Response, Scored
from ..data.fewshot import select_shots
from ..data.loading import load_items, prompt_signature
from ..integrations.lm_eval import is_lm_eval, run_lm_eval_step
from ..report.results import build_result, describe_requests, write_result
from ..servers import max_model_len, model_endpoint, slug
from .config import ExperimentConfig
from .plan import Step, build_plan, format_plan

log = logging.getLogger("pt_lm_eval")
CHUNK = 512  # itens por lote; o progresso é salvo a cada lote


def run_experiment(cfg: ExperimentConfig, config_path: Path | None = None, dry_run: bool = False) -> None:
    steps = build_plan(cfg)
    print(format_plan(steps), flush=True)
    if dry_run:
        return
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copy(config_path, out / "experiment.yaml")
    runnable = [s for s in steps if not s.skip_reason]
    for _, group in groupby(runnable, key=lambda s: id(s.model)):
        model_steps = list(group)
        model = model_steps[0].model
        log.info("modelo %s: %d passos", model.display_name, len(model_steps))
        with model_endpoint(model, cfg.server, out / "logs") as base_url:
            backend = BACKENDS.get(model.backend).create(model, base_url, max_model_len(model, cfg.server))
            asyncio.run(_run_model(model_steps, backend, cfg))
    for name in cfg.reporters:
        path = REPORTERS.get(name).write(collect_results(out), out, cfg.experiment)
        log.info("relatório %s: %s", name, path)


async def _run_model(steps: Sequence[Step], backend: Backend, cfg: ExperimentConfig) -> None:
    try:
        for step in steps:
            await run_step(step, backend, cfg)
    finally:
        await backend.aclose()


async def run_step(step: Step, backend: Backend, cfg: ExperimentConfig) -> dict:
    stem = cfg.out_dir / slug(step.model.display_name) / f"{slug(step.benchmark)}.{step.evaluation}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    if is_lm_eval(step.benchmark):
        log.info("%s: tarefa do lm-eval", step.benchmark)
        return await run_lm_eval_step(step, backend, cfg, stem)
    bench = BENCHMARKS.get(step.benchmark)
    evaluation = EVALUATIONS.get(step.evaluation)
    loaded = load_items(step.benchmark)
    items, pool = loaded.items, loaded.pool
    selected = _limit(items, step.limit, cfg.seed)
    exclude_self = bench.fewshot_source is None
    shots = {
        it.id: select_shots(
            it,
            pool,
            step.num_fewshot,
            seed=cfg.seed,
            strategy=step.fewshot_strategy,
            exclude_self=exclude_self,
        )
        for it in selected
    }
    prompt_sha = prompt_signature(evaluation, items, pool)
    samples_path = Path(f"{stem}.samples.jsonl")
    fingerprint = _fingerprint(step, bench.version, prompt_sha, cfg.seed)
    records = _resume(samples_path, fingerprint)
    todo = [it for it in selected if it.id not in records]
    log.info("%s/%s: %d itens (%d retomados)", step.benchmark, step.evaluation, len(selected), len(records))

    with samples_path.open("a") as fh:
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start : start + CHUNK]
            requests = [evaluation.build(it, shots[it.id]) for it in chunk]
            responses = await _answer(backend, [r for rs in requests for r in rs])
            offset = 0
            for item, reqs in zip(chunk, requests, strict=True):
                scored = evaluation.score(item, responses[offset : offset + len(reqs)])
                offset += len(reqs)
                record = _record(item, reqs, shots[item.id], scored)
                records[item.id] = record
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    result = build_result(
        cfg=cfg,
        step=step,
        bench=bench,
        n_items=len(items),
        n_discarded=loaded.discarded,
        prompt_sha=prompt_sha,
        records=[records[it.id] for it in selected],
        samples_path=samples_path.name,
    )
    write_result(result, Path(f"{stem}.results.json"))
    return result


async def _answer(backend: Backend, requests: list[Request]) -> list[Response]:
    """Envia cada tipo de pedido ao método certo e devolve as respostas na ordem original."""
    answers: list[Response | None] = [None] * len(requests)
    for kind, method in (("loglikelihood", backend.loglikelihood), ("generate", backend.generate)):
        idx = [i for i, r in enumerate(requests) if r.kind == kind]
        if idx:
            for i, resp in zip(idx, await method([requests[i] for i in idx]), strict=True):
                answers[i] = resp
    return answers  # type: ignore[return-value]


def collect_results(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out.glob("*/*.results.json"))]


def _limit(items: list[Item], limit: int | None, seed: int) -> list[Item]:
    if limit is None or limit >= len(items):
        return items
    keep = set(random.Random(seed).sample(range(len(items)), limit))
    return [it for i, it in enumerate(items) if i in keep]


def _record(item: Item, requests: list[Request], shots: list[Item], scored: Scored) -> dict:
    return {
        "id": item.id,
        "subject": getattr(item, "subject", None),
        "metrics": scored.metrics,
        "details": scored.details,
        "fewshot_ids": [s.id for s in shots],
        "requests": describe_requests(requests),
    }


def _fingerprint(step: Step, version: str, prompt_sha: str, seed: int) -> str:
    key = [
        asdict(step.model),
        step.benchmark,
        version,
        step.evaluation,
        prompt_sha,
        step.num_fewshot,
        step.fewshot_strategy,
        step.limit,
        seed,
    ]
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _resume(path: Path, fingerprint: str) -> dict[str, dict]:
    """Amostras já calculadas com a mesma configuração; senão, recomeça o arquivo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lines = path.read_text().splitlines()
        if lines and json.loads(lines[0]).get("_fingerprint") == fingerprint:
            records = {}
            for line in lines[1:]:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:  # última linha cortada por uma interrupção
                    continue
                records[record["id"]] = record
            # Reescreve sem linhas cortadas antes de continuar anexando.
            path.write_text(
                "\n".join([lines[0], *(json.dumps(r, ensure_ascii=False) for r in records.values())]) + "\n"
            )
            return records
        log.info("%s: configuração mudou, recomeçando", path.name)
    path.write_text(json.dumps({"_fingerprint": fingerprint}) + "\n")
    return {}
