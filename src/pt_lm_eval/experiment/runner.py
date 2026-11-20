from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..benchmarks import BENCHMARKS
from ..core import (
    GenerateRequest,
    GenerateResponse,
    Item,
    LoglikelihoodRequest,
    Request,
    Response,
    Scored,
    non_default,
)
from ..evaluations import EVALUATIONS, Evaluation
from ..inference import Client, Endpoint, ServerError
from .config import ExperimentConfig, ModelConfig, Step
from .lengths import LengthCheck, TokenCounter
from .pool import GPUPool
from .report import Results, build_result, describe_requests, write_text_atomic

log = logging.getLogger("pt_lm_eval")
CHUNK = 512  # itens por lote; o progresso é salvo a cada lote


class SmokeTestError(RuntimeError):
    pass


class GenerationError(RuntimeError):
    pass


class Experiment:
    def __init__(self, cfg: ExperimentConfig, config_path: Path | None = None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.steps = cfg.plan()
        self.results = Results(cfg.out_dir)
        self.pool = GPUPool(cfg) if cfg.gpus is not None else None

    # --- execução -------------------------------------------------------------------

    def run(self) -> None:
        out = self.cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        config_copy = out / "experiment.yaml"
        if self.config_path and self.config_path.resolve() != config_copy.resolve():
            shutil.copy(self.config_path, config_copy)
        try:
            if self.pool is not None:
                failed = self.pool.run([model for model, _ in self._by_model()])
            else:
                failed = [
                    model.display_name
                    for model, steps in self._by_model()
                    if not self.execute_model(model, steps)
                ]
        finally:
            # Só o coordenador escreve o CSV, depois de todos os processos terminarem.
            log.info("relatório: %s", self.results.write_csv(self.cfg.experiment))
        if failed:
            raise RuntimeError(f"modelos pulados: {failed}; veja os logs acima")

    def execute_model(self, model: ModelConfig, steps: Sequence[Step]) -> bool:
        """Executa todos os passos de um modelo e só retorna depois de desligar o endpoint."""
        log.info("modelo %s: %d passos", model.display_name, len(steps))
        try:
            with Endpoint.for_model(model, self.cfg.server, self.cfg.out_dir / "logs") as base_url:
                args = self.cfg.server.args_for(model)
                client = Client.for_model(
                    model, base_url, self._max_length(model), reasoning_parser=args.get("reasoning_parser")
                )
                asyncio.run(self._run_model(steps, client))
        except (ServerError, SmokeTestError, GenerationError) as exc:
            log.error("modelo %s incompleto (%s): %s", model.display_name, type(exc).__name__, exc)
            return False
        return True

    def _by_model(self) -> list[tuple[ModelConfig, list[Step]]]:
        grouped: dict[str, list[Step]] = {}
        for step in self.steps:
            if not step.skip_reason:
                grouped.setdefault(step.model.display_name, []).append(step)
        return [(steps[0].model, steps) for steps in grouped.values()]

    def _max_length(self, model: ModelConfig) -> int | None:
        """Contexto para o log-likelihood: o servidor reserva 1 token para a geração mínima."""
        value = self.cfg.server.args_for(model).get("max_model_len")
        return int(value) - 1 if value else None

    async def _run_model(self, steps: Sequence[Step], client: Client) -> None:
        try:
            await smoke_test(client, {kind for s in steps for kind in s.needs})
            log.info("teste de sanidade ok")
            for step in steps:
                await self.run_step(step, client)
        finally:
            await client.aclose()

    async def run_step(self, step: Step, client: Any) -> dict:
        bench, evaluation = BENCHMARKS.get(step.benchmark), EVALUATIONS.get(step.evaluation)
        items, shots = self.select(step)
        prompt_sha = bench.prompt_signature(evaluation)
        fingerprint = self._fingerprint(step, bench.version, prompt_sha, evaluation)
        samples_path, results_path = self.results.paths(step)
        records = _resume(samples_path, fingerprint)
        todo, resumed = [it for it in items if it.id not in records], len(records)
        if todo:
            results_path.unlink(missing_ok=True)
        log.info("%s/%s: %d itens (%d retomados)", step.benchmark, step.evaluation, len(items), resumed)

        started = time.monotonic()
        with samples_path.open("a", encoding="utf-8") as fh:
            for start in range(0, len(todo), CHUNK):
                chunk = todo[start : start + CHUNK]
                requests = [step.build(it, shots[it.id]) for it in chunk]
                responses = iter(await _answer(client, [r for rs in requests for r in rs]))
                errors = []
                for item, reqs in zip(chunk, requests, strict=True):
                    answers = [next(responses) for _ in reqs]
                    if error := next(
                        (r.error for r in answers if isinstance(r, GenerateResponse) and r.error), None
                    ):
                        errors.append(f"item {item.id}: {error}")
                        continue
                    scored = evaluation.score(item, answers)
                    records[item.id] = _record(item, reqs, shots[item.id], scored)
                    fh.write(json.dumps(records[item.id], ensure_ascii=False) + "\n")
                fh.flush()
                if errors:
                    raise GenerationError(f"{step.benchmark}: {len(errors)} falhas de geração. {errors[0]}")
                log.info(
                    "%s/%s: %d/%d itens processados",
                    step.benchmark,
                    step.evaluation,
                    min(start + len(chunk), len(todo)),
                    len(todo),
                )
        elapsed = time.monotonic() - started
        if resumed:  # retomada: soma o tempo da execução anterior com a mesma configuração
            elapsed += _previous_seconds(results_path, fingerprint)

        result = build_result(
            cfg=self.cfg,
            step=step,
            bench=bench,
            n_items=len(bench.data.items),
            n_discarded=bench.data.discarded,
            prompt_sha=prompt_sha,
            records=[records[it.id] for it in items],
            samples_path=samples_path.name,
            fingerprint=fingerprint,
            timing={"seconds": round(elapsed, 1), "items_run": len(todo), "items_resumed": resumed},
        )
        self.results.write(result, results_path)
        return result

    def select(self, step: Step) -> tuple[list[Item], dict[str, list[Item]]]:
        """Itens do passo (com `limit`, sorteados com a seed) e os exemplos few-shot de cada um."""
        bench, seed = BENCHMARKS.get(step.benchmark), self.cfg.seed
        items = bench.data.items
        if step.exclude_ids:
            excluded = set(step.exclude_ids)
            missing = excluded - {item.id for item in items}
            if missing:
                raise ValueError(f"{step.benchmark}: exclude_ids desconhecidos: {sorted(missing)}")
            items = [item for item in items if item.id not in excluded]
        if not items:
            raise ValueError(f"{step.benchmark}: nenhum item para avaliar após exclusões")
        return bench.select_for_evaluation(
            items, step.num_fewshot, step.limit, seed, holdout=step.fewshot_holdout
        )

    def _fingerprint(self, step: Step, version: str, prompt_sha: str, evaluation: Evaluation) -> str:
        """Identidade das amostras: mudou qualquer coisa que afeta o número, elas são refeitas."""
        key: list[Any] = [non_default(step.model), step.benchmark, version, step.evaluation, prompt_sha]
        key += [step.num_fewshot, step.limit, self.cfg.seed]
        server = self.cfg.server
        if step.model.base_url:
            args = server.args_for(step.model)
            settings = {key: args.get(key) for key in ("max_model_len", "reasoning_parser")}
        else:
            settings = {k: v for k, v in non_default(server).items() if k not in {"port", "startup_timeout"}}
        if settings := {k: v for k, v in settings.items() if v is not None}:
            key.append({"server": settings})
        if step.fewshot_as_multiturn:
            key.append("multiturn")
        if step.exclude_ids:
            key.append({"exclude_ids": step.exclude_ids})
        if step.fewshot_holdout:
            key.append({"fewshot_holdout": True})
        if evaluation.scoring_version:
            key.append(f"scoring:{evaluation.scoring_version}")
        return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # --- plano e checagem de contexto (--dry-run) ---------------------------------

    def describe(self) -> str:
        rows = [("modelo", "benchmark", "avaliação", "shots", "situação")]
        for s in self.steps:
            if s.skip_reason:
                rows.append(
                    (s.model.display_name, s.benchmark, s.evaluation, "-", f"pulado: {s.skip_reason}")
                )
                continue
            kind = "/".join(sorted(s.needs))
            if s.fewshot_as_multiturn:
                kind += " (chat, multi-turn)"
            elif s.model.uses_chat and kind == "generate":
                kind += " (chat)"
            rows.append((s.model.display_name, s.benchmark, s.evaluation, str(s.num_fewshot), kind))
        return table(rows)

    def check_lengths(self) -> list[LengthCheck]:
        return [self._check_length(s) for s in self.steps if not s.skip_reason]

    def _check_length(self, step: Step) -> LengthCheck:
        model = step.model
        if model.type == "api":
            return LengthCheck(step, note="API: sem tokenizer local")
        limit = self.cfg.server.args_for(model).get("max_model_len")
        if not limit:
            return LengthCheck(step, note="max_model_len não definido no YAML")
        items, shots = self.select(step)
        sizes = TokenCounter(model).sizes([step.build(it, shots[it.id]) for it in items])
        generate = "generate" in step.needs
        budget = int(limit) if generate else int(limit) - 1  # log-likelihood: 1 token fica para a geração
        over = sum(size > budget for size in sizes)
        return LengthCheck(step, None, generate, len(sizes), max(sizes, default=0), int(limit), over)

    def dry_run(self) -> bool:
        """Mostra o plano e a checagem de contexto; devolve False se algo não cabe."""
        if self.pool is not None:
            print(self.pool.describe(), flush=True)
        print(self.describe(), flush=True)
        checks = self.check_lengths()
        header = ("modelo", "benchmark", "avaliação", "itens", "maior", "limite", "situação")
        print(
            "\nContexto (prompt + geração reservada ≤ max_model_len):\n"
            + table([header, *(c.row() for c in checks)])
        )
        ok = all(c.ok for c in checks)
        if not ok:
            print("\nAjuste max_model_len, num_fewshot ou max_tokens no YAML antes de rodar.")
        return ok


# --- auxiliares -------------------------------------------------------------------


def table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)


async def smoke_test(client: Any, needs: set[str]) -> None:
    """Um pedido de cada tipo que o modelo vai usar; levanta SmokeTestError se algum falhar."""
    problems = []
    if "loglikelihood" in needs:
        try:
            (r,) = await client.loglikelihood([LoglikelihoodRequest("A capital do Brasil é", " Brasília")])
            if not (math.isfinite(r.logprob) and r.logprob < 0 and r.n_tokens > 0):
                problems.append(f"log-likelihood inválido: {r}")
        except Exception as exc:
            problems.append(f"log-likelihood falhou: {type(exc).__name__}: {exc}")
    if "generate" in needs:
        try:
            (g,) = await client.generate([GenerateRequest(prompt="Responda apenas com a palavra: ok")])
            if g.error:
                problems.append(f"geração falhou: {g.error}")
            elif not (g.text or g.reasoning):
                problems.append(f"geração vazia (finish_reason={g.finish_reason})")
        except Exception as exc:
            problems.append(f"geração falhou: {type(exc).__name__}: {exc}")
    if problems:
        raise SmokeTestError("; ".join(problems))


async def _answer(client: Any, requests: list[Request]) -> list[Response]:
    """Envia cada tipo de pedido ao método certo e devolve as respostas na ordem original."""
    answers: list[Response | None] = [None] * len(requests)
    for kind in ("loglikelihood", "generate"):
        if idx := [i for i, r in enumerate(requests) if r.kind == kind]:
            method = getattr(client, kind)
            for i, response in zip(idx, await method([requests[i] for i in idx]), strict=True):
                answers[i] = response
    return answers  # type: ignore[return-value]


def _record(item: Item, requests: list[Request], shots: list[Item], scored: Scored) -> dict:
    record = {
        "id": item.id,
        "subject": getattr(item, "subject", None),
        "metrics": scored.metrics,
        "details": scored.details,
        "fewshot_ids": [s.id for s in shots],
        "requests": describe_requests(requests),
    }
    if task_group := getattr(item, "task_group", None):
        record["task_group"] = task_group
    return record | ({"weights": scored.weights} if scored.weights else {})


def _resume(path: Path, fingerprint: str) -> dict[str, dict]:
    """Amostras já calculadas com a mesma configuração; senão, recomeça o arquivo."""
    header = {"_fingerprint": fingerprint}
    if path.exists():
        with path.open(encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
            try:
                previous = json.loads(first_line)
            except json.JSONDecodeError:
                previous = None
            if previous == header:
                records = {}
                clean = first_line.endswith("\n")
                for line in fh:
                    try:
                        record = json.loads(line)
                        if (record.get("details") or {}).get("error"):
                            clean = False
                        else:
                            records[record["id"]] = record
                    except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
                        clean = False
                        continue
                    clean &= line.endswith("\n")
                if not clean:
                    lines = [json.dumps(r, ensure_ascii=False) for r in [header, *records.values()]]
                    write_text_atomic(path, "\n".join(lines) + "\n")
                return records
        log.info("%s: configuração mudou ou cabeçalho incompleto, recomeçando", path.name)
    write_text_atomic(path, json.dumps(header) + "\n")
    return {}


def _previous_seconds(path: Path, fingerprint: str) -> float:
    """Tempo registrado pela execução anterior do passo, se ela teve a mesma configuração."""
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        return 0.0
    if previous.get("evaluation", {}).get("fingerprint") != fingerprint:
        return 0.0
    return float(previous.get("timing", {}).get("seconds") or 0.0)
