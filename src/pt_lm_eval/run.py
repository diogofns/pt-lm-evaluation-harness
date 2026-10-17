"""Execução de um experimento: um modelo por vez, na ordem do YAML, com todos os seus benchmarks.

Para cada modelo: sobe o endpoint, roda um teste de sanidade (um pedido de cada tipo que
ele vai usar) e os passos do plano, salvando cada amostra assim que pontuada; uma
execução interrompida retoma de onde parou se a configuração for a mesma. Servidor que
não sobe ou teste que falha pula o modelo, e os demais seguem.

O `--dry-run` mostra o plano e mede, com o tokenizer de cada modelo, se os prompts cabem
no `max_model_len` (prompt + `max_tokens` na geração); nada é cortado ou alterado.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from itertools import groupby
from pathlib import Path
from typing import Any

from .client import DEFAULT_MAX_TOKENS, Client, PairTokenizer
from .config import ExperimentConfig, ModelConfig, Step
from .core import (
    BENCHMARKS,
    EVALUATIONS,
    GenerateRequest,
    Item,
    LoglikelihoodRequest,
    Request,
    Response,
    Scored,
)
from .data import load_items, non_default, prompt_signature, select_shots
from .report import build_result, collect_results, describe_requests, write_csv, write_result
from .server import ServerError, model_endpoint, slug

log = logging.getLogger("pt_lm_eval")
CHUNK = 512  # itens por lote; o progresso é salvo a cada lote


class SmokeTestError(RuntimeError):
    pass


@dataclass(frozen=True)
class LengthCheck:
    """Checagem de contexto de um passo: maior prompt + geração reservada, em tokens."""

    step: Step
    note: str | None = None  # por que não foi verificado
    generate: bool = False
    items: int = 0
    longest: int = 0
    limit: int = 0
    over: int = 0  # itens que não cabem

    @property
    def ok(self) -> bool:
        return self.note is not None or self.over == 0

    def row(self) -> tuple[str, ...]:
        s = self.step
        if self.note:
            return (s.model.display_name, s.benchmark, s.evaluation, "", "", "", self.note)
        status = (
            "ok"
            if not self.over
            else f"{self.over} itens "
            + ("não cabem (o vLLM recusaria)" if self.generate else "seriam cortados pela esquerda")
        )
        return (
            s.model.display_name,
            s.benchmark,
            s.evaluation,
            str(self.items),
            str(self.longest),
            str(self.limit),
            status,
        )


class Experiment:
    def __init__(self, cfg: ExperimentConfig, config_path: Path | None = None) -> None:
        self.cfg = cfg
        self.config_path = config_path
        self.steps = cfg.plan()

    # --- execução -------------------------------------------------------------------

    def run(self) -> None:
        out = self.cfg.out_dir
        out.mkdir(parents=True, exist_ok=True)
        if self.config_path:
            shutil.copy(self.config_path, out / "experiment.yaml")
        failed = []
        runnable = [s for s in self.steps if not s.skip_reason]
        for _, group in groupby(runnable, key=lambda s: id(s.model)):
            steps = list(group)
            model = steps[0].model
            log.info("modelo %s: %d passos", model.display_name, len(steps))
            try:
                with model_endpoint(model, self.cfg.server, out / "logs") as base_url:
                    asyncio.run(
                        self._run_model(steps, Client.for_model(model, base_url, self._max_length(model)))
                    )
            except (ServerError, SmokeTestError) as exc:
                reason = (
                    "teste de sanidade falhou" if isinstance(exc, SmokeTestError) else "servidor não subiu"
                )
                log.error("modelo %s pulado (%s): %s", model.display_name, reason, exc)
                failed.append(model.display_name)
        path = write_csv(collect_results(out), out / f"{self.cfg.experiment}.csv")
        log.info("relatório: %s", path)
        if failed:
            raise RuntimeError(f"modelos pulados: {failed}; veja os logs acima")

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
        stem = self.cfg.out_dir / slug(step.model.display_name) / f"{slug(step.benchmark)}.{step.evaluation}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        bench, evaluation, loaded = (
            BENCHMARKS.get(step.benchmark),
            EVALUATIONS.get(step.evaluation),
            load_items(step.benchmark),
        )
        selected, shots = self.select(step)
        prompt_sha = prompt_signature(evaluation, loaded.items, loaded.pool)
        fingerprint = self._fingerprint(step, bench.version, prompt_sha, evaluation)
        samples_path, results_path = Path(f"{stem}.samples.jsonl"), Path(f"{stem}.results.json")
        records = _resume(samples_path, fingerprint)
        todo, resumed = [it for it in selected if it.id not in records], len(records)
        log.info("%s/%s: %d itens (%d retomados)", step.benchmark, step.evaluation, len(selected), resumed)

        started = time.monotonic()
        with samples_path.open("a") as fh:
            for start in range(0, len(todo), CHUNK):
                chunk = todo[start : start + CHUNK]
                requests = [step.build(it, shots[it.id]) for it in chunk]
                responses = iter(await _answer(client, [r for rs in requests for r in rs]))
                for item, reqs in zip(chunk, requests, strict=True):
                    scored = evaluation.score(item, [next(responses) for _ in reqs])
                    records[item.id] = _record(item, reqs, shots[item.id], scored)
                    fh.write(json.dumps(records[item.id], ensure_ascii=False) + "\n")
                fh.flush()
        elapsed = time.monotonic() - started
        if resumed:  # retomada: soma o tempo da execução anterior com a mesma configuração
            elapsed += _previous_seconds(results_path, fingerprint)

        result = build_result(
            cfg=self.cfg,
            step=step,
            bench=bench,
            n_items=len(loaded.items),
            n_discarded=loaded.discarded,
            prompt_sha=prompt_sha,
            records=[records[it.id] for it in selected],
            samples_path=samples_path.name,
            fingerprint=fingerprint,
            timing={"seconds": round(elapsed, 1), "items_run": len(todo), "items_resumed": resumed},
        )
        write_result(result, results_path)
        return result

    def select(self, step: Step) -> tuple[list[Item], dict[str, list[Item]]]:
        """Itens do passo (com `limit`, sorteados com a seed) e os exemplos few-shot de cada um."""
        loaded, seed = load_items(step.benchmark), self.cfg.seed
        items = loaded.items
        if step.limit is not None and step.limit < len(items):
            keep = set(random.Random(seed).sample(range(len(items)), step.limit))
            items = [it for i, it in enumerate(items) if i in keep]
        exclude_self = BENCHMARKS.get(step.benchmark).fewshot_source is None
        shots = {
            it.id: select_shots(it, loaded.pool, step.num_fewshot, seed=seed, exclude_self=exclude_self)
            for it in items
        }
        return items, shots

    def _fingerprint(self, step: Step, version: str, prompt_sha: str, evaluation: Any) -> str:
        """Identidade das amostras: mudou qualquer coisa que afeta o número, elas são refeitas."""
        key: list[Any] = [non_default(step.model), step.benchmark, version, step.evaluation, prompt_sha]
        key += [step.num_fewshot, step.limit, self.cfg.seed]
        if step.fewshot_as_multiturn:
            key.append("multiturn")
        if scoring := getattr(evaluation, "scoring_version", None):
            key.append(f"scoring:{scoring}")
        return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]

    # --- plano e checagem de contexto (--dry-run) ---------------------------------

    def describe(self) -> str:
        rows = [("modelo", "benchmark", "avaliação", "shots", "situação")]
        for s in self.steps:
            if s.skip_reason:
                status = f"pulado: {s.skip_reason}"
            else:
                kind = "/".join(sorted(s.needs))
                chat = (
                    " (chat, multi-turn)"
                    if s.fewshot_as_multiturn
                    else " (chat)"
                    if s.model.uses_chat and kind == "generate"
                    else ""
                )
                status = kind + chat
            shots = "-" if s.skip_reason else str(s.num_fewshot)
            rows.append((s.model.display_name, s.benchmark, s.evaluation, shots, status))
        return table(rows)

    def check_lengths(self) -> list[LengthCheck]:
        return [self._check_length(s) for s in self.steps if not s.skip_reason]

    def _check_length(self, step: Step) -> LengthCheck:
        model = step.model
        limit = self.cfg.server.args_for(model).get("max_model_len")
        if model.type == "api":
            return LengthCheck(step, note="API: sem tokenizer local")
        if not limit:
            return LengthCheck(step, note="max_model_len não definido no YAML")
        counter = _TokenCounter(model.tokenizer or model.name, model.add_bos)
        items, shots = self.select(step)
        parts = [[_parts(r, model, counter) for r in step.build(it, shots[it.id])] for it in items]
        counter.count_all(
            [t for item in parts for _, texts in item for t in texts]
        )  # em lote, bem mais rápido
        sizes = [
            max(fixed + sum(counter.count(*t) for t in texts) for fixed, texts in item) for item in parts
        ]
        generate = "generate" in step.needs
        budget = int(limit) if generate else int(limit) - 1  # log-likelihood: 1 token fica para a geração
        over = sum(size > budget for size in sizes)
        return LengthCheck(step, None, generate, len(sizes), max(sizes, default=0), int(limit), over)

    def dry_run(self) -> bool:
        """Mostra o plano e a checagem de contexto; devolve False se algo não cabe."""
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
        (g,) = await client.generate([GenerateRequest(prompt="Responda apenas com a palavra: ok")])
        if g.error:
            problems.append(f"geração falhou: {g.error}")
        elif not (g.text or g.reasoning):
            problems.append(f"geração vazia (finish_reason={g.finish_reason})")
    if problems:
        raise SmokeTestError("; ".join(problems))


async def _answer(client: Any, requests: list[Request]) -> list[Response]:
    """Envia cada tipo de pedido ao método certo e devolve as respostas na ordem original."""
    answers: list[Response | None] = [None] * len(requests)
    for kind, method in (("loglikelihood", client.loglikelihood), ("generate", client.generate)):
        if idx := [i for i, r in enumerate(requests) if r.kind == kind]:
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
    return record | ({"weights": scored.weights} if scored.weights else {})


def _resume(path: Path, fingerprint: str) -> dict[str, dict]:
    """Amostras já calculadas com a mesma configuração; senão, recomeça o arquivo."""
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


def _previous_seconds(path: Path, fingerprint: str) -> float:
    """Tempo registrado pela execução anterior do passo, se ela teve a mesma configuração."""
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        return 0.0
    if previous.get("evaluation", {}).get("fingerprint") != fingerprint:
        return 0.0
    return float(previous.get("timing", {}).get("seconds") or 0.0)


def _parts(
    request: Request, model: ModelConfig, counter: _TokenCounter
) -> tuple[int, list[tuple[str, bool]]]:
    """Tokens fixos do pedido e os textos a contar, como (texto, com tokens especiais)."""
    if isinstance(request, LoglikelihoodRequest):
        # Contexto e continuação contados à parte: pode diferir em ±1 token da junção real.
        return counter.bos, [(request.context, False), (request.continuation, False)]
    reserved = max(
        [n for n in (request.max_tokens, model.generate.get("max_tokens")) if n], default=DEFAULT_MAX_TOKENS
    )
    if not model.uses_chat:
        return reserved, [(request.prompt, True)]
    messages = list(request.messages or [{"role": "user", "content": request.prompt}])
    if model.system_prompt and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": model.system_prompt})
    kwargs = dict(model.generate.get("chat_template_kwargs") or {})
    if model.thinking is not None:
        kwargs["enable_thinking"] = model.thinking
    text = counter.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kwargs
    )
    return reserved, [(text, False)]


@cache
def _tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


class _TokenCounter:
    """Contagem de tokens em lote, com cache compartilhado entre modelos do mesmo tokenizer."""

    _caches: dict[str, dict[tuple[str, bool], int]] = {}  # noqa: RUF012

    def __init__(self, name: str, add_bos: bool | None) -> None:
        self.tokenizer = _tokenizer(name)
        self.bos = len(PairTokenizer(self.tokenizer, add_bos).bos)
        self.cache = self._caches.setdefault(name, {})

    def count_all(self, texts: list[tuple[str, bool]], batch: int = 1024) -> None:
        todo = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for special in (False, True):
            group = [text for text, s in todo if s is special]
            for start in range(0, len(group), batch):
                chunk = group[start : start + batch]
                ids = self.tokenizer(chunk, add_special_tokens=special)["input_ids"]
                self.cache.update({(text, special): len(i) for text, i in zip(chunk, ids, strict=True)})

    def count(self, text: str, special: bool) -> int:
        if (text, special) not in self.cache:
            self.count_all([(text, special)])
        return self.cache[(text, special)]
