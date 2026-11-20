from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import BinaryIO

from ..inference.server import slug
from .config import ExperimentConfig, ModelConfig

log = logging.getLogger("pt_lm_eval")


def interrupt(_signum: int, _frame: object) -> None:
    # Um segundo sinal não deve interromper a liberação dos servidores.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise KeyboardInterrupt


@dataclass
class Running:
    model: ModelConfig
    process: subprocess.Popen
    log_file: BinaryIO


class GPUPool:
    def __init__(self, cfg: ExperimentConfig) -> None:
        assert cfg.gpus is not None
        self.cfg = cfg
        self.slots = [(gpu, cfg.server.port + i) for i, gpu in enumerate(cfg.gpus)]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            allowed = {value.strip() for value in visible.split(",")}
            if not set(cfg.gpus) <= allowed:
                raise ValueError("gpus deve usar os mesmos IDs permitidos por CUDA_VISIBLE_DEVICES")

    def describe(self) -> str:
        return "Fila de modelos: " + ", ".join(f"GPU {gpu} / porta {port}" for gpu, port in self.slots)

    def _start(self, model: ModelConfig, gpu: str, port: int) -> Running:
        cfg = replace(self.cfg, models=(model,), gpus=None, server=replace(self.cfg.server, port=port))
        logs = cfg.out_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stem = slug(model.display_name)
        path = logs / f"{stem}.worker.json"
        path.write_text(json.dumps({"gpu": gpu, "config": asdict(cfg)}, ensure_ascii=False, indent=2) + "\n")
        log_path = logs / f"{stem}.eval.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PT_LM_EVAL_GPU": gpu, "PYTHONUNBUFFERED": "1"}
        fh = log_path.open("ab")
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "pt_lm_eval.experiment.worker", str(path.resolve())],
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except BaseException:
            fh.close()
            raise
        log.info("GPU %s / porta %d: iniciando %s (log: %s)", gpu, port, model.display_name, log_path)
        return Running(model, process, fh)

    def run(self, models: list[ModelConfig]) -> list[str]:
        pending = deque(models)
        active: dict[tuple[str, int], Running] = {}
        failed = []
        handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        log.info(self.describe())
        try:
            while pending or active:
                for slot in self.slots:
                    job = active.get(slot)
                    if job is not None:
                        code = job.process.poll()
                        if code is None:
                            continue
                        job.log_file.close()
                        del active[slot]
                        if code:
                            failed.append(job.model.display_name)
                            log.error(
                                "GPU %s: %s falhou (código %s); veja o .eval.log",
                                slot[0],
                                job.model.display_name,
                                code,
                            )
                        else:
                            log.info("GPU %s: %s concluído", slot[0], job.model.display_name)
                    if pending:
                        model = pending.popleft()
                        try:
                            active[slot] = self._start(model, *slot)
                        except Exception:
                            failed.append(model.display_name)
                            log.exception("GPU %s: não foi possível iniciar %s", slot[0], model.display_name)
                if active:
                    time.sleep(0.2)
        finally:
            # O worker trata SIGTERM e sai pelo contexto do Endpoint, desligando o vLLM.
            # Esperar sua saída é obrigatório antes de reutilizar uma GPU ou escrever o CSV.
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                for job in active.values():
                    if job.process.poll() is None:
                        job.process.terminate()
                for job in active.values():
                    try:
                        job.process.wait()
                    finally:
                        job.log_file.close()
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
        return failed
