"""Entrada interna do subprocesso de uma GPU; nunca escreve o CSV compartilhado."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path

from .config import parse_config
from .pool import interrupt

log = logging.getLogger("pt_lm_eval")


def main(path: str) -> int:
    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        payload = json.loads(Path(path).read_text())
        gpu = payload["gpu"]
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu or os.environ.get("PT_LM_EVAL_GPU") != gpu:
            raise ValueError("worker deve ser iniciado pelo coordenador com a GPU isolada")
        cfg = parse_config(payload["config"])
        if cfg.gpus is not None or len(cfg.models) != 1:
            raise ValueError("worker exige exatamente um modelo e gpus=null")
        from .runner import Experiment

        experiment = Experiment(cfg)
        log.info("GPU %s / porta %d: %s", gpu, cfg.server.port, cfg.models[0].display_name)
        steps = [step for step in experiment.steps if not step.skip_reason]
        return 0 if experiment.execute_model(cfg.models[0], steps) else 1
    except KeyboardInterrupt:
        log.warning("avaliação cancelada; endpoint encerrado")
        return 130
    except Exception:
        log.exception("avaliação falhou")
        return 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda r: r.name.startswith("pt_lm_eval") or r.levelno >= logging.WARNING)
    sys.exit(main(sys.argv[1]))
