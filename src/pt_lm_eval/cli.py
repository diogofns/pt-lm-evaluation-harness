"""Linha de comando: `pt-lm-eval run` e `pt-lm-eval benchmarks list|preview|check`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import catalog
from .core.protocols import BENCHMARKS
from .runner.config import load_config
from .runner.execute import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pt-lm-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="executa um experimento")
    run.add_argument("config", type=Path)
    run.add_argument("--dry-run", action="store_true", help="só mostra o plano")

    bench = sub.add_parser("benchmarks", help="lista, mostra e valida benchmarks")
    bsub = bench.add_subparsers(dest="action", required=True)
    bsub.add_parser("list", help="benchmarks disponíveis, oficiais e externos")
    preview = bsub.add_parser("preview", help="mostra os prompts sem chamar modelo")
    preview.add_argument("name")
    preview.add_argument("-e", "--evaluation")
    preview.add_argument("-n", type=int, default=2, help="quantos itens mostrar")
    preview.add_argument("--num-fewshot", type=int, default=0)
    check = bsub.add_parser("check", help="valida dados, prompts, cartão e versão")
    check.add_argument("names", nargs="*")
    check.add_argument("--all", action="store_true")
    check.add_argument("--lock", type=Path, default=Path("benchmarks.lock.json"))
    check.add_argument("--update-lock", action="store_true", help="registra os prompts atuais na trava")

    args = parser.parse_args(argv)
    _setup_logging()

    if args.command == "run":
        run_experiment(load_config(args.config), config_path=args.config, dry_run=args.dry_run)
    elif args.action == "list":
        catalog.list_benchmarks()
    elif args.action == "preview":
        catalog.preview(args.name, args.evaluation, args.n, args.num_fewshot)
    else:
        names = BENCHMARKS.names() if args.all else args.names
        if not names:
            parser.error("informe benchmarks ou --all")
        return 0 if catalog.check(names, args.lock, args.update_lock) else 1
    return 0


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    # Bibliotecas que reconfiguram os próprios loggers ao serem importadas: filtra no handler.
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda r: r.name.startswith("pt_lm_eval") or r.levelno >= logging.WARNING)


if __name__ == "__main__":
    sys.exit(main())
