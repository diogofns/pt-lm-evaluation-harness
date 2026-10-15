"""Linha de comando: `pt-lm-eval run` e `pt-lm-eval benchmarks list|preview|check`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from .core.protocols import BENCHMARKS, EVALUATIONS
from .core.registry import OFFICIAL_DISTRIBUTION
from .data.fewshot import select_shots
from .report.results import describe_requests
from .runner.config import load_config
from .runner.execute import load_items, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pt-lm-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="executa um experimento")
    run.add_argument("config", type=Path)
    run.add_argument("--dry-run", action="store_true", help="só mostra o plano")

    bench = sub.add_parser("benchmarks", help="lista, mostra e valida benchmarks")
    bsub = bench.add_subparsers(dest="action", required=True)
    bsub.add_parser("list")
    preview = bsub.add_parser("preview", help="mostra os prompts sem chamar modelo")
    preview.add_argument("name")
    preview.add_argument("-e", "--evaluation")
    preview.add_argument("-n", type=int, default=2)
    preview.add_argument("--num-fewshot", type=int, default=0)
    check = bsub.add_parser("check", help="carrega todos os itens e monta todos os prompts")
    check.add_argument("names", nargs="*")
    check.add_argument("--all", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    # Bibliotecas que reconfiguram os próprios loggers ao serem importadas: filtra no handler.
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda r: r.name.startswith("pt_lm_eval") or r.levelno >= logging.WARNING)

    if args.command == "run":
        run_experiment(load_config(args.config), config_path=args.config, dry_run=args.dry_run)
    elif args.action == "list":
        _list()
    elif args.action == "preview":
        _preview(args.name, args.evaluation, args.n, args.num_fewshot)
    else:
        names = BENCHMARKS.names() if args.all else args.names
        if not names:
            parser.error("informe benchmarks ou --all")
        return 0 if all([_check(n) for n in names]) else 1
    return 0


def _list() -> None:
    for name in BENCHMARKS.names():
        b = BENCHMARKS.get(name)
        origin = BENCHMARKS.origin(name)
        tag = "oficial" if origin == OFFICIAL_DISTRIBUTION else f"externo ({origin})"
        print(f"{name:32} v{b.version:6} {','.join(b.evaluations):18} {tag}")


def _preview(name: str, evaluation: str | None, n: int, num_fewshot: int) -> None:
    bench = BENCHMARKS.get(name)
    loaded = load_items(name)
    items, pool = loaded.items, loaded.pool
    for ev_name in [evaluation] if evaluation else bench.evaluations:
        ev = EVALUATIONS.get(ev_name)
        for item in items[:n]:
            shots = select_shots(item, pool, num_fewshot, exclude_self=bench.fewshot_source is None)
            desc = describe_requests(ev.build(item, shots))
            print(f"=== {name} · {ev_name} · item {item.id} · gold={getattr(item, 'gold', '?')}")
            for prompt in desc["prompts"]:
                print(prompt)
                print("---")
            if "continuations" in desc:
                print("continuações:", desc["continuations"])
            print()


def _check(name: str) -> bool:
    """Valida um benchmark inteiro; devolve False se algo falhar."""
    try:
        bench = BENCHMARKS.get(name)
        loaded = load_items(name)
        items, pool = loaded.items, loaded.pool
        for ev_name in bench.evaluations:
            ev = EVALUATIONS.get(ev_name)
            for item in items:
                if not isinstance(item, ev.item_type):
                    raise TypeError(
                        f"{ev_name} espera {ev.item_type.__name__}, item {item.id} é {type(item).__name__}"
                    )
                ev.build(item, pool[:1])
    except Exception as exc:  # o objetivo é relatar, não interromper os demais
        print(f"FALHOU {name}: {type(exc).__name__}: {exc}")
        return False
    subjects = Counter(getattr(it, "subject", None) for it in items)
    golds = Counter(getattr(it, "gold", None) for it in items)
    n_opts = Counter(len(getattr(it, "options", ())) for it in items)
    print(
        f"ok {name} v{bench.version}: {len(items)} itens ({loaded.discarded} descartados), "
        f"{len(subjects)} matérias, opções {dict(sorted(n_opts.items()))}"
    )
    print(
        f"   gold {dict(sorted(golds.items()))}; sempre chutar a letra mais comum acerta "
        f"{max(golds.values()) / len(items):.1%}"
    )
    print(f"   fonte: {bench.source.describe()}")
    return True


if __name__ == "__main__":
    sys.exit(main())
