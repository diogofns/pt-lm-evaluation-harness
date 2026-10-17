"""Linha de comando: `pt-lm-eval run` e `pt-lm-eval benchmarks list|preview|check`.

O `check` carrega todos os itens, monta todos os prompts, confere o cartão (card.md) e
compara o hash dos prompts com a trava (`benchmarks.lock.json`): se o prompt de um
benchmark mudar, a versão dele precisa mudar também, para resultados de prompts
diferentes nunca se misturarem.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from .benchmarks import BENCHMARKS
from .evaluations import EVALUATIONS
from .experiment import Experiment, load_config
from .experiment.report import describe_requests

CARD_SECTIONS = (
    "Fonte",
    "Licença",
    "Variedade do português",
    "Domínio",
    "Construção",
    "Contaminação",
    "Mantenedor",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pt-lm-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="executa um experimento")
    run.add_argument("config", type=Path)
    run.add_argument(
        "--dry-run", action="store_true", help="mostra o plano e confere se os prompts cabem no contexto"
    )
    bench = sub.add_parser("benchmarks", help="lista, mostra e valida benchmarks").add_subparsers(
        dest="action", required=True
    )
    bench.add_parser("list", help="benchmarks disponíveis")
    preview = bench.add_parser("preview", help="mostra os prompts sem chamar modelo")
    preview.add_argument("name")
    preview.add_argument("-e", "--evaluation")
    preview.add_argument("-n", type=int, default=2, help="quantos itens mostrar")
    preview.add_argument("--num-fewshot", type=int, default=0)
    check = bench.add_parser("check", help="valida dados, prompts, cartão e versão")
    check.add_argument("names", nargs="*")
    check.add_argument("--all", action="store_true", help="todos os benchmarks")
    check.add_argument("--lock", type=Path, default=Path("benchmarks.lock.json"), help="trava de versões")
    check.add_argument("--update-lock", action="store_true", help="registra os prompts atuais na trava")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    # Bibliotecas que reconfiguram os próprios loggers ao serem importadas: filtra no handler.
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda r: r.name.startswith("pt_lm_eval") or r.levelno >= logging.WARNING)

    if args.command == "run":
        experiment = Experiment(load_config(args.config), config_path=args.config)
        if args.dry_run:
            experiment.dry_run()
        else:
            experiment.run()
    elif args.action == "list":
        for name in BENCHMARKS.names():
            b = BENCHMARKS.get(name)
            print(f"{name:28} v{b.version:5} {','.join(b.evaluations):12} {b.description}")
    elif args.action == "preview":
        show_prompts(args.name, args.evaluation, args.n, args.num_fewshot)
    else:
        names = BENCHMARKS.names() if args.all else args.names
        if not names:
            parser.error("informe benchmarks ou --all")
        return 0 if check_benchmarks(names, args.lock, args.update_lock) else 1
    return 0


def show_prompts(name: str, evaluation: str | None = None, n: int = 2, num_fewshot: int = 0) -> None:
    bench = BENCHMARKS.get(name)
    for ev_name in [evaluation] if evaluation else bench.evaluations:
        for item in bench.data.items[:n]:
            shots = bench.select_shots(item, num_fewshot, seed=1234)
            desc = describe_requests(EVALUATIONS.get(ev_name).build(item, shots))
            print(f"=== {name} · {ev_name} · item {item.id} · gold={getattr(item, 'gold', '?')}")
            print("\n---\n".join(desc["prompts"]) + "\n---")
            if "continuations" in desc:
                print("continuações:", desc["continuations"])
            print()


def check_benchmarks(names: list[str], lock_path: Path, update_lock: bool = False) -> bool:
    """Valida os benchmarks contra a trava; devolve False se algum falhar."""
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    ok = True
    for name in names:
        try:
            ok &= _check_one(name, lock, update_lock)
        except Exception as exc:  # relata e segue para os demais
            print(f"FALHOU {name}: {type(exc).__name__}: {exc}")
            ok = False
    if update_lock:
        lock_path.write_text(json.dumps(dict(sorted(lock.items())), indent=2, ensure_ascii=False) + "\n")
        print(f"trava atualizada: {lock_path}")
    return ok


def _check_one(name: str, lock: dict, update_lock: bool) -> bool:
    bench = BENCHMARKS.get(name)
    loaded = bench.data
    items, pool = loaded.items, loaded.pool
    signatures = {}
    for ev_name in bench.evaluations:
        ev = EVALUATIONS.get(ev_name)
        for item in items:
            if not isinstance(item, ev.item_type):
                raise TypeError(
                    f"{ev_name} espera {ev.item_type.__name__}; item {item.id} é {type(item).__name__}"
                )
            ev.build(item, pool[:1])
        signatures[ev_name] = bench.prompt_signature(ev)

    problems = card_problems(bench.card)
    key = f"{name}@{bench.version}"
    if update_lock:
        lock[key] = {"source": bench.source.describe(), "prompts": signatures}
    elif key not in lock:
        problems.append(f"{key} não está em benchmarks.lock.json (rode check --update-lock)")
    elif lock[key]["prompts"] != signatures:
        changed = sorted(e for e in signatures if lock[key]["prompts"].get(e) != signatures[e])
        problems.append(f"prompts de {changed} mudaram sem mudar a versão {bench.version}: aumente a versão")

    subjects = Counter(getattr(it, "subject", None) for it in items)
    no_subject = subjects.pop(None, 0)
    status = "ok" if not problems else "FALHOU"
    extra = f" + {no_subject} itens sem matéria" if no_subject and subjects else ""
    counts = f"{len(items)} itens ({loaded.discarded} descartados), {len(subjects)} matérias{extra}"
    print(f"{status} {key}: {counts}")
    if all(hasattr(it, "options") for it in items):
        golds = Counter(it.gold for it in items)
        n_opts = Counter(len(it.options) for it in items)
        print(f"   opções {dict(sorted(n_opts.items()))}; gold {dict(sorted(golds.items()))}")
        print(f"   chutar a letra mais comum acerta {max(golds.values()) / len(items):.1%}")
    if all(hasattr(it, "instruction_ids") for it in items):
        per_item = Counter(len(it.instruction_ids) for it in items)
        kinds = {i for it in items for i in it.instruction_ids}
        total = sum(len(it.instruction_ids) for it in items)
        print(f"   {total} instruções de {len(kinds)} tipos; por prompt {dict(sorted(per_item.items()))}")
    print(f"   fonte: {bench.source.describe()}")
    for p in problems:
        print(f"   problema: {p}")
    return not problems


def card_problems(card: Path) -> list[str]:
    if not card.exists():
        return [f"cartão não encontrado: {card}"]
    headings = set(re.findall(r"^##\s+(.+?)\s*$", card.read_text(), flags=re.M))
    missing = [s for s in CARD_SECTIONS if s not in headings]
    return [f"cartão sem as seções {missing}"] if missing else []


if __name__ == "__main__":
    sys.exit(main())
