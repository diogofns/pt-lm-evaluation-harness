"""Catálogo de benchmarks: listar, mostrar prompts e validar.

A validação (`check`) carrega todos os itens, monta todos os prompts, confere o
cartão (README.md) e compara o hash dos prompts com a trava de versões
(`benchmarks.lock.json`): se o prompt de um benchmark mudar, a versão dele
precisa mudar também, para resultados de prompts diferentes nunca se misturarem.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .core.protocols import BENCHMARKS, EVALUATIONS
from .core.registry import OFFICIAL_DISTRIBUTION
from .data.fewshot import select_shots
from .data.loading import load_items, prompt_signature
from .report.results import describe_requests

CARD_SECTIONS = (
    "Fonte",
    "Licença",
    "Variedade do português",
    "Domínio",
    "Construção",
    "Contaminação",
    "Mantenedor",
)


def list_benchmarks() -> None:
    for name in BENCHMARKS.names():
        b = BENCHMARKS.get(name)
        origin = BENCHMARKS.origin(name)
        tag = "oficial" if origin == OFFICIAL_DISTRIBUTION else f"externo ({origin})"
        print(f"{name:32} v{b.version:6} {','.join(b.evaluations):18} {tag}")


def preview(name: str, evaluation: str | None = None, n: int = 2, num_fewshot: int = 0) -> None:
    bench = BENCHMARKS.get(name)
    loaded = load_items(name)
    for ev_name in [evaluation] if evaluation else bench.evaluations:
        ev = EVALUATIONS.get(ev_name)
        for item in loaded.items[:n]:
            shots = select_shots(item, loaded.pool, num_fewshot, exclude_self=bench.fewshot_source is None)
            desc = describe_requests(ev.build(item, shots))
            print(f"=== {name} · {ev_name} · item {item.id} · gold={getattr(item, 'gold', '?')}")
            for prompt in desc["prompts"]:
                print(prompt)
                print("---")
            if "continuations" in desc:
                print("continuações:", desc["continuations"])
            print()


def check(names: list[str], lock_path: Path, update_lock: bool = False) -> bool:
    """Valida os benchmarks; devolve False se algum falhar."""
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
    loaded = load_items(name)
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
        signatures[ev_name] = prompt_signature(ev, items, pool)

    problems = card_problems(bench.card)
    key = f"{name}@{bench.version}"
    locked = lock.get(key)
    if update_lock:
        lock[key] = {"source": bench.source.describe(), "prompts": signatures}
    elif locked is None:
        problems.append(f"{key} não está em benchmarks.lock.json (rode check --update-lock)")
    elif locked["prompts"] != signatures:
        changed = sorted(e for e in signatures if locked["prompts"].get(e) != signatures[e])
        problems.append(f"prompts de {changed} mudaram sem mudar a versão {bench.version}: aumente a versão")

    golds = Counter(getattr(it, "gold", None) for it in items)
    n_opts = Counter(len(getattr(it, "options", ())) for it in items)
    subjects = Counter(getattr(it, "subject", None) for it in items)
    no_subject = subjects.pop(None, 0)
    status = "ok" if not problems else "FALHOU"
    print(
        f"{status} {key}: {len(items)} itens ({loaded.discarded} descartados), {len(subjects)} matérias"
        f"{f' + {no_subject} itens sem matéria' if no_subject and subjects else ''}, "
        f"opções {dict(sorted(n_opts.items()))}"
    )
    majority = max(golds.values()) / len(items)
    print(f"   gold {dict(sorted(golds.items()))}; chutar a letra mais comum acerta {majority:.1%}")
    print(f"   fonte: {bench.source.describe()}")
    for p in problems:
        print(f"   problema: {p}")
    return not problems


def card_problems(card: Path | None) -> list[str]:
    if card is None:
        return ["sem cartão (Benchmark.card)"]
    if not card.exists():
        return [f"cartão não encontrado: {card}"]
    headings = set(re.findall(r"^##\s+(.+?)\s*$", card.read_text(), flags=re.M))
    missing = [s for s in CARD_SECTIONS if s not in headings]
    return [f"cartão sem as seções {missing}"] if missing else []
