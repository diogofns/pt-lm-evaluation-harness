"""Testes de contrato para plugins. Use no CI do seu pacote:

from pt_lm_eval.testing import check_benchmark, check_evaluation

def test_meu_benchmark():
    check_benchmark("meu_benchmark")          # baixa o dataset

def test_minha_avaliacao():
    check_evaluation("minha_avaliacao", item_de_exemplo)
"""

from __future__ import annotations

from pathlib import Path

from .catalog import CARD_SECTIONS, card_problems, check
from .core.protocols import BENCHMARKS, EVALUATIONS
from .core.types import GenerateResponse, Item, LoglikelihoodResponse, Request, Response
from .data.loading import load_items


def check_evaluation(name: str, item: Item, shot: Item | None = None) -> None:
    """A avaliação monta pedidos dos tipos declarados e pontua respostas sintéticas."""
    ev = EVALUATIONS.get(name)
    assert isinstance(item, ev.item_type), f"{name} espera {ev.item_type.__name__}"
    for shots in ([], [shot or item]):
        requests = ev.build(item, shots)
        assert requests, f"{name}: build não gerou pedidos"
        kinds = {r.kind for r in requests}
        assert kinds <= ev.needs, f"{name}: pedidos {kinds} fora de needs={set(ev.needs)}"
        scored = ev.score(item, [_fake_response(r) for r in requests])
        assert scored.metrics, f"{name}: score sem métricas"
        for metric, value in scored.metrics.items():
            assert isinstance(value, float), f"{name}: métrica {metric} não é float"


def check_benchmark(name: str, lock: str | Path | None = None) -> None:
    """O benchmark carrega, todos os itens servem às avaliações declaradas e o cartão está completo.

    Com `lock`, também exige que os prompts batam com a trava de versões, como o
    `pt-lm-eval benchmarks check` (prompt mudou sem a versão mudar = falha).
    """
    bench = BENCHMARKS.get(name)
    assert bench.version, f"{name}: sem versão"
    for ev_name in bench.evaluations:
        EVALUATIONS.get(ev_name)  # KeyError com sugestão se não existir
    problems = card_problems(bench.card)
    assert not problems, f"{name}: {problems} (seções exigidas: {CARD_SECTIONS})"
    loaded = load_items(name)
    assert loaded.items, f"{name}: nenhum item carregado"
    for ev_name in bench.evaluations:
        check_evaluation(ev_name, loaded.items[0], loaded.pool[-1])
    if lock is not None:
        assert check([name], Path(lock)), f"{name}: falhou na trava {lock} (veja a saída acima)"


def _fake_response(request: Request) -> Response:
    if request.kind == "loglikelihood":
        return LoglikelihoodResponse(logprob=-1.0, is_greedy=False, n_tokens=1)
    return GenerateResponse(text="Resposta: A", finish_reason="stop")
