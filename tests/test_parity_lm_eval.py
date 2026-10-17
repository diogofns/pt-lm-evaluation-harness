"""F2: paridade com o lm-evaluation-harness, pedido a pedido.

Reenvia pelo nosso backend exatamente os pares (contexto, continuação) que o
lm-eval enviou, e compara:
- logprob e is_greedy de cada pedido (valida tokenização, HTTP e leitura dos logprobs);
- acc e acc_norm recalculados pelas nossas métricas sobre os logprobs do lm-eval
  (valida as métricas isoladamente).

Preparação (mesmo servidor vLLM para os dois):

    lm_eval --model local-completions \\
      --model_args model=$MODELO,base_url=$URL/completions,tokenizer_backend=huggingface \\
      --tasks mmmlu_pt_br_anatomy,... --num_fewshot 5 --limit 60 --log_samples --output_path $SAIDA
    PT_LM_EVAL_URL=$URL PT_LM_EVAL_MODEL=$MODELO PT_LM_EVAL_PARITY_DIR=$SAIDA pytest -m parity
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from pt_lm_eval import EVALUATIONS, MCQItem
from pt_lm_eval.client import Client
from pt_lm_eval.core import LoglikelihoodRequest, LoglikelihoodResponse
from pt_lm_eval.mcq import argmax

pytestmark = pytest.mark.parity


def _load_lm_eval_samples(root: Path) -> list[dict]:
    docs = []
    for path in sorted(root.rglob("samples_*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            args = [r["arguments"][k] for k in sorted(r["arguments"], key=lambda k: int(k.split("_")[-1]))]
            docs.append(
                {
                    "file": path.name,
                    "pairs": [(a["arg_0"], a["arg_1"]) for a in args],
                    "lls": [float(x[0][0]) for x in r["resps"]],
                    "greedy": [x[0][1] == "True" for x in r["resps"]],
                    "gold": "ABCD".index(r["target"]),
                    "acc": r["acc"],
                    "acc_norm": r["acc_norm"],
                }
            )
    return docs


@pytest.fixture(scope="module")
def parity():
    url = os.environ.get("PT_LM_EVAL_URL")
    model = os.environ.get("PT_LM_EVAL_MODEL")
    root = os.environ.get("PT_LM_EVAL_PARITY_DIR")
    if not (url and model and root):
        pytest.skip("defina PT_LM_EVAL_URL, PT_LM_EVAL_MODEL e PT_LM_EVAL_PARITY_DIR")
    docs = _load_lm_eval_samples(Path(root))
    # add_bos=False: o lm-eval não adiciona BOS nas requisições de API.
    backend = Client(model=model, base_url=url, chat=False, tokenizer=model, add_bos=False)

    async def run():
        try:
            pairs = [LoglikelihoodRequest(c, x) for d in docs for c, x in d["pairs"]]
            return await backend.loglikelihood(pairs)
        finally:
            await backend.aclose()

    ours = iter(asyncio.run(run()))
    for d in docs:
        d["ours"] = [next(ours) for _ in d["pairs"]]
    return docs


def test_metrics_match_lm_eval_on_its_own_logprobs(parity):
    """Mesmos logprobs → mesmas métricas: acc idêntico e acc_norm = nosso acc_char."""
    for d in parity:
        fake = [LoglikelihoodResponse(lp, False, 1) for lp in d["lls"]]
        item = MCQItem(id="x", question="?", options=tuple(c.strip() for _, c in d["pairs"]), gold=d["gold"])
        got = EVALUATIONS.get("cf").score(item, fake).metrics
        assert got["acc"] == d["acc"]
        assert got["acc_char"] == d["acc_norm"]


def test_logprobs_match_lm_eval_request_by_request(parity):
    deltas, same_greedy, n = [], 0, 0
    for d in parity:
        for ours, lp, greedy in zip(d["ours"], d["lls"], d["greedy"], strict=True):
            deltas.append(abs(ours.logprob - lp))
            same_greedy += ours.is_greedy == greedy
            n += 1
    deltas.sort()
    median, p99 = deltas[len(deltas) // 2], deltas[int(0.99 * (len(deltas) - 1))]
    print(
        f"\n{n} pedidos: |Δ| mediana={median:.2e} p99={p99:.2e} máx={deltas[-1]:.2e}; "
        f"is_greedy igual em {same_greedy / n:.2%}"
    )
    assert median <= 1e-3
    assert same_greedy / n >= 0.995


def test_predictions_match_lm_eval(parity):
    same = sum(argmax([r.logprob for r in d["ours"]]) == argmax(d["lls"]) for d in parity)
    ours_acc = sum(argmax([r.logprob for r in d["ours"]]) == d["gold"] for d in parity) / len(parity)
    ref_acc = sum(d["acc"] for d in parity) / len(parity)
    print(
        f"\n{len(parity)} itens: predição igual em {same / len(parity):.2%}; "
        f"acc nosso={ours_acc:.4f} lm-eval={ref_acc:.4f}"
    )
    assert same / len(parity) >= 0.995
    assert abs(ours_acc - ref_acc) < 0.005
