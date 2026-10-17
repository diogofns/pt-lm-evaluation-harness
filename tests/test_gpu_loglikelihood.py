"""F0: o log-likelihood vindo do servidor vLLM bate com um forward do transformers.

Precisa de um servidor vLLM no ar e de torch:

    PT_LM_EVAL_URL=http://127.0.0.1:8000/v1 PT_LM_EVAL_MODEL=Qwen/Qwen2.5-0.5B pytest -m gpu
"""

from __future__ import annotations

import asyncio
import os

import pytest

from pt_lm_eval.core import LoglikelihoodRequest
from pt_lm_eval.inference.client import Client

pytestmark = pytest.mark.gpu

PAIRS = [
    ("Pergunta: Qual é a capital do Brasil?\nResposta:", " Brasília"),
    ("Pergunta: Qual é a capital do Brasil?\nResposta:", " São Paulo"),
    ("Pergunta: Quanto é 2 + 2?\nA. 3\nB. 4\nC. 5\nResposta:", " B"),
    ("Pergunta: Quanto é 2 + 2?\nA. 3\nB. 4\nC. 5\nResposta:", " A"),
    ("O rio mais extenso do Brasil é o", " Amazonas"),
    ("A Constituição Federal brasileira foi promulgada em", " 1988"),
    ("Resposta: ", "B"),  # espaço no fim do contexto: vai para a continuação
    ("Pergunta: Que dia vem depois de segunda-feira?\nResposta:", " terça-feira"),
    ("A fotossíntese ocorre principalmente nas", " folhas das plantas verdes"),
    ("Em Goiás, a capital é", " Goiânia"),
]


@pytest.fixture(scope="module")
def setup():
    url, model = os.environ.get("PT_LM_EVAL_URL"), os.environ.get("PT_LM_EVAL_MODEL")
    if not url or not model:
        pytest.skip("defina PT_LM_EVAL_URL e PT_LM_EVAL_MODEL")
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).cuda().eval()
    backend = Client(model=model, base_url=url, chat=False, tokenizer=model)
    return torch, hf, backend


def _hf_loglikelihood(torch, hf, backend, context, continuation):
    ctx, cont = backend.pair_tokenizer().encode_pair(context, continuation)
    ids = torch.tensor([ctx + cont]).cuda()
    with torch.no_grad():
        logprobs = torch.log_softmax(hf(ids).logits.float(), dim=-1)[0]
    targets = ids[0, len(ctx) :]
    picked = logprobs[len(ctx) - 1 : -1].gather(-1, targets[:, None])[:, 0]
    greedy = bool((logprobs[len(ctx) - 1 : -1].argmax(-1) == targets).all())
    return float(picked.sum()), greedy, len(cont)


def test_server_matches_transformers(setup):
    torch, hf, backend = setup

    async def run():
        try:
            return await backend.loglikelihood([LoglikelihoodRequest(c, x) for c, x in PAIRS])
        finally:
            await backend.aclose()

    served = asyncio.run(run())
    for (context, continuation), got in zip(PAIRS, served, strict=True):
        ref_lp, ref_greedy, ref_n = _hf_loglikelihood(torch, hf, backend, context, continuation)
        assert got.n_tokens == ref_n
        # bf16 nos dois lados, kernels diferentes: tolerância por token
        assert abs(got.logprob - ref_lp) <= 0.05 * ref_n + 0.02, (context, continuation, got.logprob, ref_lp)
        if abs(got.logprob - ref_lp) < 0.01:
            assert got.is_greedy == ref_greedy
