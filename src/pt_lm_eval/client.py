"""Cliente de servidores OpenAI-compatíveis (vLLM, endpoints externos, APIs).

- loglikelihood: POST /completions com o prompt em token IDs e `prompt_logprobs`; o vLLM
  devolve o logprob de cada token do prompt e somamos os da continuação. Exige
  tokenizer local, então não existe para `type: api`.
- generate: POST /chat/completions (com chat template) ou /completions (sem).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from tqdm.asyncio import tqdm_asyncio

from .core import GenerateRequest, GenerateResponse, LoglikelihoodRequest, LoglikelihoodResponse

if TYPE_CHECKING:
    from .config import ModelConfig

DEFAULT_MAX_TOKENS = 1024
LOGLIKELIHOOD_BATCH = 16  # prompts por requisição de log-likelihood


@dataclass
class Client:
    model: str
    base_url: str
    chat: bool
    api_key: str | None = None
    tokenizer: str | None = None  # None = sem log-likelihood
    add_bos: bool | None = None
    max_length: int | None = None
    concurrency: int = 32
    timeout: float = 600.0
    max_retries: int = 3
    retry_client_errors: bool = True  # False para APIs pagas: 4xx (exceto 429) não se repete
    max_tokens_field: str = "max_tokens"
    generate_params: dict[str, Any] = field(default_factory=dict)
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        self._pairs: PairTokenizer | None = None
        self._http: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(self.concurrency)

    @classmethod
    def for_model(cls, model: ModelConfig, base_url: str, max_length: int | None) -> Client:
        is_api = model.type == "api"
        params = dict(model.generate)
        if model.thinking is not None:
            # Convenção dos chat templates do Qwen3 e afins; templates sem a variável a ignoram.
            params["chat_template_kwargs"] = {
                **params.get("chat_template_kwargs", {}),
                "enable_thinking": model.thinking,
            }
        return cls(
            model=model.name,
            base_url=base_url,
            chat=model.uses_chat,
            api_key=os.environ.get(model.api_key_env or "OPENAI_API_KEY"),
            tokenizer=None if is_api else (model.tokenizer or model.name),
            add_bos=model.add_bos,
            max_length=max_length,
            concurrency=model.concurrency or (8 if is_api else 64),
            timeout=model.timeout,
            max_retries=model.max_retries,
            retry_client_errors=not is_api,
            max_tokens_field="max_completion_tokens" if is_api else "max_tokens",
            generate_params=params,
            system_prompt=model.system_prompt,
        )

    # --- log-likelihood -------------------------------------------------------------

    async def loglikelihood(self, requests: Sequence[LoglikelihoodRequest]) -> list[LoglikelihoodResponse]:
        pairs = self.pair_tokenizer()
        encoded = [
            truncate_left(*pairs.encode_pair(r.context, r.continuation), self.max_length) for r in requests
        ]
        batches = [encoded[i : i + LOGLIKELIHOOD_BATCH] for i in range(0, len(encoded), LOGLIKELIHOOD_BATCH)]
        results = await tqdm_asyncio.gather(
            *map(self._loglikelihood_batch, batches), desc="loglikelihood", leave=False
        )
        return [r for batch in results for r in batch]

    async def _loglikelihood_batch(self, batch: list[tuple[list[int], int]]) -> list[LoglikelihoodResponse]:
        payload = {
            "model": self.model,
            "prompt": [ids for ids, _ in batch],
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 1,  # o token do prompt e o mais provável em cada posição
        }
        data = await self._post("/completions", payload)
        choices = sorted(data["choices"], key=lambda c: c["index"])
        return [
            continuation_logprob(c["prompt_logprobs"], ids, start)
            for c, (ids, start) in zip(choices, batch, strict=True)
        ]

    def pair_tokenizer(self) -> PairTokenizer:
        if self._pairs is None:
            from transformers import AutoTokenizer

            self._pairs = PairTokenizer(
                AutoTokenizer.from_pretrained(self.tokenizer, trust_remote_code=True), self.add_bos
            )
        return self._pairs

    # --- geração --------------------------------------------------------------------

    async def generate(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]:
        return await tqdm_asyncio.gather(*map(self._generate_one, requests), desc="generate", leave=False)

    async def _generate_one(self, request: GenerateRequest) -> GenerateResponse:
        params = {"temperature": 0, **self.generate_params}
        # Vale o maior entre o da tarefa e o do modelo: modelos de raciocínio configurados
        # com mais tokens não são cortados pelo limite de uma tarefa.
        limits = [n for n in (request.max_tokens, params.pop("max_tokens", None)) if n]
        payload: dict[str, Any] = {
            "model": self.model,
            self.max_tokens_field: max(limits, default=DEFAULT_MAX_TOKENS),
        }
        # Stops só em completions (o modelo base não emenda a próxima questão); em chat, o
        # vLLM os aplica também ao raciocínio e corta o <think> quando o modelo cita o enunciado.
        if request.stop and not self.chat:
            payload["stop"] = list(request.stop)
        payload.update({k: v for k, v in params.items() if v is not None})
        try:
            if self.chat:
                messages = list(request.messages or [{"role": "user", "content": request.prompt}])
                if self.system_prompt and messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": self.system_prompt})
                choice = (await self._post("/chat/completions", {**payload, "messages": messages}))[
                    "choices"
                ][0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                reasoning = message.get("reasoning_content") or message.get("reasoning")
            else:
                choice = (await self._post("/completions", {**payload, "prompt": request.prompt}))["choices"][
                    0
                ]
                text, reasoning = choice.get("text") or "", None
        except Exception as exc:  # falha definitiva vira resposta vazia, registrada na amostra
            return GenerateResponse(text="", error=f"{type(exc).__name__}: {exc}")
        if reasoning is None:
            text, reasoning = split_reasoning(text)
        return GenerateResponse(
            text=text.strip(), reasoning=reasoning, finish_reason=choice.get("finish_reason")
        )

    # --- HTTP -----------------------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._http is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._http = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"), headers=headers, timeout=self.timeout
            )
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    response = await self._http.post(path, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last = RuntimeError(f"HTTP {exc.response.status_code}: {exc.response.text[:500]}")
                code = exc.response.status_code
                if not self.retry_client_errors and 400 <= code < 500 and code != 429:
                    break
            except httpx.TransportError as exc:
                last = exc
            await asyncio.sleep(min(30.0, 2.0**attempt))
        raise last or RuntimeError("falha na requisição")

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None


def continuation_logprob(prompt_logprobs: list[Any], ids: list[int], start: int) -> LoglikelihoodResponse:
    """Soma os logprobs dos tokens a partir de `start`, conferindo que são os tokens enviados.

    `is_greedy` compara valores, como o lm-eval: um token empatado com o mais provável conta
    como guloso (empates exatos são comuns em bf16; o `rank` do vLLM desempata).
    """
    if start < 1:
        raise ValueError("a continuação precisa de ao menos um token de contexto antes dela")
    total, greedy = 0.0, True
    for pos in range(start, len(ids)):
        candidates = prompt_logprobs[pos]
        logprob = candidates[str(ids[pos])]["logprob"]  # KeyError = servidor tokenizou diferente
        total += logprob
        greedy = greedy and logprob >= max(c["logprob"] for c in candidates.values())
    return LoglikelihoodResponse(logprob=total, is_greedy=greedy, n_tokens=len(ids) - start)


def split_reasoning(text: str) -> tuple[str, str | None]:
    """Separa '<think>…</think>resposta' em (resposta, raciocínio)."""
    if "</think>" not in text:
        return text, None
    reasoning, _, answer = text.rpartition("</think>")
    return answer.strip(), reasoning.replace("<think>", "").strip()


class PairTokenizer:
    """Tokenização de pares (contexto, continuação) para log-likelihood.

    `encode_pair` é portado do lm-evaluation-harness (lm_eval/api/model.py, _encode_pair e
    loglikelihood), Copyright (c) 2020 EleutherAI, licença MIT. Diferença: aqui o BOS é
    adicionado quando o tokenizer o usa por padrão (Llama, Gemma), em vez de nunca.
    """

    def __init__(self, tokenizer: Any, add_bos: bool | None = None) -> None:
        self.tokenizer = tokenizer
        bos = tokenizer.bos_token_id
        if add_bos is None:  # "auto": segue o que o tokenizer faz por padrão
            default = tokenizer("a", add_special_tokens=True).input_ids
            add_bos = bos is not None and len(default) > 0 and default[0] == bos
        self.bos = [bos] if add_bos else []
        # Contexto vazio: condiciona no BOS, ou no EOS se não houver BOS (como o lm-eval).
        self.prefix_token_id = bos if bos is not None else tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        if context == "":
            cont = self.encode(continuation)
            if cont and cont[0] == self.prefix_token_id:
                return cont[:1], cont[1:]
            return [self.prefix_token_id], cont
        # Espaços no fim do contexto vão para a continuação: " B" tokeniza como uma palavra,
        # "B" depois de "Resposta: " não.
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation, context = context[-n_spaces:] + continuation, context[:-n_spaces]
        whole = self.encode(context + continuation)
        context_ids = self.encode(context)
        return self.bos + context_ids, whole[len(context_ids) :]


def truncate_left(
    context_ids: list[int], continuation_ids: list[int], max_length: int | None
) -> tuple[list[int], int]:
    """Concatena e corta pela esquerda; devolve (ids, posição onde começa a continuação)."""
    ids = context_ids + continuation_ids
    if max_length is None or len(ids) <= max_length:
        return ids, len(context_ids)
    if len(continuation_ids) >= max_length:
        raise ValueError(
            f"continuação com {len(continuation_ids)} tokens não cabe em max_length={max_length}"
        )
    cut = len(ids) - max_length
    return ids[cut:], len(context_ids) - cut
