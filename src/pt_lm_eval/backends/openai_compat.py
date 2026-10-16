"""Backend para servidores OpenAI-compatíveis (vLLM, endpoints externos, APIs).

- loglikelihood: POST /completions com o prompt em token IDs e `prompt_logprobs`.
  O vLLM devolve o logprob e o rank de cada token do prompt; somamos os da
  continuação. Exige tokenizer local, então não existe para `type: api`.
- generate: POST /chat/completions (com chat template) ou /completions (sem).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from tqdm.asyncio import tqdm_asyncio

from ..core.protocols import BACKENDS
from ..core.types import (
    GenerateRequest,
    GenerateResponse,
    LoglikelihoodRequest,
    LoglikelihoodResponse,
)
from ..metrics.extraction import split_reasoning
from ..runner.config import ModelConfig
from .tokenization import PairTokenizer, truncate_left

DEFAULT_MAX_TOKENS = 1024
LOGLIKELIHOOD_BATCH = 16  # prompts por requisição de log-likelihood


@dataclass
class OpenAICompatSettings:
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


class OpenAICompatBackend:
    def __init__(self, settings: OpenAICompatSettings) -> None:
        self.s = settings
        self.capabilities = frozenset({"generate"} | ({"loglikelihood"} if settings.tokenizer else set()))
        self._pairs: PairTokenizer | None = None
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(settings.concurrency)

    # --- log-likelihood -----------------------------------------------------

    async def loglikelihood(self, requests: Sequence[LoglikelihoodRequest]) -> list[LoglikelihoodResponse]:
        pairs = self._pair_tokenizer()
        encoded = []
        for r in requests:
            ids, start = truncate_left(*pairs.encode_pair(r.context, r.continuation), self.s.max_length)
            encoded.append((ids, start))
        batches = [encoded[i : i + LOGLIKELIHOOD_BATCH] for i in range(0, len(encoded), LOGLIKELIHOOD_BATCH)]
        results = await tqdm_asyncio.gather(
            *(self._loglikelihood_batch(b) for b in batches), desc="loglikelihood", leave=False
        )
        return [r for batch in results for r in batch]

    async def _loglikelihood_batch(self, batch: list[tuple[list[int], int]]) -> list[LoglikelihoodResponse]:
        payload = {
            "model": self.s.model,
            "prompt": [ids for ids, _ in batch],
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 1,  # o token do prompt e o mais provável em cada posição
        }
        data = await self._post("/completions", payload)
        choices = sorted(data["choices"], key=lambda c: c["index"])
        return [
            _continuation_logprob(c["prompt_logprobs"], ids, start)
            for c, (ids, start) in zip(choices, batch, strict=True)
        ]

    def _pair_tokenizer(self) -> PairTokenizer:
        if self._pairs is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.s.tokenizer, trust_remote_code=True)
            self._pairs = PairTokenizer(tok, self.s.add_bos)
        return self._pairs

    # --- geração ------------------------------------------------------------

    async def generate(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]:
        return await tqdm_asyncio.gather(
            *(self._generate_one(r) for r in requests), desc="generate", leave=False
        )

    async def _generate_one(self, request: GenerateRequest) -> GenerateResponse:
        params = {"temperature": 0, **self.s.generate_params}
        # Vale o maior entre o pedido da tarefa e o do modelo: modelos de raciocínio
        # configurados com mais tokens não são cortados pelo limite de uma tarefa.
        limits = [n for n in (request.max_tokens, params.pop("max_tokens", None)) if n]
        payload: dict[str, Any] = {
            "model": self.s.model,
            self.s.max_tokens_field: max(limits, default=DEFAULT_MAX_TOKENS),
        }
        # Stops só em completions: servem para o modelo base não emendar a próxima
        # questão. Em chat, o vLLM os aplica também ao raciocínio e corta o <think>
        # quando o modelo cita o enunciado.
        if request.stop and not self.s.chat:
            payload["stop"] = list(request.stop)
        payload.update({k: v for k, v in params.items() if v is not None})
        try:
            if self.s.chat:
                messages = list(request.messages or [{"role": "user", "content": request.prompt}])
                if self.s.system_prompt and messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": self.s.system_prompt})
                data = await self._post("/chat/completions", {**payload, "messages": messages})
                choice = data["choices"][0]
                message = choice.get("message") or {}
                text = message.get("content") or ""
                reasoning = message.get("reasoning_content") or message.get("reasoning")
            else:
                data = await self._post("/completions", {**payload, "prompt": request.prompt})
                choice = data["choices"][0]
                text, reasoning = choice.get("text") or "", None
        except Exception as exc:  # falha definitiva vira resposta vazia, registrada na amostra
            return GenerateResponse(text="", error=f"{type(exc).__name__}: {exc}")
        if reasoning is None:
            text, reasoning = split_reasoning(text)
        return GenerateResponse(
            text=text.strip(), reasoning=reasoning, finish_reason=choice.get("finish_reason")
        )

    # --- HTTP ---------------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.s.api_key}"} if self.s.api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self.s.base_url.rstrip("/"), headers=headers, timeout=self.s.timeout
            )
        last: Exception | None = None
        for attempt in range(self.s.max_retries):
            try:
                async with self._semaphore:
                    response = await self._client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last = _with_body(exc)
                code = exc.response.status_code
                if not self.s.retry_client_errors and 400 <= code < 500 and code != 429:
                    break
            except httpx.TransportError as exc:
                last = exc
            await asyncio.sleep(min(30.0, 2.0**attempt))
        raise last or RuntimeError("falha na requisição")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _continuation_logprob(prompt_logprobs: list[Any], ids: list[int], start: int) -> LoglikelihoodResponse:
    """Soma os logprobs dos tokens a partir de `start`, conferindo que são os tokens enviados.

    `is_greedy` compara valores, como o lm-eval: um token empatado com o mais provável
    conta como guloso (empates exatos são comuns em bf16; o `rank` do vLLM desempata).
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


def _with_body(exc: httpx.HTTPStatusError) -> Exception:
    return RuntimeError(f"HTTP {exc.response.status_code}: {exc.response.text[:500]}")


# --- fábrica registrada -------------------------------------------------------


class OpenAICompatFactory:
    """Traduz a configuração do modelo para as configurações do backend."""

    def capabilities(self, model: ModelConfig) -> frozenset[str]:
        return frozenset({"generate"} if model.type == "api" else {"generate", "loglikelihood"})

    def create(self, model: ModelConfig, base_url: str, max_length: int | None) -> OpenAICompatBackend:
        is_api = model.type == "api"
        key_env = model.api_key_env or "OPENAI_API_KEY"
        params = dict(model.generate)
        if model.thinking is not None:
            # Convenção dos chat templates do Qwen3 e afins; templates sem a variável a ignoram.
            params["chat_template_kwargs"] = {
                **params.get("chat_template_kwargs", {}),
                "enable_thinking": model.thinking,
            }
        return OpenAICompatBackend(
            OpenAICompatSettings(
                model=model.name,
                base_url=base_url,
                chat=model.uses_chat,
                api_key=os.environ.get(key_env),
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
        )


BACKENDS.register("openai_compat", OpenAICompatFactory())
