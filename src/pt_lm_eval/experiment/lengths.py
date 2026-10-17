"""Checagem de contexto do `--dry-run`: os prompts cabem no `max_model_len`?

Quem decide os limites é o YAML; aqui só se mede, com o tokenizer de cada modelo e os
mesmos itens e exemplos few-shot da execução. Na geração, prompt (com chat template,
system prompt e thinking) + `max_tokens` precisa caber, senão o vLLM recusa o pedido; no
log-likelihood, o que não cabe é cortado pela esquerda, e a checagem só informa.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar

from ..core import LoglikelihoodRequest, Request
from ..inference.client import DEFAULT_MAX_TOKENS, PairTokenizer

if TYPE_CHECKING:
    from .config import ModelConfig, Step


@dataclass(frozen=True)
class LengthCheck:
    """Resultado de um passo: maior prompt + geração reservada, em tokens, e quantos não cabem."""

    step: Step
    note: str | None = None  # por que não foi verificado
    generate: bool = False
    items: int = 0
    longest: int = 0
    limit: int = 0
    over: int = 0

    @property
    def ok(self) -> bool:
        return self.note is not None or self.over == 0

    def row(self) -> tuple[str, ...]:
        s = self.step
        if self.note:
            return (s.model.display_name, s.benchmark, s.evaluation, "", "", "", self.note)
        if not self.over:
            status = "ok"
        elif self.generate:
            status = f"{self.over} itens não cabem (o vLLM recusaria)"
        else:
            status = f"{self.over} itens seriam cortados pela esquerda"
        return (
            s.model.display_name,
            s.benchmark,
            s.evaluation,
            str(self.items),
            str(self.longest),
            str(self.limit),
            status,
        )


class TokenCounter:
    """Conta tokens dos pedidos de um modelo, em lote e com cache por tokenizer."""

    _caches: ClassVar[dict[str, dict[tuple[str, bool], int]]] = {}

    def __init__(self, model: ModelConfig) -> None:
        name = model.tokenizer or model.name
        self.model = model
        self.tokenizer = _tokenizer(name)
        self.bos = len(PairTokenizer(self.tokenizer, model.add_bos).bos)
        self.cache = self._caches.setdefault(name, {})

    def sizes(self, requests_per_item: Sequence[Sequence[Request]]) -> list[int]:
        """Por item, o maior pedido: prompt + geração reservada (ou contexto + continuação)."""
        parts = [[self._parts(r) for r in requests] for requests in requests_per_item]
        self.count_all([t for item in parts for _, texts in item for t in texts])  # em lote, bem mais rápido
        return [max(fixed + sum(self.count(*t) for t in texts) for fixed, texts in item) for item in parts]

    def _parts(self, request: Request) -> tuple[int, list[tuple[str, bool]]]:
        """Tokens fixos do pedido e os textos a contar, como (texto, com tokens especiais)."""
        model = self.model
        if isinstance(request, LoglikelihoodRequest):
            # Contexto e continuação contados à parte: pode diferir em ±1 token da junção real.
            return self.bos, [(request.context, False), (request.continuation, False)]
        limits = [n for n in (request.max_tokens, model.generate.get("max_tokens")) if n]
        reserved = max(limits, default=DEFAULT_MAX_TOKENS)
        if not model.uses_chat:
            return reserved, [(request.prompt, True)]
        messages = list(request.messages or [{"role": "user", "content": request.prompt}])
        if model.system_prompt and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": model.system_prompt})
        kwargs = dict(model.generate.get("chat_template_kwargs") or {})
        if model.thinking is not None:
            kwargs["enable_thinking"] = model.thinking
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **kwargs
        )
        return reserved, [(text, False)]

    def count_all(self, texts: list[tuple[str, bool]], batch: int = 1024) -> None:
        todo = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for special in (False, True):
            group = [text for text, s in todo if s is special]
            for start in range(0, len(group), batch):
                chunk = group[start : start + batch]
                ids = self.tokenizer(chunk, add_special_tokens=special)["input_ids"]
                self.cache.update({(text, special): len(i) for text, i in zip(chunk, ids, strict=True)})

    def count(self, text: str, special: bool) -> int:
        if (text, special) not in self.cache:
            self.count_all([(text, special)])
        return self.cache[(text, special)]


@cache
def _tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
