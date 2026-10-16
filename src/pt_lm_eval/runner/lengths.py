"""Checagem de contexto do `--dry-run`: os prompts do experimento cabem no `max_model_len`?

Quem decide os limites é o YAML (`max_model_len`, `num_fewshot`, `max_tokens`,
thinking); aqui só se mede, com o tokenizer de cada modelo e os mesmos itens e
exemplos few-shot da execução, e nada é cortado ou alterado.

- geração: prompt (com chat template, system prompt e thinking) + `max_tokens` precisa
  caber; senão o vLLM recusa o pedido e o item conta como erro do modelo.
- log-likelihood: o que não cabe é cortado pela esquerda (como no lm-eval); a
  checagem só informa quantos itens seriam cortados.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

from ..backends.openai_compat import DEFAULT_MAX_TOKENS
from ..backends.tokenization import PairTokenizer
from ..core.protocols import EVALUATIONS
from ..core.types import GenerateRequest, LoglikelihoodRequest
from ..integrations.lm_eval import is_lm_eval
from ..servers import server_args
from .config import ExperimentConfig, ModelConfig
from .execute import build_requests, select_items
from .plan import Step


@dataclass(frozen=True)
class LengthCheck:
    step: Step
    kind: str  # "generate" | "loglikelihood" | ""
    n_items: int = 0
    longest: int = 0  # maior prompt + geração reservada, em tokens
    limit: int | None = None  # max_model_len
    over: int = 0  # itens que não cabem
    note: str | None = None  # motivo de não ter verificado

    @property
    def ok(self) -> bool:
        return self.note is not None or self.over == 0


def check_lengths(steps: list[Step], cfg: ExperimentConfig) -> list[LengthCheck]:
    return [_check(s, cfg) for s in steps if not s.skip_reason]


def _check(step: Step, cfg: ExperimentConfig) -> LengthCheck:
    model = step.model
    if is_lm_eval(step.benchmark):
        return LengthCheck(step, "", note="tarefa do lm-eval: não verificada")
    if model.type == "api":
        return LengthCheck(step, "", note="API: sem tokenizer local")
    limit = server_args(model, cfg.server).get("max_model_len")
    if not limit:
        return LengthCheck(step, "", note="max_model_len não definido no YAML")
    limit = int(limit)
    evaluation = EVALUATIONS.get(step.evaluation)
    counter = _Counter(model.tokenizer or model.name, model.add_bos)
    selected, shots = select_items(step, cfg)
    parts = [
        [_parts(r, model, counter) for r in build_requests(step, evaluation, it, shots[it.id])]
        for it in selected
    ]
    # Todos os textos de uma vez: tokenizar em lote é muito mais rápido que um a um.
    counter.count_all([t for item in parts for _, texts in item for t in texts])
    sizes = [max(fixed + sum(counter.count(*t) for t in texts) for fixed, texts in item) for item in parts]
    kind = "generate" if "generate" in evaluation.needs else "loglikelihood"
    budget = limit if kind == "generate" else limit - 1  # log-likelihood: 1 token fica para a geração
    return LengthCheck(step, kind, len(sizes), max(sizes, default=0), limit, sum(s > budget for s in sizes))


def _parts(request: Any, model: ModelConfig, counter: _Counter) -> tuple[int, list[tuple[str, bool]]]:
    """Tokens fixos do pedido e os textos a contar, como (texto, com tokens especiais)."""
    if isinstance(request, LoglikelihoodRequest):
        # Contexto e continuação contados à parte: pode diferir em ±1 token da junção real.
        return counter.bos, [(request.context, False), (request.continuation, False)]
    assert isinstance(request, GenerateRequest)
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
    return reserved, [(counter.chat_text(messages, kwargs), False)]


_COUNTS: dict[str, dict[tuple[str, bool], int]] = {}  # por tokenizer: (texto, especiais) → tokens


class _Counter:
    """Contagem de tokens com cache, compartilhada entre modelos com o mesmo tokenizer."""

    def __init__(self, name: str, add_bos: bool | None) -> None:
        self.tokenizer = _tokenizer(name)
        self.bos = len(PairTokenizer(self.tokenizer, add_bos).bos)
        self.cache = _COUNTS.setdefault(name, {})

    def chat_text(self, messages: list[dict[str, str]], kwargs: dict[str, Any]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **kwargs
        )

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


def format_lengths(checks: list[LengthCheck]) -> str:
    rows = [("modelo", "benchmark", "avaliação", "itens", "maior", "limite", "situação")]
    for c in checks:
        s = c.step
        if c.note:
            status = c.note
        elif c.over == 0:
            status = "ok"
        elif c.kind == "generate":
            status = f"{c.over} itens não cabem (o vLLM recusaria)"
        else:
            status = f"{c.over} itens seriam cortados pela esquerda"
        rows.append(
            (
                s.model.display_name,
                s.benchmark,
                s.evaluation,
                str(c.n_items) if not c.note else "",
                str(c.longest) if not c.note else "",
                str(c.limit) if c.limit else "",
                status,
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows)
