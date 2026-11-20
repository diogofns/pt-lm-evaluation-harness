from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core import GenerateRequest

if TYPE_CHECKING:
    from ..experiment.config import ModelConfig

DEFAULT_MAX_TOKENS = 1024


def generation_params(model: ModelConfig) -> dict[str, Any]:
    """Parâmetros do modelo, incluindo a opção de raciocínio do chat template."""
    params = dict(model.generate)
    if model.thinking is not None:
        params["chat_template_kwargs"] = {
            **(params.get("chat_template_kwargs") or {}),
            "enable_thinking": model.thinking,
        }
    return params


def token_budget(
    request: GenerateRequest, params: dict[str, Any], *, override_task_max_tokens: bool = False
) -> int:
    """Limite fixo da tarefa ou o maior limite entre tarefa e modelo."""
    if override_task_max_tokens and params.get("max_tokens"):
        return params["max_tokens"]
    if request.enforce_generation_config and request.max_tokens:
        return request.max_tokens
    return max((n for n in (request.max_tokens, params.get("max_tokens")) if n), default=DEFAULT_MAX_TOKENS)


def chat_messages(request: GenerateRequest, system_prompt: str | None) -> list[dict[str, str]]:
    """Conversa efetiva; um system do próprio pedido tem precedência sobre o do modelo."""
    messages = list(request.messages or [{"role": "user", "content": request.prompt}])
    if system_prompt and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages
