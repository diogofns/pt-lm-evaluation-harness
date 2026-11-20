import asyncio
from types import SimpleNamespace

import pytest

from pt_lm_eval.core import GenerateRequest
from pt_lm_eval.experiment import lengths
from pt_lm_eval.experiment.config import ModelConfig
from pt_lm_eval.inference.client import Client


@pytest.mark.parametrize(
    "model_type,model_limit,task_limit,enforce,override,budget",
    [
        ("base", None, None, False, False, 1024),
        ("instruct", 2048, 256, False, False, 2048),
        ("instruct", 2048, 256, True, False, 256),
        ("base", 32, 128, False, False, 128),
        ("instruct", 64, None, True, False, 64),
        ("instruct", 8192, 2048, True, True, 8192),
        ("instruct", 8192, 2048, False, True, 8192),
        ("instruct", None, 2048, True, True, 2048),
    ],
)
def test_context_budget_matches_sent_request(
    monkeypatch, model_type, model_limit, task_limit, enforce, override, budget
):
    model = ModelConfig(
        name="model",
        type=model_type,
        generate={"max_tokens": model_limit},
        add_bos=False,
        override_task_max_tokens=override,
    )
    request = GenerateRequest(prompt="pergunta", max_tokens=task_limit, enforce_generation_config=enforce)
    tokenizer = _Tokenizer()
    monkeypatch.setattr(lengths, "_tokenizer", lambda name: tokenizer)
    monkeypatch.setattr(lengths.TokenCounter, "_caches", {})
    captured = []

    async def post(path, payload):
        captured.append((path, payload))
        return {"choices": [{"message": {"content": "ok"}, "text": "ok"}]}

    client = Client.for_model(model, "http://unused", None)
    monkeypatch.setattr(client, "_post", post)
    assert lengths.TokenCounter(model).sizes([[request]]) == [budget + len(request.prompt)]
    asyncio.run(client.generate([request]))
    path, payload = captured[0]
    assert payload["max_tokens"] == budget
    assert path == ("/chat/completions" if model_type == "instruct" else "/completions")
    assert model.generate == {"max_tokens": model_limit}


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("task_system", [True, False])
def test_context_chat_matches_sent_conversation(monkeypatch, thinking, task_system):
    model = ModelConfig(
        name="model",
        type="instruct",
        system_prompt="system do modelo",
        thinking=thinking,
        generate={"chat_template_kwargs": {"custom": "value"}},
    )
    turns = [
        {"role": "user", "content": "exemplo"},
        {"role": "assistant", "content": "resposta"},
        {"role": "user", "content": "pergunta"},
    ]
    system = {"role": "system", "content": "system da tarefa" if task_system else "system do modelo"}
    request = GenerateRequest(
        prompt="registro textual", messages=tuple([system, *turns] if task_system else turns)
    )
    tokenizer = _Tokenizer()
    monkeypatch.setattr(lengths, "_tokenizer", lambda name: tokenizer)
    monkeypatch.setattr(lengths.TokenCounter, "_caches", {})
    captured = []

    async def post(path, payload):
        captured.append(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    client = Client.for_model(model, "http://unused", None)
    monkeypatch.setattr(client, "_post", post)
    lengths.TokenCounter(model).sizes([[request]])
    asyncio.run(client.generate([request]))
    messages, kwargs = tokenizer.chat
    assert messages == captured[0]["messages"] == [system, *turns]
    assert kwargs == captured[0]["chat_template_kwargs"] == {"custom": "value", "enable_thinking": thinking}
    assert model.generate == {"chat_template_kwargs": {"custom": "value"}}
    assert request.messages == tuple([system, *turns] if task_system else turns)


class _Tokenizer:
    bos_token_id = None
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return SimpleNamespace(input_ids=list(range(len(text))))
        return {"input_ids": [list(range(len(s))) for s in text]}

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        assert add_generation_prompt and not tokenize
        self.chat = (messages, kwargs)
        return messages[-1]["content"]
