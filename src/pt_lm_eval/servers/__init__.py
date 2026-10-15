"""Escolhe de onde vem o endpoint de um modelo: servidor gerenciado ou URL existente."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..runner.config import ModelConfig, ServerConfig
from .vllm import vllm_server


def server_args(model: ModelConfig, server: ServerConfig) -> dict[str, Any]:
    return {**server.args, **model.server_args}


def max_model_len(model: ModelConfig, server: ServerConfig) -> int | None:
    """Contexto máximo conhecido; o log-likelihood reserva 1 token para a geração mínima."""
    value = server_args(model, server).get("max_model_len")
    return int(value) - 1 if value else None


@contextmanager
def model_endpoint(model: ModelConfig, server: ServerConfig, log_dir: Path) -> Iterator[str]:
    if model.base_url:
        yield model.base_url
        return
    if server.kind != "vllm":
        raise ValueError(f"server.kind desconhecido: {server.kind}")
    with vllm_server(
        model.name,
        args=server_args(model, server),
        log_path=log_dir / f"{slug(model.display_name)}.vllm.log",
        mode=server.mode,
        port=server.port,
        image=server.image,
        executable=server.executable,
        startup_timeout=server.startup_timeout,
    ) as base_url:
        yield base_url


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name.strip("/"))
