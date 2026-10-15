"""Leitura do YAML de experimento. Chaves desconhecidas são erro, para pegar erros de digitação."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MODEL_TYPES = ("base", "instruct", "api")


@dataclass(frozen=True)
class ServerConfig:
    kind: str = "vllm"
    mode: str = "local"  # local | docker
    port: int = 8000
    image: str | None = None  # obrigatório no modo docker
    executable: str = "vllm"
    args: dict[str, Any] = field(default_factory=dict)  # flags do `vllm serve`, em snake_case
    startup_timeout: int = 1800


@dataclass(frozen=True)
class ModelConfig:
    name: str
    type: str  # base | instruct | api
    backend: str = "openai_compat"
    base_url: str | None = None  # endpoint já existente: não sobe servidor
    api_key_env: str | None = None  # variável com a chave; padrão: OPENAI_API_KEY
    tokenizer: str | None = None  # padrão: o próprio modelo (exceto api)
    add_bos: bool | None = None  # None = o que o tokenizer faz por padrão
    server_args: dict[str, Any] = field(default_factory=dict)  # somadas a server.args
    generate: dict[str, Any] = field(default_factory=dict)  # max_tokens, temperature, extras
    system_prompt: str | None = None
    chat_template: bool | None = None  # None = sim para instruct e api, não para base
    concurrency: int | None = None
    timeout: float = 600.0
    max_retries: int = 3
    label: str | None = None  # nome nos relatórios; padrão: name

    def __post_init__(self) -> None:
        if self.type not in MODEL_TYPES:
            raise ValueError(f"modelo {self.name}: type deve ser um de {MODEL_TYPES}")
        if self.type == "api" and not self.base_url:
            raise ValueError(f"modelo {self.name}: type api precisa de base_url")

    @property
    def display_name(self) -> str:
        return self.label or self.name

    @property
    def uses_chat(self) -> bool:
        return self.chat_template if self.chat_template is not None else self.type != "base"


@dataclass(frozen=True)
class BenchmarkRun:
    name: str
    evaluations: tuple[str, ...] | None = None  # None = as declaradas no benchmark
    num_fewshot: int | None = None
    fewshot_strategy: str | None = None
    limit: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    models: tuple[ModelConfig, ...]
    benchmarks: tuple[BenchmarkRun, ...]
    server: ServerConfig = field(default_factory=ServerConfig)
    output_dir: str = "output"
    seed: int = 1234
    limit: int | None = None  # itens por benchmark, sorteados com a seed
    num_fewshot: int = 0
    fewshot_strategy: str = "random"  # random | first_n
    reporters: tuple[str, ...] = ("csv",)

    @property
    def out_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: o YAML precisa ser um objeto")
    return parse_config(data)


def parse_config(data: dict[str, Any]) -> ExperimentConfig:
    data = dict(data)
    data["server"] = _build(ServerConfig, data.get("server", {}), "server")
    data["models"] = tuple(
        _build(ModelConfig, m, f"models[{i}]") for i, m in enumerate(data.get("models", []))
    )
    data["benchmarks"] = tuple(
        _build(BenchmarkRun, {"name": b} if isinstance(b, str) else b, f"benchmarks[{i}]")
        for i, b in enumerate(data.get("benchmarks", []))
    )
    for key in ("reporters",):
        if key in data:
            data[key] = tuple(data[key])
    cfg = _build(ExperimentConfig, data, "experimento")
    if not cfg.models or not cfg.benchmarks:
        raise ValueError("o experimento precisa de ao menos um modelo e um benchmark")
    return cfg


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{where}: esperava um objeto, veio {type(data).__name__}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{where}: chaves desconhecidas {sorted(unknown)}; válidas: {sorted(known)}")
    if "evaluations" in data and data["evaluations"] is not None:
        data = {**data, "evaluations": tuple(data["evaluations"])}
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"{where}: {exc}") from exc
