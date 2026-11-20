from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..benchmarks import BENCHMARKS
from ..core import Item, Request
from ..evaluations import EVALUATIONS
from ..inference.server import slug

MODEL_TYPES = ("base", "instruct", "api")
SCENARIO = {
    "base": {"loglikelihood"},
    "instruct": {"loglikelihood", "generate"},
    "api": {"generate"},
}


@dataclass(frozen=True)
class ServerConfig:
    mode: str = "local"  # local | docker
    port: int = 8000
    image: str | None = None  # obrigatório no modo docker
    executable: str = "vllm"
    args: dict[str, Any] = field(default_factory=dict)  # flags do `vllm serve`, em snake_case
    startup_timeout: int = 1800

    def args_for(self, model: ModelConfig) -> dict[str, Any]:
        return {
            key.replace("-", "_"): value
            for args in (self.args, model.server_args)
            for key, value in args.items()
        }


@dataclass(frozen=True)
class ModelConfig:
    name: str
    type: str  # base | instruct | api
    base_url: str | None = None  # endpoint já existente: não sobe servidor
    api_key_env: str | None = None  # api.openai.com usa OPENAI_API_KEY por padrão
    tokenizer: str | None = None  # padrão: o próprio modelo (exceto api)
    add_bos: bool | None = None  # None = o que o tokenizer faz por padrão
    server_args: dict[str, Any] = field(default_factory=dict)  # somadas a server.args
    generate: dict[str, Any] = field(default_factory=dict)  # max_tokens, temperature, extras
    override_task_sampling: bool = False  # usa temperature/top_p do modelo mesmo em tarefas com valores fixos
    override_task_max_tokens: bool = False  # usa generate.max_tokens mesmo em tarefas com limite fixo
    system_prompt: str | None = None
    chat_template: bool | None = None  # None = sim para instruct e api, não para base
    thinking: bool | None = None  # liga/desliga o raciocínio (enable_thinking); None = padrão do modelo
    concurrency: int | None = None  # None = herda o experimento ou o padrão do cliente
    timeout: float = 600.0
    max_retries: int = 3
    label: str | None = None  # nome nos relatórios; padrão: name

    def __post_init__(self) -> None:
        _validate_concurrency(self.concurrency, f"modelo {self.name}")
        if type(self.override_task_sampling) is not bool:
            raise ValueError(f"modelo {self.name}: override_task_sampling deve ser booleano")
        if type(self.override_task_max_tokens) is not bool:
            raise ValueError(f"modelo {self.name}: override_task_max_tokens deve ser booleano")
        if self.type not in MODEL_TYPES:
            raise ValueError(f"modelo {self.name}: type deve ser um de {MODEL_TYPES}")
        if self.type == "api" and not self.base_url:
            raise ValueError(f"modelo {self.name}: type api precisa de base_url")
        if self.thinking is not None and self.type == "api":
            raise ValueError(
                f"modelo {self.name}: thinking vale para modelos servidos pelo vLLM; "
                "em APIs, use generate.reasoning_effort"
            )
        if self.thinking is not None and not self.uses_chat:
            raise ValueError(f"modelo {self.name}: thinking exige chat template")

    @property
    def display_name(self) -> str:
        """Nome nos relatórios e diretórios; com `thinking`, o modo entra no nome para que o
        mesmo modelo com e sem raciocínio fique em linhas separadas."""
        if self.label:
            return self.label
        if self.thinking is None:
            return self.name
        return f"{self.name}:{'thinking' if self.thinking else 'no-thinking'}"

    @property
    def uses_chat(self) -> bool:
        return self.chat_template if self.chat_template is not None else self.type != "base"


@dataclass(frozen=True)
class BenchmarkRun:
    name: str
    evaluations: tuple[str, ...] | None = None  # opcional: restringe as avaliações do cenário
    num_fewshot: int | None = None
    fewshot_as_multiturn: bool | None = None
    limit: int | str | None = None  # None = o geral; "all" = todos os itens, mesmo com limit geral
    exclude_ids: tuple[str, ...] = ()  # exclusões explícitas, iguais para todos os modelos
    fewshot_holdout: bool = False  # reserva primeiros N exemplos por grupo fora do eval

    def __post_init__(self) -> None:
        if type(self.fewshot_holdout) is not bool:
            raise ValueError(f"benchmark {self.name}: fewshot_holdout deve ser booleano")
        _validate_selection(self.num_fewshot, self.limit, f"benchmark {self.name}", allow_all=True)


@dataclass(frozen=True)
class Step:
    """Um passo do plano: um modelo, um benchmark e uma avaliação."""

    model: ModelConfig
    benchmark: str
    evaluation: str  # "-" quando o benchmark não tem avaliação para o cenário do modelo
    num_fewshot: int
    limit: int | None
    skip_reason: str | None = None
    fewshot_as_multiturn: bool = False  # efetivo: só com chat, geração e ao menos um exemplo
    exclude_ids: tuple[str, ...] = ()
    fewshot_holdout: bool = False

    @property
    def needs(self) -> frozenset[str]:
        return EVALUATIONS.get(self.evaluation).needs

    def build(self, item: Item, shots: list[Item]) -> list[Request]:
        return EVALUATIONS.get(self.evaluation).build(item, shots, multiturn=self.fewshot_as_multiturn)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str
    models: tuple[ModelConfig, ...]
    benchmarks: tuple[BenchmarkRun, ...]
    server: ServerConfig = field(default_factory=ServerConfig)
    output_dir: str = "output"
    seed: int = 1234
    limit: int | None = None  # itens por benchmark, sorteados com a seed
    num_fewshot: int | None = None  # None = padrão de cada benchmark
    fewshot_as_multiturn: bool = False  # em chat: exemplos como turnos usuário/assistente

    concurrency: int | None = None  # padrão global; None mantém 64 no vLLM e 8 em APIs
    gpus: tuple[str, ...] | None = None  # fila: um modelo por GPU; None = execução sequencial

    def __post_init__(self) -> None:
        _validate_concurrency(self.concurrency, "experimento")
        _validate_selection(self.num_fewshot, self.limit, "experimento")
        names = [slug(model.display_name) for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("modelos compartilham o diretório de saída; defina labels únicos")
        if any(name in {"", ".", "..", "logs"} for name in names):
            raise ValueError("nome de saída reservado; defina um label para o modelo")
        if self.gpus is not None:
            object.__setattr__(self, "gpus", _gpu_ids(self.gpus))
            self._validate_gpu_pool()
        if self.concurrency is not None:
            # Resolve antes de montar o plano: cliente, resultados e fingerprint usam
            # o mesmo valor, independentemente de onde ele foi escrito no YAML.
            object.__setattr__(
                self,
                "models",
                tuple(
                    dataclasses.replace(model, concurrency=self.concurrency)
                    if model.concurrency is None
                    else model
                    for model in self.models
                ),
            )

    def _validate_gpu_pool(self) -> None:
        if self.server.mode not in {"local", "docker"}:
            raise ValueError("gpus exige server.mode local ou docker")
        assert self.gpus is not None
        if type(self.server.port) is not int or not 1 <= self.server.port <= 65536 - len(self.gpus):
            raise ValueError("server.port e as portas consecutivas das GPUs devem estar entre 1 e 65535")
        for model in self.models:
            if model.base_url:
                raise ValueError("gpus exige modelos com vLLM gerenciado, sem base_url")
            args = self.server.args_for(model)
            if "port" in args:
                raise ValueError("com gpus, configure a porta inicial em server.port, não em args")
            for key in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
                if args.get(key, 1) != 1:
                    raise ValueError(f"modelo {model.name}: gpus aloca uma GPU por modelo; {key} deve ser 1")

    @property
    def out_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment

    def plan(self) -> list[Step]:
        """Para cada modelo, na ordem do YAML, cada benchmark com as avaliações do cenário."""
        steps = []
        for model in self.models:
            for run in self.benchmarks:
                bench = BENCHMARKS.get(run.name)
                limit = None if run.limit == "all" else run.limit if run.limit is not None else self.limit
                shots = bench.resolve_num_fewshot(run.num_fewshot, self.num_fewshot)
                evaluations = []
                for name in self._evaluations(run):
                    ev = EVALUATIONS.get(name)
                    allowed = SCENARIO[model.type]
                    if ev.supports_base_generation or name in bench.base_generation_evaluations:
                        allowed = allowed | {"generate"}
                    if ev.needs <= allowed:
                        evaluations.append(name)
                if not evaluations:
                    steps.append(
                        Step(model, run.name, "-", shots, limit, f"não se aplica a modelos {model.type}")
                    )
                for evaluation in evaluations:
                    multiturn = self._multiturn(run, model, evaluation, shots)
                    steps.append(
                        Step(
                            model,
                            run.name,
                            evaluation,
                            shots,
                            limit,
                            fewshot_as_multiturn=multiturn,
                            exclude_ids=run.exclude_ids,
                            fewshot_holdout=run.fewshot_holdout,
                        )
                    )
        return steps

    @staticmethod
    def _evaluations(run: BenchmarkRun) -> tuple[str, ...]:
        declared = BENCHMARKS.get(run.name).evaluations
        if run.evaluations is None:
            return declared
        unknown = [e for e in run.evaluations if e not in declared]
        if unknown:
            raise ValueError(f"{run.name}: avaliações {unknown} não declaradas no benchmark ({declared})")
        return run.evaluations

    def _multiturn(self, run: BenchmarkRun, model: ModelConfig, evaluation: str, shots: int) -> bool:
        wanted = self.fewshot_as_multiturn if run.fewshot_as_multiturn is None else run.fewshot_as_multiturn
        ev = EVALUATIONS.get(evaluation)
        if not (wanted and shots > 0 and model.uses_chat and ev.needs == {"generate"}):
            return False
        if not ev.supports_multiturn:
            raise ValueError(f"{run.name}: a avaliação {evaluation} não suporta fewshot_as_multiturn")
        return True


def _gpu_ids(values: Any) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("gpus deve ser uma lista não vazia de índices ou UUIDs de GPUs")
    ids = []
    for value in values:
        if type(value) is int and value >= 0:
            gpu = str(value)
        elif isinstance(value, str) and value and not any(c.isspace() or c == "," for c in value):
            if value.isdecimal():
                gpu = str(int(value))
            elif value.startswith(("GPU-", "MIG-")) and len(value) > 4:
                gpu = value
            else:
                raise ValueError(f"GPU inválida: {value!r}")
        else:
            raise ValueError(f"GPU inválida: {value!r}")
        if gpu in ids:
            raise ValueError(f"GPU repetida: {gpu}")
        ids.append(gpu)
    return tuple(ids)


def _validate_concurrency(value: int | None, where: str) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{where}: concurrency deve ser um inteiro positivo")


def _validate_selection(shots: int | None, limit: int | str | None, where: str, *, allow_all=False) -> None:
    if shots is not None and (type(shots) is not int or shots < 0):
        raise ValueError(f"{where}: num_fewshot deve ser um inteiro não negativo")
    if limit is None or (allow_all and limit == "all"):
        return
    if type(limit) is not int or limit < 1:
        options = " ou 'all'" if allow_all else ""
        raise ValueError(f"{where}: limit deve ser um inteiro positivo{options}")


def load_config(path: str | Path) -> ExperimentConfig:
    return parse_config(_read_config(path))


def load_profile_config(
    profile: Path,
    vllm_config: Path,
    benchmarks: Sequence[str],
    *,
    model: str | None = None,
    model_type: str | None = None,
    experiment: str | None = None,
    limit: int | None = None,
) -> ExperimentConfig:
    settings = _read_config(profile)
    name = settings.get("name") if model is None else model
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{profile}: informe --model com o ID, checkpoint ou pasta de checkpoints")
    if model_type is not None and model_type != settings.get("type"):
        settings["type"] = model_type
        settings.pop("chat_template", None)
        if model_type == "base":
            settings.pop("thinking", None)
    names = _checkpoint_names(name)
    models = [{**settings, "name": candidate} for candidate in names]
    gpus = None
    if len(models) > 1:
        if settings.get("base_url"):
            raise ValueError("uma pasta de checkpoints exige vLLM gerenciado, sem base_url")
        if label := settings.get("label"):
            for entry in models:
                entry["label"] = f"{label}/{Path(entry['name']).name}"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible not in {"", "-1"}:
            gpus = [gpu.strip() for gpu in visible.split(",")]
    return parse_config(
        {
            "experiment": experiment if experiment is not None else profile.stem,
            "limit": limit,
            "server": _read_config(vllm_config),
            "models": models,
            "gpus": gpus,
            "benchmarks": _select_benchmarks(
                benchmarks, prefer_generation=settings.get("type") == "instruct"
            ),
        }
    )


def _checkpoint_names(name: str) -> list[str]:
    name = os.path.expanduser(name)
    path = Path(name)
    if not path.is_dir() or (path / "config.json").is_file():
        return [name]
    checkpoints = [str(child) for child in sorted(path.iterdir()) if (child / "config.json").is_file()]
    if not checkpoints:
        raise ValueError(f"{path}: nenhum checkpoint com config.json nesta pasta ou nas subpastas diretas")
    return checkpoints


def _read_config(path: str | Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: YAML inválido: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: o YAML precisa ser um objeto")
    return data


def _select_benchmarks(selectors: Sequence[str], *, prefer_generation: bool) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for spec in selectors:
        name, sep, shots = spec.partition("|")
        if not name or (sep and not shots.isdecimal()):
            raise ValueError(f"benchmark inválido {spec!r}: use nome ou 'nome|shots' com shots >= 0")
        if name in selected:
            raise ValueError(f"benchmark {name!r} repetido em --benchmarks")
        run: dict[str, Any] = {"name": name, "num_fewshot": int(shots) if sep else None}
        if prefer_generation and "gen" in BENCHMARKS.get(name).evaluations:
            run["evaluations"] = ["gen"]
        selected[name] = run
    return list(selected.values())


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
    cfg = _build(ExperimentConfig, data, "experimento")
    if not cfg.models or not cfg.benchmarks:
        raise ValueError("o experimento precisa de ao menos um modelo e um benchmark")
    return cfg


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{where}: esperava um objeto, veio {type(data).__name__}")
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        valid = sorted(f.name for f in dataclasses.fields(cls))
        raise ValueError(f"{where}: chaves desconhecidas {sorted(unknown)}; válidas: {valid}")
    if data.get("evaluations") is not None:
        data = {**data, "evaluations": tuple(data["evaluations"])}
    if "exclude_ids" in data:
        ids = data["exclude_ids"]
        if not isinstance(ids, (list, tuple)) or any(not isinstance(i, str) for i in ids):
            raise ValueError(f"{where}: exclude_ids deve ser uma lista de IDs")
        data = {**data, "exclude_ids": tuple(sorted(set(ids)))}
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"{where}: {exc}") from exc
