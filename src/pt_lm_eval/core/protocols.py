"""Contratos dos pontos de extensão e os registros onde eles são publicados.

Para estender o pt-lm-eval, implemente um destes protocolos e registre a
implementação com um nome no registro correspondente.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .registry import Registry
from .types import (
    GenerateRequest,
    GenerateResponse,
    Item,
    LoglikelihoodRequest,
    LoglikelihoodResponse,
    Request,
    Response,
    Scored,
)


class Source(Protocol):
    """De onde vêm as linhas de um benchmark."""

    def load(self) -> list[dict[str, Any]]: ...
    def describe(self) -> str: ...  # identificação reproduzível, ex.: "hf:org/ds:test@<sha>"


@dataclass(frozen=True)
class Benchmark:
    """O quê: dados e conversão de cada linha em item.

    `to_item` pode devolver None para descartar uma linha inválida; o total de
    descartes aparece no `benchmarks check` e no results.json.
    """

    version: str
    source: Source
    to_item: Callable[[dict[str, Any]], Item | None]
    evaluations: tuple[str, ...]
    fewshot_source: Source | None = None  # None = sorteia do próprio conjunto, sem o item
    description: str = ""
    card: Path | None = None  # README.md com fonte, licença, construção etc.


class Evaluation(Protocol):
    """Como: transforma itens em pedidos e pontua as respostas."""

    name: str
    item_type: type[Item]
    needs: frozenset[str]  # tipos de pedido usados, ex.: {"loglikelihood"}

    def build(self, item: Item, shots: Sequence[Item]) -> list[Request]: ...
    def score(self, item: Item, responses: Sequence[Response]) -> Scored: ...


class Backend(Protocol):
    """Quem responde: atende os tipos de pedido listados em `capabilities`."""

    capabilities: frozenset[str]

    async def loglikelihood(
        self, requests: Sequence[LoglikelihoodRequest]
    ) -> list[LoglikelihoodResponse]: ...
    async def generate(self, requests: Sequence[GenerateRequest]) -> list[GenerateResponse]: ...
    async def aclose(self) -> None: ...


class BackendFactory(Protocol):
    """Cria o backend de um modelo. `capabilities` é consultado antes de subir servidores."""

    def capabilities(self, model: Any) -> frozenset[str]: ...
    def create(self, model: Any, base_url: str, max_length: int | None) -> Backend: ...


class Reporter(Protocol):
    """Onde fica: grava os resultados de uma execução em algum formato."""

    def write(self, results: list[dict[str, Any]], out_dir: Path, experiment: str) -> Path: ...


BENCHMARKS: Registry[Benchmark] = Registry("benchmark", "pt_lm_eval.benchmarks")
EVALUATIONS: Registry[Evaluation] = Registry("avaliação", "pt_lm_eval.evaluations")
BACKENDS: Registry[BackendFactory] = Registry("backend", "pt_lm_eval.backends")
REPORTERS: Registry[Reporter] = Registry("relatório", "pt_lm_eval.reporters")
