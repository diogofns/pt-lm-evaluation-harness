"""pt-lm-eval: avaliador de LLMs em português.

Este módulo é a API pública. Plugins devem importar apenas daqui; o resto do
pacote pode mudar entre versões.
"""

from .core.protocols import (
    BACKENDS,
    BENCHMARKS,
    EVALUATIONS,
    REPORTERS,
    Backend,
    BackendFactory,
    Benchmark,
    Evaluation,
    Reporter,
    Source,
)
from .core.types import (
    LETTERS,
    GenerateRequest,
    GenerateResponse,
    Item,
    LoglikelihoodRequest,
    LoglikelihoodResponse,
    MCQItem,
    Scored,
)
from .data.columns import columns, letter_index
from .data.sources import HFDataset, LocalFile

__all__ = [
    "BACKENDS",
    "BENCHMARKS",
    "EVALUATIONS",
    "LETTERS",
    "REPORTERS",
    "Backend",
    "BackendFactory",
    "Benchmark",
    "Evaluation",
    "GenerateRequest",
    "GenerateResponse",
    "HFDataset",
    "Item",
    "LocalFile",
    "LoglikelihoodRequest",
    "LoglikelihoodResponse",
    "MCQItem",
    "Reporter",
    "Scored",
    "Source",
    "columns",
    "letter_index",
]
