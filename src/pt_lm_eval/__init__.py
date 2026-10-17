"""pt-lm-eval: avaliador de LLMs em português.

Importar o pacote registra as avaliações (mcq, ifeval) e os benchmarks.
"""

from . import benchmarks, ifeval, mcq  # noqa: F401  (registram o que definem)
from .core import BENCHMARKS, EVALUATIONS, LETTERS, Benchmark, InstructionItem, Item, MCQItem, Scored
from .data import HFDataset, columns, letter_index

__all__ = [
    "BENCHMARKS",
    "EVALUATIONS",
    "LETTERS",
    "Benchmark",
    "HFDataset",
    "InstructionItem",
    "Item",
    "MCQItem",
    "Scored",
    "columns",
    "letter_index",
]
