"""pt-lm-eval: avaliador de LLMs em português.

Importar o pacote registra as avaliações (evaluations/) e os benchmarks (benchmarks/).
"""

from .benchmarks import BENCHMARKS, Benchmark, ColumnBenchmark, Columns, HFDataset
from .core import LETTERS, InstructionItem, Item, MCQItem, Scored
from .evaluations import EVALUATIONS, Evaluation

__all__ = [
    "BENCHMARKS",
    "EVALUATIONS",
    "LETTERS",
    "Benchmark",
    "ColumnBenchmark",
    "Columns",
    "Evaluation",
    "HFDataset",
    "InstructionItem",
    "Item",
    "MCQItem",
    "Scored",
]
