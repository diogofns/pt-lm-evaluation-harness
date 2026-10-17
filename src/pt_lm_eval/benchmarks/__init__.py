"""Benchmarks: um por pasta (benchmarks/<nome>/benchmark.py + card.md).

Importar este pacote importa cada `benchmark.py`, e cada subclasse concreta de
Benchmark se registra em BENCHMARKS.
"""

import importlib
from pathlib import Path

from .base import BENCHMARKS, Benchmark, ColumnBenchmark, Columns, HFDataset, Loaded, letter_index

for _module in sorted(Path(__file__).parent.glob("*/benchmark.py")):
    importlib.import_module(f"{__name__}.{_module.parent.name}.benchmark")

__all__ = ["BENCHMARKS", "Benchmark", "ColumnBenchmark", "Columns", "HFDataset", "Loaded", "letter_index"]
