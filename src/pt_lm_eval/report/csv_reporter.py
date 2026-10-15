"""Relatório CSV: uma linha por modelo, uma coluna por benchmark@versão|avaliação|métrica.

É reconstruído a partir de todos os `*.results.json` do experimento, então
rodar de novo um modelo atualiza a linha dele sem afetar as outras. Versões
diferentes de um benchmark ficam em colunas diferentes e nunca se misturam.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from ..core.protocols import REPORTERS


class CSVReporter:
    def write(self, results: list[dict[str, Any]], out_dir: Path, experiment: str) -> Path:
        rows: dict[str, dict[str, Any]] = {}
        columns: dict[str, None] = {}
        for r in results:
            model = r["model"].get("label") or r["model"]["name"]
            prefix = f"{r['benchmark']['name']}@{r['benchmark']['version']}|{r['evaluation']['name']}"
            row = rows.setdefault(model, {"model": model})
            for metric, value in r["metrics"].items():
                col = f"{prefix}|{metric}"
                columns[col] = None
                row[col] = (
                    ""
                    if value is None or (isinstance(value, float) and math.isnan(value))
                    else round(value, 6)
                )
        path = out_dir / f"{experiment}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["model", *sorted(columns)])
            writer.writeheader()
            writer.writerows(rows.values())
        return path


REPORTERS.register("csv", CSVReporter())
