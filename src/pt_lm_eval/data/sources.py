"""Fontes de dados. Cada linha recebe `_split` e `_index` para gerar IDs estáveis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HFDataset:
    """Dataset do Hugging Face Hub. `split="*"` carrega todos os splits.

    `subset` pode ser uma tupla (ex.: um config por ano): as linhas de todos são
    juntadas e cada uma recebe `_subset`.
    """

    repo: str
    split: str = "test"
    subset: str | tuple[str, ...] | None = None
    revision: str | None = None  # fixe um commit para resultados reproduzíveis

    @property
    def subsets(self) -> tuple[str | None, ...]:
        return self.subset if isinstance(self.subset, tuple) else (self.subset,)

    def splits(self, subset: str | None = None) -> list[str]:
        if self.split != "*":
            return [self.split]
        from datasets import get_dataset_split_names

        return get_dataset_split_names(self.repo, subset, revision=self.revision)

    def load(self) -> list[dict[str, Any]]:
        from datasets import load_dataset

        rows: list[dict[str, Any]] = []
        for subset in self.subsets:
            for split in self.splits(subset):
                ds = load_dataset(self.repo, subset, split=split, revision=self.revision)
                rows.extend(
                    {**row, "_subset": subset, "_split": split, "_index": i} for i, row in enumerate(ds)
                )
        return rows

    def describe(self) -> str:
        subsets = "+".join(s for s in self.subsets if s)
        name = f"{self.repo}/{subsets}" if subsets else self.repo
        return f"hf:{name}:{self.split}@{_resolve_revision(self.repo, self.revision)}"


@dataclass(frozen=True)
class LocalFile:
    """Arquivo local (.jsonl, .json, .csv ou .parquet)."""

    path: str

    def load(self) -> list[dict[str, Any]]:
        from datasets import load_dataset

        fmt = {"jsonl": "json", "json": "json", "csv": "csv", "parquet": "parquet"}
        kind = fmt[Path(self.path).suffix.lstrip(".")]
        ds = load_dataset(kind, data_files=self.path, split="train")
        return [{**row, "_split": "train", "_index": i} for i, row in enumerate(ds)]

    def describe(self) -> str:
        digest = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()[:12]
        return f"file:{self.path}@sha256:{digest}"


@cache
def _resolve_revision(repo: str, revision: str | None) -> str:
    """Commit exato do dataset, para registrar junto dos resultados."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo, revision=revision).sha or (revision or "main")
    except Exception:  # sem rede ou sem permissão: registra o que foi pedido
        return revision or "main"
