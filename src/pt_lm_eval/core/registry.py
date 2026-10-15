"""O único mecanismo de extensão do pt-lm-eval.

Benchmarks, avaliações, backends e relatórios são registrados por nome num
`Registry`. Os plugins (inclusive os que vêm com o projeto) são módulos
declarados em entry points do Python; importá-los registra o que definem.
"""

from __future__ import annotations

import difflib
from importlib.metadata import entry_points
from typing import Generic, TypeVar

T = TypeVar("T")

OFFICIAL_DISTRIBUTION = "pt-lm-eval"


class Registry(Generic[T]):
    def __init__(self, kind: str, entry_point_group: str) -> None:
        self.kind = kind
        self.group = entry_point_group
        self._items: dict[str, T] = {}
        self._origin: dict[str, str] = {}
        self._loading_from = OFFICIAL_DISTRIBUTION
        self._loaded = False

    def register(self, name: str, obj: T) -> T:
        if name in self._items:
            raise ValueError(f"{self.kind} '{name}' já registrado ({self._origin[name]})")
        self._items[name] = obj
        self._origin[name] = self._loading_from
        return obj

    def get(self, name: str) -> T:
        self._load_plugins()
        if name not in self._items:
            close = difflib.get_close_matches(name, self._items, n=3)
            hint = f" Quis dizer: {', '.join(close)}?" if close else ""
            raise KeyError(f"{self.kind} '{name}' não existe.{hint}")
        return self._items[name]

    def names(self) -> list[str]:
        self._load_plugins()
        return sorted(self._items)

    def origin(self, name: str) -> str:
        """Pacote que registrou o nome; 'pt-lm-eval' indica um plugin oficial."""
        self._load_plugins()
        return self._origin[name]

    def _load_plugins(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for ep in sorted(entry_points(group=self.group), key=lambda e: e.value):
            self._loading_from = ep.dist.name if ep.dist else "?"
            try:
                ep.load()
            finally:
                self._loading_from = OFFICIAL_DISTRIBUTION
