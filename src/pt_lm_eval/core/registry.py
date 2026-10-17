"""Registro por nome, com sugestão quando o nome não existe."""

from __future__ import annotations

import difflib
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, obj: T) -> T:
        if name in self._items:
            raise ValueError(f"{self.kind} '{name}' já registrado")
        self._items[name] = obj
        return obj

    def get(self, name: str) -> T:
        if name not in self._items:
            close = difflib.get_close_matches(name, self._items, n=3)
            hint = f" Quis dizer: {', '.join(close)}?" if close else ""
            raise KeyError(f"{self.kind} '{name}' não existe.{hint}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)
