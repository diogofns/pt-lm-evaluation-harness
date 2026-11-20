from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from ..core import Item, Registry, Request, Response, Scored


class Evaluation(ABC):
    """Uma forma de avaliar itens. Subclasses concretas com `name` se registram em EVALUATIONS.

    - `needs`: tipos de pedido usados; decidem o cenário em que a avaliação roda
      (base: log-likelihood; instruct/api: geração).
    - `supports_multiturn`: aceita few-shot como turnos usuário/assistente.
    - `scoring_version`: entra na identidade das amostras; aumente ao mudar a pontuação.
    """

    name: ClassVar[str]
    item_type: ClassVar[type[Item]]
    needs: ClassVar[frozenset[str]]
    supports_multiturn: ClassVar[bool] = False
    supports_fewshot: ClassVar[bool] = True
    supports_base_generation: ClassVar[bool] = False
    scoring_version: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls) and "name" in cls.__dict__:
            EVALUATIONS.register(cls.name, cls())

    @abstractmethod
    def build(self, item: Item, shots: Sequence[Item], multiturn: bool = False) -> list[Request]: ...

    @abstractmethod
    def score(self, item: Item, responses: Sequence[Response]) -> Scored: ...


EVALUATIONS: Registry[Evaluation] = Registry("avaliação")
