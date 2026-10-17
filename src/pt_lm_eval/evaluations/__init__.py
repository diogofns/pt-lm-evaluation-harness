"""Avaliações: cada subclasse concreta de Evaluation se registra em EVALUATIONS."""

from .base import EVALUATIONS, Evaluation
from .ifeval import InstructionFollowing
from .multiple_choice import ClozeFormulation, GenerativeMCQ, LoglikelihoodMCQ, MultipleChoiceFormulation

__all__ = [
    "EVALUATIONS",
    "ClozeFormulation",
    "Evaluation",
    "GenerativeMCQ",
    "InstructionFollowing",
    "LoglikelihoodMCQ",
    "MultipleChoiceFormulation",
]
