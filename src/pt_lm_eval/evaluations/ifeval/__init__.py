from .evaluation import InstructionFollowing
from .instructions import INSTRUCTIONS, Instruction, follows_loose, follows_strict, instruction

__all__ = [
    "INSTRUCTIONS",
    "Instruction",
    "InstructionFollowing",
    "follows_loose",
    "follows_strict",
    "instruction",
]
