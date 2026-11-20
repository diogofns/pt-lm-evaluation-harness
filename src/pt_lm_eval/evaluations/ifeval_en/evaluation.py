from __future__ import annotations

from collections.abc import Sequence

from ...core import GenerateRequest, GenerateResponse, InstructionItem, Scored
from ..base import Evaluation


class EnglishInstructionFollowing(Evaluation):
    supports_base_generation = True
    supports_fewshot = False
    name = "ifeval_en"
    item_type = InstructionItem
    needs = frozenset({"generate"})
    scoring_version = "ifeval-en-1"

    def build(self, item: InstructionItem, shots: Sequence[InstructionItem], multiturn=False):
        if shots:
            raise ValueError("IFEval exige num_fewshot=0")
        return [
            GenerateRequest(
                prompt=item.prompt, max_tokens=1280, params={"temperature": 0}, enforce_generation_config=True
            )
        ]

    def score(self, item: InstructionItem, responses: Sequence[GenerateResponse]) -> Scored:
        from .vendor.utils import process_results

        response = responses[0]
        doc = {
            "key": item.id,
            "prompt": item.prompt,
            "instruction_id_list": list(item.instruction_ids),
            "kwargs": list(item.kwargs),
        }
        # Reuse the exact strict/loose reference implementation, including its kwargs handling.
        values = process_results(doc, [response.text])
        n = len(item.instruction_ids)
        return Scored(
            metrics={k: sum(v) / n if isinstance(v, list) else float(v) for k, v in values.items()},
            weights={"inst_level_strict_acc": n, "inst_level_loose_acc": n},
            details={
                "instructions": list(item.instruction_ids),
                "strict": values["inst_level_strict_acc"],
                "loose": values["inst_level_loose_acc"],
                **response.details(),
            },
        )
