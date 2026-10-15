"""Tokenização de pares (contexto, continuação) para log-likelihood.

`encode_pair` é portado do lm-evaluation-harness (lm_eval/api/model.py,
_encode_pair e loglikelihood), Copyright (c) 2020 EleutherAI, licença MIT.
Diferença: aqui o BOS é adicionado quando o tokenizer o usa por padrão
(Llama, Gemma), em vez de nunca ser adicionado.
"""

from __future__ import annotations

from typing import Any


class PairTokenizer:
    def __init__(self, tokenizer: Any, add_bos: bool | None = None) -> None:
        self.tokenizer = tokenizer
        bos = tokenizer.bos_token_id
        if add_bos is None:  # "auto": segue o que o tokenizer faz por padrão
            default = tokenizer("a", add_special_tokens=True).input_ids
            add_bos = bos is not None and len(default) > 0 and default[0] == bos
        self.bos = [bos] if add_bos else []
        # Contexto vazio: condiciona no BOS, ou no EOS se não houver BOS (como o lm-eval).
        self.prefix_token_id = bos if bos is not None else tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        if context == "":
            cont = self.encode(continuation)
            if cont and cont[0] == self.prefix_token_id:
                return cont[:1], cont[1:]
            return [self.prefix_token_id], cont

        # Espaços no fim do contexto vão para a continuação: " B" tokeniza
        # como uma palavra, "B" depois de "Resposta: " não.
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole = self.encode(context + continuation)
        context_ids = self.encode(context)
        return self.bos + context_ids, whole[len(context_ids) :]


def truncate_left(
    context_ids: list[int], continuation_ids: list[int], max_length: int | None
) -> tuple[list[int], int]:
    """Concatena e corta pela esquerda; devolve (ids, posição onde começa a continuação)."""
    ids = context_ids + continuation_ids
    if max_length is None or len(ids) <= max_length:
        return ids, len(context_ids)
    if len(continuation_ids) >= max_length:
        raise ValueError(
            f"continuação com {len(continuation_ids)} tokens não cabe em max_length={max_length}"
        )
    cut = len(ids) - max_length
    return ids[cut:], len(context_ids) - cut
