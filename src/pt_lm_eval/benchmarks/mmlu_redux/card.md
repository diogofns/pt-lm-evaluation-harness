# MMLU-Redux 2.0 filtrado

## Fonte

Dataset: https://huggingface.co/datasets/fxmarty/mmlu-redux-2.0-ok
Revisão fixada: `8dd06e599d20d57e83b590d497c95869a2d18781`.

## Licença

CC-BY-4.0, segundo o cartão do dataset.

## Variedade do português

Não se aplica: benchmark original em inglês, sem tradução.

## Domínio

57 matérias do MMLU com revisão de erros.

## Construção

Nome mmlu_redux_generative. Usa a cópia filtrada de edinburgh-dawg/mmlu-redux-2.0 com error_type=ok; não é o dataset original edinburgh-dawg/mmlu-redux. 57 configs, split test, zero-shot. Cabeçalho por matéria, extração da primeira letra maiúscula [ABCD], exact_match,default, média micro. 256 tokens, stop </s>, temperatura zero.

## Contaminação

Dataset público de avaliação; deve ser considerado ao auditar contaminação dos modelos.

## Mantenedor

Dataset: fxmarty. Protocolo implementado no pt-lm-eval.
