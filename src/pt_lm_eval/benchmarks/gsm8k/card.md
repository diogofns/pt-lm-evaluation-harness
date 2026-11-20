# GSM8K

## Fonte

Dataset: https://huggingface.co/datasets/openai/gsm8k
Revisão fixada: `740312add88f781978c0658806c59bc2815b9866`.

## Licença

MIT, segundo o cartão do dataset.

## Variedade do português

Não se aplica: benchmark original em inglês, sem tradução.

## Domínio

Problemas de matemática escolar com respostas numéricas.

## Construção

Config main; test para avaliação e train para 5 exemplos completos (incluindo as soluções originais). Essa é a tarefa gsm8k, não gsm8k_cot. O sampler segue random.Random(seed), padrão 1234, ao longo de todos os itens test na ordem original. Extrações strict-match e flexible-extract e exact_match; limite de geração: 256 tokens. Stops: Question:, </s> e <|im_end|>. Temperatura zero.

## Contaminação

Dataset público de avaliação; deve ser considerado ao auditar contaminação dos modelos.

## Mantenedor

Dataset: openai. Protocolo implementado no pt-lm-eval.
