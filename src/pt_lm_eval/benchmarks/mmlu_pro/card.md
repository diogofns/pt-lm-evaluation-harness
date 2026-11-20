# MMLU-Pro

## Fonte

Dataset: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
Revisão fixada: `b189ec765aa7ed75c8acfea42df31fdae71f97be`.

## Licença

MIT, segundo o cartão do dataset.

## Variedade do português

Não se aplica: benchmark original em inglês, sem tradução.

## Domínio

Conhecimento e raciocínio em 14 matérias, até 10 alternativas.

## Construção

Grupo mmlu_pro (tarefa generativa, não leaderboard_mmlu_pro). Split test, 5 primeiros exemplos do validation de cada category, com as soluções do campo cot_content. Prompt Answer: Let's think step by step.; extração answer is \(?([ABCDEFGHIJ])\)?. Métrica exact_match,custom-extract; média micro ponderada pelo tamanho das matérias. 2048 tokens, temperatura zero, stop Question:.

## Contaminação

Dataset público de avaliação; deve ser considerado ao auditar contaminação dos modelos.

## Mantenedor

Dataset: TIGER-Lab. Protocolo implementado no pt-lm-eval.
