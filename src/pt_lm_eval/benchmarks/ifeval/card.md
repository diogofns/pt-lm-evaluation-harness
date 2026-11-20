# Google IFEval

## Fonte

Dataset: https://huggingface.co/datasets/google/IFEval
Revisão fixada: `966cd89545d6b6acfd7638bc708b98261ca58e84`.

## Licença

Apache-2.0; verificadores Google Apache-2.0 e utilitários de suporte MIT (veja NOTICE).

## Variedade do português

Não se aplica: benchmark original em inglês, sem tradução.

## Domínio

Seguimento de instruções em inglês com verificação programática.

## Construção

Nome ifeval, avaliação ifeval_en. Split train, zero-shot, 1280 tokens e temperatura zero, sem stops da tarefa. Verificadores em inglês incluídos no pacote, com imports relativos. prompt_level_strict_acc, inst_level_strict_acc, prompt_level_loose_acc e inst_level_loose_acc. As métricas por instrução pesam pelo número de instruções de cada prompt. Dependências no extra [ifeval]; NLTK punkt_tab. Portuguese-IFEval permanece no benchmark ifeval_pt.

## Contaminação

Dataset público de avaliação; deve ser considerado ao auditar contaminação dos modelos.

## Mantenedor

Dataset: google. Protocolo implementado no pt-lm-eval.
