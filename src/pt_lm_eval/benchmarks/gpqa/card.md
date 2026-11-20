# GPQA

## Fonte

Dataset: https://huggingface.co/datasets/Idavidrein/gpqa
Revisão fixada: `83022cefff930aea54f654c0b282e74b9eeda5c6`.

## Licença

cc-by-4.0; dataset gated, exige aceite e acesso no Hugging Face.

## Variedade do português

Não se aplica: benchmark original em inglês, sem tradução.

## Domínio

Questões científicas de pós-graduação em biologia, física e química.

## Construção

Main, diamond e extended, todos do split train (usado para avaliação). Alternativas pré-processadas e embaralhadas em ordem de dataset com random.Random(0). Pontuações das continuações " (A)" a " (D)"; acc e acc_norm. Zero-shot fixo. Nomes: gpqa_main_zeroshot, gpqa_diamond_zeroshot, gpqa_extended_zeroshot. A seed 0 usa um gerador privado, independente da ordem dos modelos e benchmarks.

## Contaminação

Dataset público de avaliação; deve ser considerado ao auditar contaminação dos modelos.

## Mantenedor

Dataset: Idavidrein. Protocolo implementado no pt-lm-eval.
