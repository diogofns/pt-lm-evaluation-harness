# global_piqa_nonparallel_pt_br

## Fonte

- Dataset: [`mrlbenchmarks/global-piqa-nonparallel`](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel).
- Config: `por_latn_braz`; split `test`; revisão `6777742fa3634c0583cda3b7f8a482ea7b1b0937`.
- 100 itens, duas alternativas, nenhum descarte. Não inclui o conjunto paralelo nem PT-PT.
- Gabarito: `label`, índice 0-based. Distribuição: 40 respostas A e 60 respostas B;
  escolher sempre B acerta 60%, enquanto a escolha uniforme aleatória acerta 50%.
- Sem split separado de few-shot: se solicitado, usa os demais itens, excluindo a própria questão.
  A configuração de comparação usa zero-shot.

## Licença

CC-BY-SA-4.0, segundo o Hub. Os autores destinam o conjunto à avaliação e explicitamente
proíbem seu uso para treinar sistemas de IA ou como semente de dados sintéticos.

## Variedade do português

Português brasileiro (`por_latn_braz`).

## Domínio

Senso comum físico e cultural: objetos, ações, comida e situações cotidianas brasileiras.

## Construção

Questões locais do conjunto nonparallel, verificadas por falantes nativos. As colunas
em inglês e os metadados suplementares não entram nos prompts.

## Contaminação

Uso exclusivo para avaliação; não incorporar este conjunto ao corpus de CPT/KD.

## Mantenedor

`mrlbenchmarks`; [repositório Global PIQA](https://github.com/mrlbenchmarks/global-piqa).

## Protocolo local

MCF/CF usam os prompts comuns do harness; CF é normalizado por caracteres.
Reportar separadamente de Global PIQA parallel, que contém perguntas e número de opções diferentes.
