# global_piqa_nonparallel_pt_br

## Fonte

- Dataset: [`mrlbenchmarks/global-piqa-nonparallel`](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel).
- Config: `por_latn_braz`; split `test`; revisão `6777742fa3634c0583cda3b7f8a482ea7b1b0937`.
- 100 itens, duas alternativas, nenhum descarte. Não inclui o conjunto paralelo nem PT-PT.
- Gabarito: `label`, índice 0-based. Distribuição: 40 respostas A e 60 respostas B;
  escolher sempre B acerta 60%, enquanto a escolha uniforme aleatória acerta 50%.
- Sem split separado de few-shot: a tarefa usa o teste como fallback, sem
  excluir explicitamente a própria questão. O port reproduz esse comportamento.
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

## Protocolo

- Cloze: o contexto é `prompt`, as continuações são as soluções precedidas por espaço. Reporta `acc`, `acc_norm` (normalização por caracteres) e `acc_bytes`.
- Geração: usa a instrução e o regex definidos no protocolo, selecionando a última
  correspondência válida. Reporta `exact_match`, com `acc` como alias complementar;
  temperatura 0,8, `top_p=0.95`, limite obrigatório de 2048 tokens. Disponível também
  para modelos base. Few-shot usa `Random(seed)` na ordem dos documentos.
- Versão 2.0: preserva whitespace do contexto e das soluções, inclusive nos exemplos
  few-shot; `acc_norm` divide pelo número de caracteres do texto original da solução,
  sem contar o espaço delimitador. `limit` toma os primeiros documentos.
- Reportar separadamente de Global PIQA parallel, que contém perguntas e número de opções diferentes.

## Versão 2.1 — geração em PT-BR

O template de geração usa `Situação`, `Alternativa A/B` e solicita `Resposta: X`,
com X igual a A ou B. Os exemplos few-shot seguem o mesmo formato. O extrator comum
PT-BR usa a última resposta no formato, com normalização de caixa e Markdown; também
aceita letra isolada ou seguida de marcador. Respostas no antigo formato em inglês
não são pontuadas pelo protocolo novo. CF, dados e regras de normalização permanecem
iguais. Os padrões de amostragem e o limite da tarefa continuam os mesmos.
