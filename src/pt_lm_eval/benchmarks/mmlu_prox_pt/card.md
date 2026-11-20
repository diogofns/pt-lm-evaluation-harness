# mmlu_prox_pt

## Fonte

- Dataset: [`li-lab/MMLU-ProX`](https://huggingface.co/datasets/li-lab/MMLU-ProX), config `pt`.
- Revisão: `8e6106a6c6ce1c5027e66cc338143cf997b2aa09`.
- Teste completo: 11.759 questões, 14 matérias, 3–10 alternativas; nenhum item descartado.
- Few-shot: 70 questões do split `validation`, separado do teste.
- Gabarito: campo `answer`, que contém a letra correta.
  O item 3787 tem `answer=C` e `answer_index=1`; usamos a letra C, sem corrigir os dados.
- As opções nulas são padding final e são omitidas. O texto literal `N/A` dos exemplos
  de validação é preservado, como na referência. `cot_content` não entra nos prompts.

## Licença

MIT, conforme o dataset card.

## Variedade do português

Português (`pt`); variedade regional não especificada pelo dataset card.

## Domínio

Conhecimento acadêmico e raciocínio em 14 áreas, incluindo matemática, ciências e humanas.

## Construção

Tradução multilíngue do MMLU-Pro, com tradução por modelos e validação por especialistas.
Usamos a versão completa, não a Lite.

## Contaminação

O benchmark deriva de MMLU-Pro e de questões acadêmicas públicas. Estes itens devem
permanecer separados dos dados de treino dos experimentos avaliados.

## Mantenedor

`li-lab`, projeto [MMLU-ProX](https://mmluprox.github.io/).

## Protocolo local

MCF/CF usam os prompts comuns do harness. CF é normalizado por caracteres.
A referência publicada usa geração com exemplos de raciocínio; nossos resultados
MCF/CF não reproduzem esse protocolo. A chance aleatória depende do número de alternativas.
