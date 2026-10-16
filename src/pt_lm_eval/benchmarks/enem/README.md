# enem

Questões do Exame Nacional do Ensino Médio de 2022, 2023 e 2024, todas as áreas
(linguagens, humanas, natureza, matemática), com 5 alternativas.

## Fonte

- Dataset: [`maritaca-ai/enem`](https://huggingface.co/datasets/maritaca-ai/enem), configs `2022`, `2023` e `2024`, split `train`
- Revisão fixada: `b2a7257356b0f56b740d57f23f8a575542ff6c0c`
- 540 questões; as 3 com gabarito "Anulado" são descartadas (537 itens). Matéria = ano.
- Conversão igual à do lighteval (`enem_adapter`): a descrição textual de cada figura
  substitui o `[[placeholder]]` correspondente no enunciado (145 questões têm figura).

## Licença

Apache-2.0, segundo o Hub.

## Variedade do português

Português do Brasil.

## Domínio

Conhecimentos do ensino médio e interpretação de texto; parte das questões depende de
figuras, aqui substituídas pela descrição em texto.

## Construção

Questões oficiais das provas do INEP, transcritas pela Maritaca AI, com descrições das
figuras.

## Contaminação

Provas públicas e muito divulgadas, com gabaritos e resoluções na web: risco alto de
estarem no pré-treino dos modelos. A prova de 2024 é a menos exposta.

## Mantenedor

Organização `maritaca-ai` no Hugging Face.
