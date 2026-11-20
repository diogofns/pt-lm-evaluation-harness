# global_mmlu_pt

MMLU (57 matérias, 4 alternativas) em português, da coleção Global-MMLU da Cohere.

## Fonte

- Dataset: [`CohereLabs/Global-MMLU`](https://huggingface.co/datasets/CohereLabs/Global-MMLU) (antigo `CohereForAI`), config `pt`
- Revisão fixada: `0e619dbeb34206cd48705a1a0ea7fb21cae09993`
- 14.042 itens no split `test`; exemplos few-shot do split `dev` (285 itens).
- Matéria = `subject`. Nenhuma alternativa vazia nesta revisão.

## Licença

Apache-2.0, segundo o Hub.

## Variedade do português

Português; o cartão do dataset não especifica a variedade.

## Domínio

Conhecimento geral e especializado (STEM, humanas, ciências sociais, outros), como o MMLU.

## Construção

Tradução do MMLU com revisão humana parcial (`is_annotated`) e marcação de questões
culturalmente sensíveis (`cultural_sensitivity_label`); aqui são usadas todas as questões.

## Contaminação

Tradução de um benchmark muito difundido: o conteúdo original (em inglês) tem risco alto
de contaminação.

## Mantenedor

Organização `CohereLabs` no Hugging Face.

## Observações

- Versão 2.0: define o protocolo de múltipla escolha (conjunto completo, não Lite).
  MCF usa enunciado, opções A–D com o texto original e `Answer:`; pontua as letras
  e reporta `acc`. Few-shot usa os primeiros exemplos do split `dev` do mesmo assunto.
- `limit` conta os primeiros documentos de cada assunto. A média principal é ponderada
  pelo número de questões e o erro-padrão agrupa as variâncias das subtarefas por caracteres.
- Versão 2.1: somente MCF, segundo o protocolo definido; CF e geração complementares removidos.

- O `mmmlu_pt_br` é outra tradução do mesmo MMLU (OpenAI); os dois medem o mesmo
  conteúdo com traduções diferentes.

## Versão 2.2 — templates PT-BR

O prompt termina em `Resposta:`. Enunciados, opções, gabaritos e a pontuação MCF
permanecem os mesmos. Resultados das versões anteriores preservam seus arquivos.

## Versão 2.3 — geração complementar

Modelos instruct e APIs também podem usar `gen`: enunciado, alternativas e cabeçalho
PT-BR, com a instrução `Responda com a letra da alternativa correta no formato
"Resposta: X".`. A resposta é extraída pelo filtro PT-BR e reportada como `acc`,
`extracted` e `truncated`. MCF permanece disponível e seu prompt não muda.
A seleção de itens e exemplos few-shot é a mesma em MCF e geração, inclusive as
reservas e exclusões aplicáveis ao dataset. Para rodar somente geração, configure
`evaluations: [gen]`. Na geração instruct, o cliente aplica o chat template do modelo.
