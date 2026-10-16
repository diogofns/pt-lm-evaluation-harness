# global_mmlu_pt

MMLU (57 matérias, 4 alternativas) em português, da coleção Global-MMLU da Cohere.

## Fonte

- Dataset: [`CohereLabs/Global-MMLU`](https://huggingface.co/datasets/CohereLabs/Global-MMLU) (antigo `CohereForAI`), config `pt`
- Revisão fixada: `0e619dbeb34206cd48705a1a0ea7fb21cae09993`
- 14.042 itens no split `test`; exemplos few-shot do split `dev` (285 itens), como no lighteval.
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

- O `mmmlu_pt_br` é outra tradução do mesmo MMLU (OpenAI); os dois medem o mesmo
  conteúdo com traduções diferentes.
