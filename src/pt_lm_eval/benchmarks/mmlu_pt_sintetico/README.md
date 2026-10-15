# mmlu_pt_sintetico

Questões no estilo MMLU (4 alternativas, 61 matérias) em português, geradas por modelo.

## Fonte

- Dataset: [`cemig-ceia/benchmark_Qwen3-235B-A22B_MMLU_v2`](https://huggingface.co/datasets/cemig-ceia/benchmark_Qwen3-235B-A22B_MMLU_v2), split `train`
- Revisão fixada: `831ba6bbae5b6057fbc3cae40e13eae7b55ebb26`
- 11.791 itens, nenhum descartado.

## Licença

Não declarada no Hugging Face Hub. **A definir pelo mantenedor.** Por ser gerado por
modelo, vale conferir também os termos de uso do modelo gerador.

## Variedade do português

Português do Brasil.

## Domínio

Conhecimento geral nas matérias do MMLU (campo `subject`), com níveis em `level`.

## Construção

Sintético: gerado por Qwen3-235B-A22B, segundo o nome do dataset. O processo
(prompts, filtros, revisão humana) não está documentado. **A preencher pelo mantenedor.**

## Contaminação

Questões novas, mas geradas por um modelo da família Qwen3: modelos dessa família
podem ter vantagem por estilo. Leve isso em conta ao comparar famílias.

## Mantenedor

Organização `cemig-ceia` no Hugging Face. **Contato a definir.**

## Observações

- **Gabarito desbalanceado**: B 39,3%, C 35,4%, A 16,2%, D 9,1%. Chutar sempre "B"
  acerta 39,3%, bem acima dos 25% esperados. Compare resultados com essa referência,
  não com 25%. Recomenda-se rebalancear (embaralhar as alternativas) numa versão 2.0.
