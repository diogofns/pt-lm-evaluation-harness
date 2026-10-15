# mmmlu_pt_br

O MMLU original (57 matérias, 4 alternativas) traduzido para o português do Brasil
por tradutores profissionais contratados pela OpenAI.

## Fonte

- Dataset: [`openai/MMMLU`](https://huggingface.co/datasets/openai/MMMLU), subset `PT_BR`, split `test`
- Revisão fixada: `325a01dc3e173cac1578df94120499aaca2e2504`
- 14.042 itens, nenhum descartado.

## Licença

MIT (declarada no Hub).

## Variedade do português

Português do Brasil.

## Domínio

Conhecimento geral e acadêmico (matemática, direito, medicina, história etc.). O
conteúdo é o do MMLU, com contexto majoritariamente estadunidense (ex.: história e
política dos EUA), apenas traduzido.

## Construção

Tradução humana profissional do conjunto de teste do MMLU (Hendrycks et al., 2021).

## Contaminação

Alta exposição: o MMLU original é público desde 2020 e amplamente usado; a tradução
reduz, mas não elimina, o risco de memorização.

## Mantenedor

OpenAI (dataset original). Neste repositório: equipe do pt-lm-eval.

## Observações

- É o mesmo conjunto da tarefa `mmmlu_pt_br` do lm-evaluation-harness, usado no teste
  de paridade (`tests/test_parity_lm_eval.py`).
