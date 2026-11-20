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

- O subset brasileiro é avaliado integralmente, com agrupamento por assunto.
- MCF usa enunciado, alternativas e `Resposta:` e reporta `acc` e `acc_norm`; para as opções de uma letra,
  ambos coincidem. Os exemplos few-shot são os primeiros itens do mesmo assunto no split `test`,
  excluindo a questão avaliada.

## Versões

- 2.1: somente MCF, segundo o protocolo definido; geração complementar removida.

- 2.0: preserva whitespace das alternativas, exclui documentos iguais ao avaliado no
  few-shot canônico e aplica `limit` por assunto. Agrega erros-padrão por subtarefas
  por assunto. Nesta versão a geração ainda era complementar.

- 1.2: prompt MCF com `Answer:`; few-shot canônico por assunto.
- 1.1: gabarito lido sempre como letra (`answer_is="letter"`). Na 1.0, 2 itens em que a letra do
  gabarito coincidia com o texto de outra alternativa ficaram com o gabarito errado.

## Versão 2.2 — templates PT-BR

O cabeçalho e o marcador `Resposta:` estão em português brasileiro. Os 57 nomes de
matérias são traduzidos somente na apresentação; `subject`, grupos e IDs dos dados
são preservados. O cabeçalho termina com uma linha em branco antes da questão.

## Versão 2.3 — geração complementar

Modelos instruct e APIs também podem usar `gen`: enunciado, alternativas e cabeçalho
PT-BR, com a instrução `Responda com a letra da alternativa correta no formato
"Resposta: X".`. A resposta é extraída pelo filtro PT-BR e reportada como `acc`,
`extracted` e `truncated`. MCF permanece disponível e seu prompt não muda.
A seleção de itens e exemplos few-shot é a mesma em MCF e geração, inclusive as
reservas e exclusões aplicáveis ao dataset. Com `--model-type instruct`, o CLI usa
geração e o cliente aplica o chat template do modelo por padrão.
