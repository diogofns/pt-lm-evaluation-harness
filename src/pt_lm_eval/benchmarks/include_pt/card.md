# include_pt

## Fonte

- Dataset: [`CohereLabs/include-base-44`](https://huggingface.co/datasets/CohereLabs/include-base-44), configuração `Portuguese`, split `test`.
- Revisão fixada: `d2e1f6015f67a43c02a9a68db98e2298e2d6a660`.
- O INCLUDE contém 22.637 questões de múltipla escolha em 44 idiomas. A configuração portuguesa contém 551 itens nesta revisão.

## Licença

Apache-2.0, segundo o Hugging Face Hub.

## Variedade do português

Português. O dataset não separa as amostras em PT-BR e PT-PT.

## Domínio

Conhecimento e raciocínio acadêmico e profissional em 57 tópicos, incluindo conhecimento regional.

## Construção

Questões de provas acadêmicas e profissionais, com quatro alternativas. O gabarito `answer` é um índice 0-based.

## Contaminação

O split de teste é público. Considere o risco de exposição ao interpretar resultados.

## Mantenedor

Cohere Labs.

## Protocolo

Versão 2.0: define a avaliação de português do INCLUDE. O prompt concatena a descrição do
domínio diretamente ao enunciado, sem linha em branco; preserva as opções A–D,
inclusive o espaço antes de D no template, e termina em `Answer:`. A métrica é `acc`.

Few-shot usa `Random(seed)` por domínio, avançado na ordem dos documentos. A tarefa
padrão usa o split de teste como fallback e **não exclui explicitamente a
própria questão**; o port reproduz esse comportamento. Use zero-shot para evitar
esse possível vazamento. `limit` conta os primeiros documentos de cada domínio.

Os resultados agregam as seis subtarefas por domínio, ponderadas pelo tamanho,
com erro-padrão agrupado por subtarefa.

Versão 2.1: somente MCF, segundo o protocolo definido; geração complementar removida.

## Versão 2.2 — templates PT-BR

Cabeçalho, seis rótulos de domínio e marcador `Resposta:` em português brasileiro.
Uma linha em branco separa o cabeçalho do enunciado. Os identificadores internos dos
domínios, os gabaritos, a ordem e o whitespace das alternativas permanecem os mesmos.

## Versão 2.3 — geração complementar

Modelos instruct e APIs também podem usar `gen`: enunciado, alternativas e cabeçalho
PT-BR, com a instrução `Responda com a letra da alternativa correta no formato
"Resposta: X".`. A resposta é extraída pelo filtro PT-BR e reportada como `acc`,
`extracted` e `truncated`. MCF permanece disponível e seu prompt não muda.
A seleção de itens e exemplos few-shot é a mesma em MCF e geração, inclusive as
reservas e exclusões aplicáveis ao dataset. Para rodar somente geração, configure
`evaluations: [gen]`. Na geração instruct, o cliente aplica o chat template do modelo.
