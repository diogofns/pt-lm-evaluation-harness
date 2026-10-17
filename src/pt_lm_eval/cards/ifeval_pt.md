# ifeval_pt

Seguimento de instruções verificáveis em português: cada prompt traz de uma a três
restrições checáveis por regra (número de palavras, palavras proibidas, JSON, título,
crase, mesóclise, os quatro porquês etc.). Pontuação strict e loose, por prompt e por
instrução, como no IFEval.

## Fonte

- Dataset: [`gap-ufg/Portuguese-IFEval`](https://huggingface.co/datasets/gap-ufg/Portuguese-IFEval), split `test`
- Revisão fixada: `0ff3fe8d82a5cdb43153f86b301b2c9d5c6ea098` (dados idênticos ao commit
  `a8d8b3c` de [ark-aiml-research/Portuguese-IFEval](https://github.com/ark-aiml-research/Portuguese-IFEval);
  SHA-256 `d2246309…c38708`)
- 535 prompts, 1.238 instruções, 31 tipos de instrução; nenhum item descartado.

## Licença

Apache-2.0 (dataset e código de referência). Os verificadores em
`ifeval.py` são portados do repositório de referência,
com o aviso de copyright original.

## Variedade do português

Português; o cartão do dataset não especifica a variedade. Os prompts foram
regionalizados (não traduzidos literalmente) e revisados por três pesquisadores
nativos.

## Domínio

Instruções gerais (redação, listas, cartas, postagens). Mede obediência a restrições
explícitas, não correção factual nem qualidade do texto.

## Construção

Adapta as categorias do IFEval/M-IFEval e acrescenta seis restrições do português:
frequência de cedilha, ausência de til, crase, mesóclise, tratamento por "vós" e os
quatro porquês. Prompts gerados e reescritos com `Qwen/Qwen3-235B-A22B-FP8`,
filtrados automaticamente e revisados à mão: de 595 candidatos, ficaram 535.

## Contaminação

Prompts de teste públicos e sintéticos; o cartão pede que não sejam usados em treino.
Não há verificação de contaminação conhecida.

## Mantenedor

Organização `gap-ufg` no Hugging Face; código de referência em `ark-aiml-research`.

## Observações

- Paridade com os verificadores de referência conferida sobre as respostas publicadas
  no repositório de referência (ver `tests/test_parity_ifeval_pt.py`).
- Diferença deliberada: o langdetect usa semente fixa (a referência não fixa), para
  que o veredito de idioma seja reproduzível.
- Geração com até 2048 tokens, como na referência; modelos com `generate.max_tokens`
  maior usam o maior.
