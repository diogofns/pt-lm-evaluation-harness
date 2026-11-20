# math_pt_br

## Fonte

- Dataset: [`tiagoteixeira03/MATH-PT`](https://huggingface.co/datasets/tiagoteixeira03/MATH-PT).
- Subset: **`ptbr_multiple_choice`**, split `test`.
- Revisão: `11c44f85c32d5d6b3dce76d163201ae4c6abd4b2`.
- 638 questões, todas com cinco alternativas A–E; nenhum descarte.
- Campos: `problem`, `choices`, `answer` (letra), `level` (nível de dificuldade).
- Níveis nesta revisão: 2 (118 itens), 3 (140), 4 (270) e 5 (110).
- Não inclui os subsets PT-PT nem os de resposta aberta.

## Licença

MIT, conforme o cartão do dataset no Hugging Face.

## Variedade do português

Português brasileiro.

## Domínio

Raciocínio matemático em problemas de provas e competições brasileiras. O cartão
da coleção cita OBMEP, OMIF, Competição Elon Lages Lima e vestibular do ITA.

## Construção

Segundo os autores, a coleção reúne problemas originalmente escritos em português,
coletados de provas e competições. Este benchmark usa exclusivamente as questões
brasileiras de múltipla escolha. O texto, o LaTeX e a ordem A–E são preservados.
Os IDs locais são `test-<índice>` na ordem publicada.

## Contaminação

Questões de provas públicas e split de teste disponível no Hub têm risco de exposição
em corpora de treinamento. Não se deve usar os itens avaliados como demonstrações.

## Mantenedor

Tiago Teixeira (`tiagoteixeira03`) e coautores da coleção MATH-PT, publicada no PROPOR 2026.

## Protocolo

- MCF para modelos `base` e `instruct`: enunciado, opções A–E e `Resposta:`, seguindo
  o mesmo formato e pontuação do Global MMLU PT. Na modalidade MCF, a resposta é pontuada diretamente por likelihood.
- A letra de maior log-probabilidade determina a predição. A métrica é `acc`;
  `random_baseline` é 0,2. O nível de dificuldade é usado para os grupos de relatório.
- Zero-shot por padrão. Com `num_fewshot=N`, os primeiros N exemplos de cada nível
  são reservados como demonstrações fixas e excluídos do eval, inclusive cópias das
  mesmas questões. A reserva precede `limit` e respeita exclusões explícitas.
- `limit` conta os primeiros itens restantes de cada nível. A média principal é
  ponderada pelo número de questões, com erro-padrão agrupado como no Global MMLU PT.
- Os IDs das demonstrações ficam registrados no resultado. Os arquivos incluem `@1.2`.

## Qualidade dos gabaritos

Os gabaritos publicados são preservados, sem correção automática. Há ao menos uma
inconsistência identificada: na primeira questão, a mesa mede 8 × 22 = 176 cm e
corresponde a 11 palmos de Carolina, resultando em 16 cm (opção A), mas `answer` é D.
Essa observação deve ser considerada na interpretação dos escores.

## Versões

- 1.1: marcador `Resposta:` em PT-BR; demais regras preservadas.
- 1.0: protocolo inicial com marcador `Answer:`.

## Versão 1.2 — geração complementar

Além de MCF, modelos instruct e APIs podem usar `gen` com instrução de resposta no
formato `Resposta: X` em PT-BR. Os itens, níveis, gabaritos e demonstrações reservadas
são os mesmos do MCF. A geração reporta `acc`, `extracted` e `truncated`. Com
`--model-type instruct`, o CLI usa geração com o chat template do modelo por padrão.
