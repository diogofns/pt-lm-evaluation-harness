# mmlu_pt (4 variantes)

Questões reais de provas brasileiras (concursos, vestibulares, olimpíadas), de 3 a 6
alternativas, organizadas por área. Quatro benchmarks:

| Nome | Dataset | Itens | Descartados |
|---|---|---:|---:|
| `mmlu_pt_undergraduate` | `mmlu-pt/mmlu-pt-undergraduate-full` | 19.319 | 28 |
| `mmlu_pt_undergraduate_hard` | `mmlu-pt/mmlu-pt-undergraduate-hard` | 11.184 | 28 |
| `mmlu_pt_high_school` | `mmlu-pt/mmlu-pt-high-school-full` | 7.187 | 46 |
| `mmlu_pt_high_school_hard` | `mmlu-pt/mmlu-pt-high-school-hard` | 2.029 | 13 |

## Fonte

Datasets da organização [`mmlu-pt`](https://huggingface.co/mmlu-pt) no Hugging Face;
revisões fixadas em `benchmark.py`. Cada split é uma área (ex.: `law`, `stem`) e vira
a matéria do item. As colunas `exam`, `exam_edition` e `exam_url` identificam a prova
de origem de cada questão.

## Licença

Não declarada no Hugging Face Hub. **A definir pelo mantenedor.** As questões vêm de
provas públicas; vale verificar os termos de reprodução das bancas.

## Variedade do português

Português do Brasil.

## Domínio

Graduação (contabilidade, direito, economia, saúde, exatas etc.) e ensino médio,
com versões "hard" filtradas por dificuldade (`difficulty_level`, `difficulty_score`).

## Construção

Questões de provas reais coletadas de sites públicos de provas (ex.: pciconcursos),
com alternativas e gabarito estruturados. O processo de coleta e limpeza não está
documentado nos datasets. **A preencher pelo mantenedor.**

## Métricas

Desde a versão 1.1, modelos locais `base` e `instruct` são avaliados somente por
`mcf` (`acc`): compara-se o log-likelihood das letras das alternativas. A geração
complementar foi removida das quatro variantes.

Desde a versão **2.0**, as quatro variantes reutilizam o protocolo do Global MMLU PT:
enunciado, alternativas preservadas e `Answer:`, sem o prefixo `Pergunta:`. A maior
log-probabilidade de uma letra determina a resposta; reporta-se `acc`. O mesmo método
vale para modelos base e instruct, preservando as 3–6 alternativas de cada questão.

Cada área/split é uma subtarefa: `limit` toma os primeiros documentos por área e
a acurácia global é ponderada pelo número de questões. O erro-padrão usa o mesmo
agrupamento de variâncias do Global MMLU PT. Os arquivos incluem `@<versão>` no nome,
preservando os resultados anteriores.

O padrão é zero-shot. Desde a versão **2.1**, `num_fewshot=N` reserva os primeiros
N exemplos válidos de cada área como um conjunto **fixo** de demonstrações. Esses
exemplos e cópias da mesma questão são removidos do eval de todas as questões,
antes de aplicar `limit`. Exclusões explícitas (`exclude_ids`) são aplicadas antes
da reserva. Cada área precisa ter questões restantes; caso contrário, há um erro.
O relatório registra os IDs reservados e `n_evaluated` conta apenas os itens pontuados.

Essa é a adaptação dos dados: Global MMLU tem `dev` separado; MMLU-PT não tem esse
split. Com zero-shot, nenhum item é reservado e o conjunto continua completo.

## Contaminação

Risco alto: as provas estão publicadas na web há anos, com gabarito, e provavelmente
aparecem em corpora de pré-treino.

## Mantenedor

Organização `mmlu-pt` no Hugging Face. **Contato a definir.**

## Observações

- **Itens descartados**: ~0,5% dos itens têm alternativas vazias, quase sempre todas
  (provavelmente questões com imagem). O light-benchmark avaliava esses itens mesmo
  assim; aqui eles são descartados e contados em `n_discarded`.
- O número de alternativas varia (3 a 6) dentro do mesmo benchmark; a referência de
  chute fica entre 22,8% e 24,3%.

## Versão 2.2 — templates PT-BR

As quatro variantes usam `Resposta:` no lugar de `Answer:`. Todas as regras de
pontuação, agrupamento, reserva de few-shot e exclusão de questões são preservadas.

## Versão 2.3 — geração complementar

Modelos instruct e APIs também podem usar `gen`: enunciado, alternativas e cabeçalho
PT-BR, com a instrução `Responda com a letra da alternativa correta no formato
"Resposta: X".`. A resposta é extraída pelo filtro PT-BR e reportada como `acc`,
`extracted` e `truncated`. MCF permanece disponível e seu prompt não muda.
A seleção de itens e exemplos few-shot é a mesma em MCF e geração, inclusive as
reservas e exclusões aplicáveis ao dataset. Com `--model-type instruct`, o CLI usa
geração e o cliente aplica o chat template do modelo por padrão.
