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
