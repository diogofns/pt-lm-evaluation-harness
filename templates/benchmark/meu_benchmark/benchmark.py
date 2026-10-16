"""Modelo de benchmark. Troque `meu_benchmark` pelo nome do seu e ajuste os campos.

Escolha UMA das duas formas de converter linhas em itens:
- `columns(...)`, quando o dataset já tem pergunta, opções e resposta em colunas;
- uma função `to_item(row) -> MCQItem | None`, quando os dados precisam de tratamento.
"""

from pathlib import Path

from pt_lm_eval import BENCHMARKS, Benchmark, HFDataset, MCQItem, columns, letter_index  # noqa: F401

# Forma 1: colunas comuns. `options` pode ser um caminho para uma lista
# ("choices.text") ou uma lista de colunas (["A", "B", "C", "D"]). `answer_is` diz
# como ler o gabarito: "letter", "index" (0-based) ou "text" (o texto da opção correta).
to_item = columns(
    question="question",
    options="choices.text",
    answer="answer",
    answer_is="letter",
    subject="subject",  # opcional: gera resultados por matéria
)


# Forma 2: função. Devolva None para descartar uma linha inválida; o total de
# descartes aparece no `pt-lm-eval benchmarks check` e nos resultados.
#
# def to_item(row: dict) -> MCQItem | None:
#     if not row["enunciado"].strip():
#         return None
#     return MCQItem(
#         id=f"{row['_split']}-{row['_index']}",
#         question=row["enunciado"],
#         options=tuple(row["alternativas"]),
#         gold=letter_index(row["gabarito"]),
#         subject=row.get("area"),
#     )


BENCHMARKS.register(
    "meu_benchmark",
    Benchmark(
        version="1.0",  # aumente sempre que dados ou prompts mudarem
        description="Uma linha dizendo o que o benchmark mede.",
        source=HFDataset("organizacao/dataset", split="test", revision="<commit do dataset>"),
        to_item=to_item,
        evaluations=("mcf", "cf", "gen"),
        card=Path(__file__).with_name("README.md"),
    ),
)
