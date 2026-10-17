from ..base import ColumnBenchmark, Columns, HFDataset


class EnergyRegulacao(ColumnBenchmark):
    name = "energy_regulacao"
    version = "1.0"
    description = "Regulação do setor elétrico brasileiro (PRORET/ANEEL), 5 alternativas."
    source = HFDataset(
        "cemig-ceia/energy-eval", split="train", revision="cac76ae9456fc3c9354d761880bc3c3a5b54d1e1"
    )
    columns = Columns(
        id="id", question="question", options="choices.text", answer="answerKey", answer_is="letter"
    )
