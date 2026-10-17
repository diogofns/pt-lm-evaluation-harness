from ..base import ColumnBenchmark, Columns, HFDataset


class OABExams(ColumnBenchmark):
    name = "oab_exams"
    version = "1.0"
    description = "Exame da OAB, 2010–2018 (4 alternativas); matéria = área do direito, quando há."
    source = HFDataset(
        "eduagarcia/oab_exams", split="train", revision="fe810531d30e7799dd83c2e5be302ca2f2bdf930"
    )
    columns = Columns(
        id="id",
        question="question",
        options="choices.text",
        answer="answerKey",
        answer_is="letter",
        subject="question_type",
    )
