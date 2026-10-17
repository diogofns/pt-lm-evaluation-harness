"""As instruções verificáveis do Portuguese-IFEval, uma classe por instrução.

Portado de ark-aiml-research/Portuguese-IFEval (instructions/pt_instructions.py,
instruction_utils/pt_instructions_util.py e evaluation_main.py, commit a8d8b3c), que
deriva do IFEval da Google Research e do M-IFEval. Licença Apache-2.0, Copyright 2024
The Google Research Authors. Como na referência, cada instrução é uma classe que recebe
os parâmetros e verifica uma resposta; a semântica de cada uma é a da referência,
inclusive as esquisitices (palavras-chave como regex cru, nltk em inglês em
CapitalWordFrequency), para que o mesmo texto receba o mesmo veredito
(tests/test_parity_ifeval_pt.py). Diferença deliberada: o langdetect tem semente fixa.

Dependências (extra `ifeval`): spaCy com `pt_core_news_sm`, langdetect e nltk.
"""

from __future__ import annotations

import collections
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any, ClassVar

LESS_THAN, AT_LEAST = "menos que", "ao menos"
RELATIONS = (LESS_THAN, AT_LEAST)


class Instruction(ABC):
    """Uma instrução verificável. Subclasses declaram os ids que atendem (sem o prefixo "pt:")."""

    ids: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for name in cls.ids:
            INSTRUCTIONS[f"pt:{name}"] = cls

    def __init__(self, kwargs: Mapping[str, Any] | None = None, prompt: str = "") -> None:
        self.kwargs = dict(kwargs or {})
        self.prompt = prompt

    @abstractmethod
    def check(self, response: str) -> bool: ...

    def follows_strict(self, response: str) -> bool:
        return bool(response.strip()) and self.check(response)

    def follows_loose(self, response: str) -> bool:
        """Aceita se alguma variante passar: sem a primeira e/ou a última linha, sem asteriscos."""
        return any(r.strip() and self.check(r) for r in loose_variants(response))


INSTRUCTIONS: dict[str, type[Instruction]] = {}


def instruction(
    instruction_id: str, kwargs: Mapping[str, Any] | None = None, prompt: str = ""
) -> Instruction:
    return INSTRUCTIONS[instruction_id](kwargs, prompt)


def follows_strict(response: str, instruction_id: str, kwargs: Mapping[str, Any] | None, prompt: str) -> bool:
    return instruction(instruction_id, kwargs, prompt).follows_strict(response)


def follows_loose(response: str, instruction_id: str, kwargs: Mapping[str, Any] | None, prompt: str) -> bool:
    return instruction(instruction_id, kwargs, prompt).follows_loose(response)


def loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    # A referência calcula também a variante sem aspas, mas não a usa.
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


# --- utilitários de língua ----------------------------------------------------------


@cache
def _nlp() -> Any:
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("o IFEval PT requer o extra: uv pip install -e '.[ifeval]'") from exc
    nlp = spacy.load("pt_core_news_sm", disable=["tagger", "parser", "ner"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def count_words(text: str) -> int:
    return sum(1 for token in _nlp()(text) if not token.is_punct)


def count_sentences(text: str) -> int:
    return len(list(_nlp()(text).sents))


@cache
def _langdetect() -> Any:
    import langdetect

    langdetect.DetectorFactory.seed = 0
    return langdetect


def language_is(text: str, language: str) -> bool:
    langdetect = _langdetect()
    try:
        return langdetect.detect(text) == language
    except langdetect.LangDetectException:  # a referência conta como seguida
        return True


@cache
def _word_tokenize() -> Callable[[str], list[str]]:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return nltk.word_tokenize


def compare(actual: int, relation: str, threshold: int) -> bool:
    if relation not in RELATIONS:
        raise ValueError(f"relação deve ser uma de {RELATIONS}, veio {relation!r}")
    return actual < threshold if relation == LESS_THAN else actual >= threshold


# --- idioma e caixa -----------------------------------------------------------------


class ResponseLanguage(Instruction):
    ids = ("language:response_language",)

    def check(self, response: str) -> bool:
        return language_is(response, self.kwargs["language"])


class AllCapital(Instruction):
    ids = ("change_case:english_capital", "change_case:portuguese_capital")

    def check(self, response: str) -> bool:
        return response.isupper() and language_is(response, "pt")


class AllLowercase(Instruction):
    ids = ("change_case:english_lowercase", "change_case:portuguese_lowercase")

    def check(self, response: str) -> bool:
        return response.islower() and language_is(response, "pt")


class CapitalWordFrequency(Instruction):
    ids = ("change_case:capital_word_frequency",)

    def check(self, response: str) -> bool:
        capital = sum(1 for word in _word_tokenize()(response) if word.isupper())
        return compare(capital, self.kwargs["capital_relation"], self.kwargs["capital_frequency"])


# --- tamanho ------------------------------------------------------------------------


class NumberOfSentences(Instruction):
    ids = ("length_constraints:number_sentences",)

    def check(self, response: str) -> bool:
        return compare(count_sentences(response), self.kwargs["relation"], self.kwargs["num_sentences"])


class NumberOfWords(Instruction):
    ids = ("length_constraints:number_words",)

    def check(self, response: str) -> bool:
        return compare(count_words(response), self.kwargs["relation"], self.kwargs["num_words"])


class NumberOfParagraphs(Instruction):
    ids = ("length_constraints:number_paragraphs",)

    def check(self, response: str) -> bool:
        paragraphs = re.split(r"\s?\*\*\*\s?", response)
        count = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index in (0, len(paragraphs) - 1):
                    count -= 1
                else:
                    return False
        return count == self.kwargs["num_paragraphs"]


class NthParagraphFirstWord(Instruction):
    ids = ("length_constraints:nth_paragraph_first_word",)

    def check(self, response: str) -> bool:
        paragraphs = re.split(r"\n\n", response)
        count = sum(1 for p in paragraphs if p.strip())
        nth = self.kwargs["nth_paragraph"]
        if nth > count:
            return False
        paragraph = paragraphs[nth - 1].strip()
        if not paragraph:
            return False
        word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
        first_word = ""
        for letter in word:
            if letter in {".", ",", "?", "!", "'", '"'}:
                break
            first_word += letter.lower()
        return count == self.kwargs["num_paragraphs"] and first_word == self.kwargs["first_word"].lower()


# --- palavras e letras --------------------------------------------------------------


class KeywordsExistence(Instruction):
    ids = ("keywords:existence",)

    def check(self, response: str) -> bool:
        return all(re.search(k, response, flags=re.IGNORECASE) for k in sorted(self.kwargs["keywords"]))


class KeywordFrequency(Instruction):
    ids = ("keywords:frequency",)

    def check(self, response: str) -> bool:
        occurrences = len(re.findall(self.kwargs["keyword"].strip(), response, flags=re.IGNORECASE))
        return compare(occurrences, self.kwargs["relation"], self.kwargs["frequency"])


class ForbiddenWords(Instruction):
    ids = ("keywords:forbidden_words",)

    def check(self, response: str) -> bool:
        words = sorted(set(self.kwargs["forbidden_words"]))
        return not any(re.search(r"\b" + w + r"\b", response, flags=re.IGNORECASE) for w in words)


class LetterFrequency(Instruction):
    ids = ("keywords:letter_frequency",)

    def check(self, response: str) -> bool:
        letter = self.kwargs["letter"].strip().lower()
        count = collections.Counter(response.lower())[letter]
        return compare(count, self.kwargs["let_relation"], self.kwargs["let_frequency"])


class CedilhaFrequency(Instruction):
    ids = ("keywords:cedilha_frequency",)

    def check(self, response: str) -> bool:
        return response.lower().count("ç") == self.kwargs["count"]


# --- pontuação ----------------------------------------------------------------------


class NoComma(Instruction):
    ids = ("punctuation:no_comma",)

    def check(self, response: str) -> bool:
        return "," not in response


class NoTilde(Instruction):
    ids = ("punctuation:no_tilde",)

    def check(self, response: str) -> bool:
        return re.search(r"[ãõñÃÕÑ]", response) is None


class CrasePresence(Instruction):
    ids = ("punctuation:crase_presence",)

    def check(self, response: str) -> bool:
        return (
            re.search(r"\b(?:à|às|àquele|àquela|àqueles|àquelas|àquilo)\b", response, re.IGNORECASE)
            is not None
        )


# --- formato ------------------------------------------------------------------------


class Placeholders(Instruction):
    ids = ("detectable_content:number_placeholders",)

    def check(self, response: str) -> bool:
        return len(re.findall(r"\[.*?\]", response)) >= self.kwargs["num_placeholders"]


class BulletLists(Instruction):
    ids = ("detectable_format:number_bullet_lists",)

    def check(self, response: str) -> bool:
        stars = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
        dashes = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
        return len(stars) + len(dashes) == self.kwargs["num_bullets"]


class ConstrainedResponse(Instruction):
    ids = ("detectable_format:constrained_response",)

    def check(self, response: str) -> bool:
        response = response.strip()
        return any(option in response for option in ("Sim.", "Não.", "Talvez."))


class HighlightedSections(Instruction):
    ids = ("detectable_format:number_highlighted_sections",)

    def check(self, response: str) -> bool:
        single = [h for h in re.findall(r"\*[^\n\*]*\*", response) if h.strip("*").strip()]
        double = [
            h
            for h in re.findall(r"\*\*[^\n\*]*\*\*", response)
            if h.removeprefix("**").removesuffix("**").strip()
        ]
        return len(single) + len(double) >= self.kwargs["num_highlights"]


class Sections(Instruction):
    ids = ("detectable_format:multiple_sections",)

    def check(self, response: str) -> bool:
        splitter = self.kwargs["section_spliter"].strip()
        sections = re.split(r"\s?" + splitter + r"\s?\d+\s?", response)
        return len(sections) - 1 >= self.kwargs["num_sections"]


class JsonFormat(Instruction):
    ids = ("detectable_format:json_format",)

    def check(self, response: str) -> bool:
        response = response.strip()
        for prefix in ("```json", "```Json", "```JSON", "```"):
            response = response.removeprefix(prefix)
        try:
            json.loads(response.removesuffix("```").strip())
        except ValueError:
            return False
        return True


class Title(Instruction):
    ids = ("detectable_format:title",)

    def check(self, response: str) -> bool:
        return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", response))


class Postscript(Instruction):
    ids = ("detectable_content:postscript",)
    PATTERNS: ClassVar[dict[str, str]] = {
        "OBS.": r"^\s*obs\b\.?\s*:?\s*\S.*$",
        "P.S.": r"\s*p\.\s?s\..*$",
        "Nota": r"\s*nota.*$",
    }

    def check(self, response: str) -> bool:
        marker = self.kwargs["postscript_marker"].strip()
        pattern = self.PATTERNS.get(marker, r"\s*" + marker.lower() + r".*$")
        return bool(re.findall(pattern, response.lower(), flags=re.MULTILINE))


# --- início, fim e combinações --------------------------------------------------------


class EndPhrase(Instruction):
    ids = ("startend:end_checker",)

    def check(self, response: str) -> bool:
        return response.strip().strip('"').lower().endswith(self.kwargs["end_phrase"].strip().lower())


class Quotation(Instruction):
    ids = ("startend:quotation",)

    def check(self, response: str) -> bool:
        response = response.strip()
        return len(response) > 1 and response[0] == '"' and response[-1] == '"'


class TwoResponses(Instruction):
    ids = ("combination:two_responses",)

    def check(self, response: str) -> bool:
        valid = []
        parts = response.split("******")
        for index, part in enumerate(parts):
            if not part.strip():
                if index not in (0, len(parts) - 1):
                    return False
            else:
                valid.append(part)
        return len(valid) == 2 and valid[0].strip() != valid[1].strip()


class RepeatPrompt(Instruction):
    ids = ("combination:repeat_prompt",)

    def check(self, response: str) -> bool:
        return response.strip().lower().startswith(self.kwargs["prompt_to_repeat"].strip().lower())


# --- construções do português ---------------------------------------------------------


class Mesoclise(Instruction):
    ids = ("detectable_content:mesoclise_verification",)
    PATTERN = re.compile(
        r"\b\w+-(me|te|se|nos|vos|lhe|lhes|o|a|os|as|lo|la|los|las|no|na|nos|nas)"
        r"-(ei|ás|á|emos|eis|ão|ia|ias|íamos|íeis|iam)\b",
        re.IGNORECASE,
    )

    def check(self, response: str) -> bool:
        return self.PATTERN.search(response) is not None


class VosAddress(Instruction):
    ids = ("detectable_content:vos_address_verification",)

    def check(self, response: str) -> bool:
        response = response.lower()
        pronoun = re.search(r"\b(vós|vosso|vossa|vossos|vossas)\b", response)
        regular = re.search(r"\w+(ais|eis|istes|astes|sseis|ireis|ereis)\b", response)
        irregular = re.search(r"\b(sois|ides|tendes|vedes)\b", response)
        return pronoun is not None and (regular is not None or irregular is not None)


class SpecificPorque(Instruction):
    """Exatamente uma ocorrência de algum porquê, e ela é a forma pedida, bem empregada."""

    ids = ("detectable_content:specific_porque",)
    TYPES = ("porque", "porquê", "por que", "por quê")
    ALL_PORQUES = re.compile(r"(?i)\b(?:porque|porquê|porquês|por\s+que|por\s+quê)\b")
    DETERMINERS = frozenset(
        "o a os as um uma uns umas do da dos das no na nos nas ao aos à às "
        "este esta estes estas esse essa esses essas aquele aquela aqueles aquelas "
        "meu minha meus minhas teu tua teus tuas seu sua seus suas nosso nossa nossos nossas "
        "vosso vossa vossos vossas algum alguma alguns algumas nenhum nenhuma nenhuns nenhumas "
        "todo toda todos todas muito muita muitos muitas pouco pouca poucos poucas "
        "certo certa certos certas outro outra outros outras qual quais tal tais "
        "quanto quanta quantos quantas".split()
    )
    RELATIVE_NOUNS = frozenset(
        "motivo motivos razão razões razao razoes causa causas explicação explicações "
        "explicacao explicacoes caminho caminhos modo modos forma formas maneira maneiras "
        "meio meios".split()
    )
    INDIRECT_QUESTION = re.compile(
        r"(?i)\b(?:não\s+sei|nao\s+sei|sei|saber|sabia|sabiam|"
        r"gostaria(?:\s+de)?\s+saber|quero\s+saber|queria\s+saber|"
        r"entendo|entender|compreendo|compreender|explico|explicar|explique|"
        r"pergunto|perguntar|questiono|questionar|desconheço|desconheco|desconhecer|"
        r"investigo|investigar|descubro|descobrir|mostro|mostrar|digo|dizer|verifico|verificar)\b"
    )
    ALLOWED_AFTER_POR_QUE = frozenset(".?!,:;)]}\"'»”’")

    def check(self, response: str) -> bool:
        kind = self.kwargs["porque_type"]
        if kind not in self.TYPES:
            raise ValueError(f"porque_type deve ser um de {self.TYPES}, veio {kind!r}")
        text = response.strip()
        matches = list(self.ALL_PORQUES.finditer(text))
        if len(matches) != 1:
            return False
        match = matches[0]
        word = re.sub(r"\s+", " ", match.group().strip().lower())
        start, end = match.span()

        sentence_start = next((i + 1 for i in range(start - 1, -1, -1) if text[i] in ".?!"), 0)
        sentence_end = next((i + 1 for i in range(end, len(text)) if text[i] in ".?!"), len(text))
        sentence = text[sentence_start:sentence_end]
        prefix, suffix = sentence[: start - sentence_start], sentence[end - sentence_start :]
        previous_words = re.findall(r"\b\w+\b", prefix.lower(), flags=re.UNICODE)
        previous = previous_words[-1] if previous_words else ""
        next_char = next((c for c in suffix if not c.isspace()), "")
        has_question_mark = "?" in sentence
        sentence_initial = re.search(r"\w", prefix, flags=re.UNICODE) is None
        phrase_final = re.search(r"\w", suffix, flags=re.UNICODE) is None

        if kind == "porque":
            return (
                word == "porque"
                and previous not in self.DETERMINERS
                and not phrase_final
                and not (sentence_initial and has_question_mark)
            )
        if kind == "porquê":
            return word in {"porquê", "porquês"} and previous in self.DETERMINERS
        if kind == "por que":
            if word != "por que" or previous in self.DETERMINERS or phrase_final:
                return False
            if has_question_mark or previous in self.RELATIVE_NOUNS:
                return True
            return (
                bool(previous_words)
                and self.INDIRECT_QUESTION.search(" ".join(previous_words[-6:])) is not None
            )
        return word == "por quê" and (next_char == "" or next_char in self.ALLOWED_AFTER_POR_QUE)
