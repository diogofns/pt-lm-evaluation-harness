"""Verificadores de instruções do Portuguese-IFEval (strict e loose).

Portado de ark-aiml-research/Portuguese-IFEval (instructions/pt_instructions.py,
instruction_utils/pt_instructions_util.py e evaluation_main.py, commit a8d8b3c),
que deriva do IFEval da Google Research e do M-IFEval. Licença Apache-2.0,
Copyright 2024 The Google Research Authors.

A semântica de cada verificador é a da referência, inclusive as esquisitices
(palavras-chave usadas como regex cru, `capital_word_frequency` tokenizando com o
nltk em inglês), para que o mesmo texto receba o mesmo veredito; a paridade é
conferida em tests/test_parity_ifeval_pt.py. Diferença deliberada: o langdetect
recebe semente fixa, porque sem ela o veredito de textos limítrofes muda a cada execução.

Dependências (extra `ifeval`): spaCy com `pt_core_news_sm`, langdetect e nltk.
"""

from __future__ import annotations

import collections
import json
import re
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

LESS_THAN, AT_LEAST = "menos que", "ao menos"
_RELATIONS = (LESS_THAN, AT_LEAST)

Checker = Callable[[str, Mapping[str, Any], str], bool]
CHECKERS: dict[str, Checker] = {}


def _checker(*names: str) -> Callable[[Checker], Checker]:
    def register(fn: Checker) -> Checker:
        for name in names:
            CHECKERS[f"pt:{name}"] = fn
        return fn

    return register


# --- avaliação de uma resposta ---------------------------------------------------


def follows_strict(response: str, instruction: str, kwargs: Mapping[str, Any] | None, prompt: str) -> bool:
    check = CHECKERS[instruction]
    return bool(response.strip()) and check(response, kwargs or {}, prompt)


def follows_loose(response: str, instruction: str, kwargs: Mapping[str, Any] | None, prompt: str) -> bool:
    """Aceita se alguma variante passar: sem a primeira e/ou a última linha, sem asteriscos."""
    check = CHECKERS[instruction]
    return any(r.strip() and check(r, kwargs or {}, prompt) for r in loose_variants(response))


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


# --- utilitários de língua ---------------------------------------------------------


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


def _language_is(text: str, language: str) -> bool:
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


def _compare(actual: int, relation: str, threshold: int) -> bool:
    if relation not in _RELATIONS:
        raise ValueError(f"relação deve ser uma de {_RELATIONS}, veio {relation!r}")
    return actual < threshold if relation == LESS_THAN else actual >= threshold


# --- verificadores -----------------------------------------------------------------


@_checker("language:response_language")
def _response_language(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return _language_is(value, kw["language"])


@_checker("length_constraints:number_sentences")
def _number_sentences(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return _compare(count_sentences(value), kw["relation"], kw["num_sentences"])


@_checker("detectable_content:number_placeholders")
def _placeholders(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return len(re.findall(r"\[.*?\]", value)) >= kw["num_placeholders"]


@_checker("detectable_format:number_bullet_lists")
def _bullet_lists(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    stars = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
    return len(stars) + len(dashes) == kw["num_bullets"]


@_checker("detectable_format:constrained_response")
def _constrained_response(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    value = value.strip()
    return any(option in value for option in ("Sim.", "Não.", "Talvez."))


@_checker("detectable_format:number_highlighted_sections")
def _highlighted_sections(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    single = [h for h in re.findall(r"\*[^\n\*]*\*", value) if h.strip("*").strip()]
    double = [
        h for h in re.findall(r"\*\*[^\n\*]*\*\*", value) if h.removeprefix("**").removesuffix("**").strip()
    ]
    return len(single) + len(double) >= kw["num_highlights"]


@_checker("detectable_format:multiple_sections")
def _sections(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    splitter = kw["section_spliter"].strip()
    sections = re.split(r"\s?" + splitter + r"\s?\d+\s?", value)
    return len(sections) - 1 >= kw["num_sections"]


@_checker("length_constraints:number_paragraphs")
def _paragraphs(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index in (0, len(paragraphs) - 1):
                count -= 1
            else:
                return False
    return count == kw["num_paragraphs"]


@_checker("detectable_content:postscript")
def _postscript(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    marker = kw["postscript_marker"].strip()
    patterns = {"OBS.": r"^\s*obs\b\.?\s*:?\s*\S.*$", "P.S.": r"\s*p\.\s?s\..*$", "Nota": r"\s*nota.*$"}
    pattern = patterns.get(marker, r"\s*" + marker.lower() + r".*$")
    return bool(re.findall(pattern, value.lower(), flags=re.MULTILINE))


@_checker("keywords:existence")
def _keywords(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return all(re.search(k, value, flags=re.IGNORECASE) for k in sorted(kw["keywords"]))


@_checker("keywords:frequency")
def _keyword_frequency(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    occurrences = len(re.findall(kw["keyword"].strip(), value, flags=re.IGNORECASE))
    return _compare(occurrences, kw["relation"], kw["frequency"])


@_checker("length_constraints:number_words")
def _number_words(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return _compare(count_words(value), kw["relation"], kw["num_words"])


@_checker("detectable_format:json_format")
def _json(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    value = value.strip()
    for prefix in ("```json", "```Json", "```JSON", "```"):
        value = value.removeprefix(prefix)
    try:
        json.loads(value.removesuffix("```").strip())
    except ValueError:
        return False
    return True


@_checker("length_constraints:nth_paragraph_first_word")
def _nth_paragraph_first_word(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    paragraphs = re.split(r"\n\n", value)
    count = sum(1 for p in paragraphs if p.strip())
    nth = kw["nth_paragraph"]
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
    return count == kw["num_paragraphs"] and first_word == kw["first_word"].lower()


@_checker("keywords:forbidden_words")
def _forbidden_words(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    words = sorted(set(kw["forbidden_words"]))
    return not any(re.search(r"\b" + w + r"\b", value, flags=re.IGNORECASE) for w in words)


@_checker("combination:two_responses")
def _two_responses(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    valid = []
    parts = value.split("******")
    for index, part in enumerate(parts):
        if not part.strip():
            if index not in (0, len(parts) - 1):
                return False
        else:
            valid.append(part)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


@_checker("combination:repeat_prompt")
def _repeat_prompt(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return value.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())


@_checker("startend:end_checker")
def _end_checker(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return value.strip().strip('"').lower().endswith(kw["end_phrase"].strip().lower())


@_checker("detectable_format:title")
def _title(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", value))


@_checker("keywords:letter_frequency")
def _letter_frequency(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    letter = kw["letter"].strip().lower()
    return _compare(collections.Counter(value.lower())[letter], kw["let_relation"], kw["let_frequency"])


@_checker("change_case:english_capital", "change_case:portuguese_capital")
def _all_capital(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return value.isupper() and _language_is(value, "pt")


@_checker("change_case:english_lowercase", "change_case:portuguese_lowercase")
def _all_lowercase(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return value.islower() and _language_is(value, "pt")


@_checker("punctuation:no_comma")
def _no_comma(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return "," not in value


@_checker("change_case:capital_word_frequency")
def _capital_word_frequency(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    capital = sum(1 for word in _word_tokenize()(value) if word.isupper())
    return _compare(capital, kw["capital_relation"], kw["capital_frequency"])


@_checker("startend:quotation")
def _quotation(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    value = value.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


@_checker("keywords:cedilha_frequency")
def _cedilha(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return value.lower().count("ç") == kw["count"]


@_checker("punctuation:no_tilde")
def _no_tilde(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return re.search(r"[ãõñÃÕÑ]", value) is None


@_checker("punctuation:crase_presence")
def _crase(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    return re.search(r"\b(?:à|às|àquele|àquela|àqueles|àquelas|àquilo)\b", value, re.IGNORECASE) is not None


@_checker("detectable_content:mesoclise_verification")
def _mesoclise(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    pattern = (
        r"\b\w+-(me|te|se|nos|vos|lhe|lhes|o|a|os|as|lo|la|los|las|no|na|nos|nas)"
        r"-(ei|ás|á|emos|eis|ão|ia|ias|íamos|íeis|iam)\b"
    )
    return re.search(pattern, value, re.IGNORECASE) is not None


@_checker("detectable_content:vos_address_verification")
def _vos(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    value = value.lower()
    pronoun = re.search(r"\b(vós|vosso|vossa|vossos|vossas)\b", value)
    regular = re.search(r"\w+(ais|eis|istes|astes|sseis|ireis|ereis)\b", value)
    irregular = re.search(r"\b(sois|ides|tendes|vedes)\b", value)
    return pronoun is not None and (regular is not None or irregular is not None)


# --- os quatro porquês -------------------------------------------------------------

_PORQUE_TYPES = ("porque", "porquê", "por que", "por quê")
_ALL_PORQUES = re.compile(r"(?i)\b(?:porque|porquê|porquês|por\s+que|por\s+quê)\b")
_DETERMINERS = frozenset(
    "o a os as um uma uns umas do da dos das no na nos nas ao aos à às "
    "este esta estes estas esse essa esses essas aquele aquela aqueles aquelas "
    "meu minha meus minhas teu tua teus tuas seu sua seus suas nosso nossa nossos nossas "
    "vosso vossa vossos vossas algum alguma alguns algumas nenhum nenhuma nenhuns nenhumas "
    "todo toda todos todas muito muita muitos muitas pouco pouca poucos poucas "
    "certo certa certos certas outro outra outros outras qual quais tal tais "
    "quanto quanta quantos quantas".split()
)
_RELATIVE_NOUNS = frozenset(
    "motivo motivos razão razões razao razoes causa causas explicação explicações "
    "explicacao explicacoes caminho caminhos modo modos forma formas maneira maneiras "
    "meio meios".split()
)
_INDIRECT_QUESTION = re.compile(
    r"(?i)\b(?:não\s+sei|nao\s+sei|sei|saber|sabia|sabiam|"
    r"gostaria(?:\s+de)?\s+saber|quero\s+saber|queria\s+saber|"
    r"entendo|entender|compreendo|compreender|explico|explicar|explique|"
    r"pergunto|perguntar|questiono|questionar|desconheço|desconheco|desconhecer|"
    r"investigo|investigar|descubro|descobrir|mostro|mostrar|digo|dizer|verifico|verificar)\b"
)
_ALLOWED_AFTER_POR_QUE = frozenset(".?!,:;)]}\"'»”’")


@_checker("detectable_content:specific_porque")
def _specific_porque(value: str, kw: Mapping[str, Any], prompt: str) -> bool:
    """Exatamente uma ocorrência de algum porquê, e ela é a forma pedida, bem empregada."""
    kind = kw["porque_type"]
    if kind not in _PORQUE_TYPES:
        raise ValueError(f"porque_type deve ser um de {_PORQUE_TYPES}, veio {kind!r}")
    text = value.strip()
    matches = list(_ALL_PORQUES.finditer(text))
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
            and previous not in _DETERMINERS
            and not phrase_final
            and not (sentence_initial and has_question_mark)
        )
    if kind == "porquê":
        return word in {"porquê", "porquês"} and previous in _DETERMINERS
    if kind == "por que":
        if word != "por que" or previous in _DETERMINERS or phrase_final:
            return False
        if has_question_mark or previous in _RELATIVE_NOUNS:
            return True
        return bool(previous_words) and _INDIRECT_QUESTION.search(" ".join(previous_words[-6:])) is not None
    return word == "por quê" and (next_char == "" or next_char in _ALLOWED_AFTER_POR_QUE)
