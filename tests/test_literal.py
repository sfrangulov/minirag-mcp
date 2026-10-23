import unicodedata

import pytest

from minirag_mcp.literal import MAX_FORM_CHARS, fold, matches, term


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def found(raw: str, text: str) -> str | None:
    (m,) = matches([term(raw)], text)
    if m is None:
        return None
    return unicodedata.normalize("NFC", text)[m.span[0] : m.span[1]]


def test_fold_ignores_case_yo_and_composition():
    assert fold("ОТЧЁТ") == fold("отчет") == fold(nfd("Отчёт"))
    assert fold(nfd("Свод показателей")) == "свод показателей"


def test_fold_never_changes_length_so_spans_map_back():
    # "İ".lower() is two characters; a span found in folded text must index the original.
    text = "İstanbul ОТЧЁТ"
    assert len(fold(text)) == len(unicodedata.normalize("NFC", text))
    assert found("отчет", text) == "ОТЧЁТ"


@pytest.mark.parametrize(
    "text",
    ["Акт сверки", "два акта", "актов нет", "(акт)", "ТЗ-акт", "file_акт", "1.акт", "актуально"],
)
def test_a_term_matches_at_the_start_of_a_word_with_any_ending(text):
    assert found("акт", text) is not None


@pytest.mark.parametrize("text", ["контакт", "фактически", "характер", "2акт"])
def test_a_term_inside_a_word_does_not_match(text):
    assert found("акт", text) is None


def test_the_matched_span_is_the_whole_word_as_written_in_the_text():
    assert found("договор", "Подписан Договора №5") == "Договора"


@pytest.mark.parametrize("raw", ['"акт"', "«акт»", "“акт”", "'акт'"])
def test_a_quoted_term_matches_only_that_exact_word(raw):
    assert found(raw, "Акт сверки") == "Акт"
    assert found(raw, "два акта") is None
    assert found(raw, "актуально") is None
    assert found(raw, "акт.") == "акт"


def test_a_quoted_phrase_is_exact_at_both_ends():
    assert found('"акт сверки"', "Акт  сверки №5") == "Акт  сверки"
    assert found('"акт сверки"', "акт сверкий") is None


def test_every_matched_word_form_is_reported_folded():
    (m,) = matches([term("акт")], "Акт, акта, АКТУАЛЬНО и снова акт; контакт")
    assert m.forms == frozenset({"акт", "акта", "актуально"})
    assert m.span == (0, 3)


def test_a_phrase_form_has_its_whitespace_collapsed():
    (m,) = matches([term("акт свер")], "Акт\n  сверки")
    assert m.forms == frozenset({"акт сверки"})


def test_a_term_that_does_not_occur_has_no_match():
    missing, present = matches([term("акт"), term("склад")], "склады")
    assert missing is None
    assert present.forms == frozenset({"склады"})


def test_a_phrase_matches_across_any_run_of_whitespace():
    assert found("акт сверки", "Акт\n   сверки взаиморасчётов") == "Акт\n   сверки"
    assert found("акт сверки", "Акт\u00a0сверки") == "Акт\u00a0сверки"


def test_a_term_starting_with_punctuation_needs_no_word_start():
    assert found("#12", "issue#12 closed") == "#12"


def test_regex_metacharacters_in_a_term_are_literal():
    assert found("4.2.1", "пункт 4.2.1 регламента") == "4.2.1"
    assert found("4.2.1", "пункт 4x2y1") is None
    assert found("(c)", "copyright (c) 2026") == "(c)"


def test_an_nfd_text_matches_an_nfc_term():
    assert found("показателей", nfd("Свод показателей")) is not None


def test_markdown_escapes_in_converted_documents_are_transparent():
    # markitdown writes DOCX, HTML and XLSX text through markdownify, which escapes these.
    assert found("ТЗ_2024_015", r"см. ТЗ\_2024\_015") == r"ТЗ\_2024\_015"
    assert found("5*3", r"итого 5\*3") == r"5\*3"
    (m,) = matches([term("тз_2024")], r"ТЗ\_2024\_015")
    assert m.forms == frozenset({"тз_2024"})


@pytest.mark.parametrize("mark", ["\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"])
def test_invisible_characters_inside_a_word_are_transparent(mark):
    assert found("договор", f"Наш дого{mark}вор") == f"дого{mark}вор"


def test_greek_final_sigma_matches_either_way():
    assert found("οδος", "ΟΔΟΣ") == "ΟΔΟΣ"
    assert found("ΟΔΟΣ", "οδος") == "οδος"


def test_dotted_capital_i_matches_a_plain_i():
    assert found("istanbul", "İstanbul") == "İstanbul"


def test_a_runaway_word_form_is_cut():
    (m,) = matches([term("x")], "x" + "y" * 5000)
    (form,) = m.forms
    assert len(form) == MAX_FORM_CHARS + 1 and form.endswith("…")
