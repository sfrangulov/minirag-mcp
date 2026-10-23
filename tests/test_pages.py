"""PDF page numbers: where each page starts in the converted text, and which pages
each chunk came from. Built from real PDFs, converted by the real markitdown."""

from __future__ import annotations

import pytest

from conftest import FakeEmbedder
from minirag_mcp import ocr
from minirag_mcp.chunker import chunk_markdown
from minirag_mcp.config import load_config
from minirag_mcp.ingest import pages
from minirag_mcp.ingest.parser import parse_file
from pdf_builder import build_pdf

# Cells laid out in three aligned columns: markitdown converts a document with a page
# like this through pdfplumber, page by page, instead of through pdfminer.
TABLE = [
    (72, 700, "Name"),
    (250, 700, "Qty"),
    (400, 700, "Price"),
    (72, 680, "Apple"),
    (250, 680, "3"),
    (400, 680, "1.20"),
    (72, 660, "Pear"),
    (250, 660, "5"),
    (400, 660, "2.10"),
    (72, 640, "Plum"),
    (250, 640, "7"),
    (400, 640, "0.50"),
]
LONG = "long enough text line to clear the default per-page threshold easily"


def _pdf(tmp_path, page_list, name="doc.pdf"):
    path = tmp_path / name
    path.write_bytes(build_pdf(page_list))
    return path


def _chunk_pages(path, config=None) -> list[tuple[str, str]]:
    doc = parse_file(path, config)
    chunks = chunk_markdown(doc.markdown, count_tokens=FakeEmbedder().count_tokens, title=doc.title)
    texts = [c.text for c in chunks]
    return list(zip(texts, pages.chunk_pages(doc.markdown, doc.page_starts, texts), strict=True))


def _page_of(pairs: list[tuple[str, str]], needle: str) -> str:
    found = [label for text, label in pairs if needle in text]
    assert len(found) == 1, f"{needle!r} is in {len(found)} chunks: {pairs}"
    return found[0]


def test_every_chunk_of_a_prose_pdf_names_its_page(tmp_path):
    path = _pdf(
        tmp_path,
        [
            ["Alpha opens the first page."],
            ["Beta fills the second page."],
            ["Gamma closes on the third page."],
        ],
    )
    got = _chunk_pages(path)
    assert _page_of(got, "Alpha") == "1"
    assert _page_of(got, "Beta") == "2"
    assert _page_of(got, "Gamma") == "3"


def test_a_paragraph_that_starts_mid_page_carries_that_page(tmp_path):
    path = _pdf(
        tmp_path,
        [
            ["Alpha is the whole first page."],
            ["Beta opens page two.", "", "Gamma starts halfway down page two."],
            ["Delta is on page three."],
        ],
    )
    got = _chunk_pages(path)
    assert _page_of(got, "Gamma") == "2"
    assert _page_of(got, "Delta") == "3"


def test_a_chunk_that_runs_across_a_page_break_names_both_pages(tmp_path):
    """A transcript window is one section, so a page break inside it lands inside a
    chunk."""
    path = _pdf(
        tmp_path,
        [
            ["Weekly sync", "0:03", "Dana opens the meeting.", "0:20", "Sam reports the budget."],
            [
                "and finishes the budget report.",
                "0:45",
                "Dana asks about travel.",
                "1:10",
                "Sam agrees.",
                "1:30",
                "Dana closes.",
            ],
        ],
    )
    got = _chunk_pages(path)
    assert _page_of(got, "Sam reports the budget") == "1-2"


def test_a_table_page_carries_its_page_and_the_pages_after_it_stay_right(tmp_path):
    path = _pdf(
        tmp_path,
        [["Intro prose on page one.", "", "More intro prose."], TABLE, [], ["Closing prose."]],
    )
    doc = parse_file(path)
    assert "\f" not in doc.markdown, "this has to take markitdown's pdfplumber route"
    got = _chunk_pages(path)
    assert _page_of(got, "Intro prose") == "1"
    assert _page_of(got, "Apple") == "2"
    assert _page_of(got, "Closing prose") == "4"


def test_a_page_of_bare_spaces_does_not_shift_the_pages_after_it(tmp_path):
    """markitdown strips each line's trailing whitespace, so the form feed that sits on
    a line of spaces is lost with them: counting form feeds puts Gamma on page 2."""
    path = _pdf(tmp_path, [["Alpha on page one."], " ", ["Gamma on page three."]])
    assert parse_file(path).markdown.count("\f") == 1
    got = _chunk_pages(path)
    assert _page_of(got, "Alpha") == "1"
    assert _page_of(got, "Gamma") == "3"


def test_blank_pages_before_and_after_the_text_leave_its_numbers_right(tmp_path):
    path = _pdf(tmp_path, ["", ["Beta on page two."], ["Gamma on page three."], "", ""])
    got = _chunk_pages(path)
    assert _page_of(got, "Beta") == "2"
    assert _page_of(got, "Gamma") == "3"


TWO_PARAGRAPHS = ["First paragraph.", "", "Second paragraph."]


@pytest.mark.parametrize(
    ("page_list", "page"),
    [
        (["", TWO_PARAGRAPHS, ""], "2"),
        # every form feed is on the blank pages' line, and goes with it
        ([TWO_PARAGRAPHS, "", ""], "1"),
    ],
)
def test_text_on_one_page_of_several_is_that_page(tmp_path, page_list, page):
    got = _chunk_pages(_pdf(tmp_path, page_list))
    assert _page_of(got, "First paragraph") == page
    assert _page_of(got, "Second paragraph") == page


def test_a_single_page_pdf_is_page_one(tmp_path):
    got = _chunk_pages(_pdf(tmp_path, [["Only page.", "", "Still the only page."]]))
    assert {label for _, label in got} == {"1"}


def test_separators_that_do_not_match_the_pages_give_no_numbers_rather_than_wrong_ones(
    tmp_path, monkeypatch
):
    path = _pdf(tmp_path, [["Alpha on page one."], " ", ["Gamma on page three."]])
    monkeypatch.setattr(pages, "text_pages", lambda p: [0, 1, 2])
    assert parse_file(path).page_starts == ()
    assert {label for _, label in _chunk_pages(path)} == {""}


def test_an_unreadable_page_tree_leaves_the_document_without_numbers(tmp_path, monkeypatch):
    path = _pdf(tmp_path, [["Alpha."], ["Beta."]])

    def broken(p):
        raise ValueError("no page tree")

    monkeypatch.setattr(pages, "page_count", broken)
    doc = parse_file(path)
    assert "Alpha" in doc.markdown
    assert doc.page_starts == ()


def test_other_formats_carry_no_page_starts(tmp_path):
    md = tmp_path / "note.md"
    md.write_text("# Note\n\nNothing paged here.", encoding="utf-8")
    assert parse_file(md).page_starts == ()


# --- scanned pages: OCR text is appended after the converted document --------------


@pytest.fixture
def fake_ocr(monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_pdf", lambda path, config, todo: {i: f"Recognized page index {i}" for i in todo}
    )


def test_a_scanned_page_keeps_its_own_page_number_though_its_text_comes_last(tmp_path, fake_ocr):
    path = _pdf(tmp_path, ["", [LONG], ""])
    got = _chunk_pages(path, load_config({}, cwd=tmp_path))
    assert _page_of(got, "long enough") == "2"
    assert _page_of(got, "page index 0") == "1"
    assert _page_of(got, "page index 2") == "3"


def test_recognized_pages_follow_the_converted_text_one_blank_line_apart(tmp_path, fake_ocr):
    doc = parse_file(_pdf(tmp_path, ["", [LONG], ""]), load_config({}, cwd=tmp_path))
    assert doc.markdown == f"{LONG}\n\nRecognized page index 0\n\nRecognized page index 2"


def test_a_scanned_page_after_a_table_page_keeps_its_number(tmp_path, fake_ocr):
    path = _pdf(tmp_path, [TABLE, "", [LONG]])
    got = _chunk_pages(path, load_config({}, cwd=tmp_path))
    assert _page_of(got, "Apple") == "1"
    assert _page_of(got, "page index 1") == "2"
    assert _page_of(got, "long enough") == "3"


def test_when_the_converted_pages_are_unknown_only_the_scanned_ones_are_numbered(
    tmp_path, fake_ocr, monkeypatch
):
    path = _pdf(tmp_path, [[LONG], " ", [LONG + " again"], ""])
    monkeypatch.setattr(pages, "text_pages", lambda p: [0, 1, 2])
    got = _chunk_pages(path, load_config({}, cwd=tmp_path))
    assert _page_of(got, "again") == ""
    assert _page_of(got, "page index 1") == "2"
    assert _page_of(got, "page index 3") == "4"


# --- chunk_pages: chunks against the text they were cut from ------------------------


def _starts(markdown: str) -> pages.PageStarts:
    """Page 1 from the start, and the next page after every form feed."""
    feeds = [i + 1 for i, ch in enumerate(markdown) if ch == "\f"]
    return ((0, 1), *((at, n) for n, at in enumerate(feeds, start=2)))


def test_context_a_chunk_repeats_does_not_widen_its_pages():
    """A table's header row, repeated in front of rows from a later page, must not pull
    the chunk back to the header's page."""
    header = "| Name | Qty |\n| --- | --- |"
    markdown = f"{header}\n| a | 1 |\n| b | 2 |\n\f| c | 3 |\n| d | 4 |"
    texts = [f"{header}\n| a | 1 |\n| b | 2 |", f"{header}\n| c | 3 |\n| d | 4 |"]
    assert pages.chunk_pages(markdown, _starts(markdown), texts) == ["1", "2"]


def test_a_breadcrumb_from_an_earlier_page_does_not_widen_a_chunk():
    """The outer heading ends page 2 and is carried in the breadcrumb; the chunk's own
    heading and text are on page 3."""
    markdown = "Intro words on page one.\n\n\f# Part\n\n\f## Scope\n\nClause text on page three."
    texts = ["Intro words on page one.", "Part > Scope\n\nClause text on page three."]
    assert pages.chunk_pages(markdown, _starts(markdown), texts) == ["1", "3"]


def test_a_seeded_title_does_not_stop_the_first_chunk_from_being_found():
    markdown = "Opening words of the body.\n\n\fSecond page words."
    texts = ["# Some Title\n\nOpening words of the body.", "Second page words."]
    assert pages.chunk_pages(markdown, _starts(markdown), texts) == ["1", "2"]


def test_a_chunk_too_short_to_place_gets_the_pages_between_its_neighbours():
    """`ok` under a breadcrumb that is nowhere in the text verbatim: the chunk cannot be
    found, but it can only be in what lies between the chunks either side of it."""
    markdown = "First page text here.\n\n\fok\n\n\fThird page text here."
    texts = ["First page text here.", "Some > Breadcrumb\n\nok", "Third page text here."]
    assert pages.chunk_pages(markdown, _starts(markdown), texts) == ["1", "2", "3"]


def test_a_chunk_is_found_after_the_one_before_it_not_at_an_earlier_copy():
    markdown = "Confidential notice.\n\nBody one.\n\n\fConfidential notice.\n\nBody two."
    texts = ["Confidential notice.", "Body one.", "Confidential notice.", "Body two."]
    assert pages.chunk_pages(markdown, _starts(markdown), texts) == ["1", "1", "2", "2"]


def test_text_on_an_unknown_page_gets_no_label():
    markdown = "Unknown origin words.\n\nKnown page words."
    starts = ((0, pages.UNKNOWN), (markdown.index("Known"), 7))
    texts = ["Unknown origin words.", "Known page words."]
    assert pages.chunk_pages(markdown, starts, texts) == ["", "7"]


def test_a_joined_part_without_pages_does_not_take_the_page_of_the_part_before_it():
    text, starts = pages.joined([("Known words.", ((0, 4),)), ("Unknown words.", ())])
    assert text == "Known words.\n\nUnknown words."
    assert pages.chunk_pages(text, starts, ["Known words.", "Unknown words."]) == ["4", ""]


def test_no_page_starts_means_no_labels():
    assert pages.chunk_pages("Some text.", (), ["Some text."]) == [""]
