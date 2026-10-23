"""PDF page numbers: where each page starts in the converted text, and which pages
each chunk came from. Built from real PDFs, converted by the real markitdown."""

from __future__ import annotations

import re

import pytest

from conftest import FakeEmbedder
from minirag_mcp import ocr
from minirag_mcp.chunker import Chunk, chunk_markdown, estimate_tokens
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
# Text that says which page it was set on: P<page>L<line>.
MARKER = re.compile(r"P(\d+)L\d+")


def _pdf(tmp_path, page_list, name="doc.pdf"):
    path = tmp_path / name
    path.write_bytes(build_pdf(page_list))
    return path


COUNT = FakeEmbedder().count_tokens


def _chunks(path, config=None, count_tokens=COUNT) -> tuple[list[Chunk], list[str]]:
    doc = parse_file(path, config)
    chunks = chunk_markdown(doc.markdown, count_tokens=count_tokens, title=doc.title)
    return chunks, pages.chunk_pages(doc.markdown, doc.page_starts, chunks)


def _chunk_pages(path, config=None) -> list[tuple[str, str]]:
    chunks, labels = _chunks(path, config)
    return list(zip([c.text for c in chunks], labels, strict=True))


def _page_of(pairs: list[tuple[str, str]], needle: str) -> str:
    found = [label for text, label in pairs if needle in text]
    assert len(found) == 1, f"{needle!r} is in {len(found)} chunks: {pairs}"
    return found[0]


def _misplaced(path, config=None) -> tuple[list[tuple[str, str]], int]:
    """Chunks whose label is not exactly the pages of the markers in their own text,
    and how many chunks carry a label at all."""
    chunks, labels = _chunks(path, config)
    wrong = []
    for chunk, label in zip(chunks, labels, strict=True):
        found = sorted({int(p) for p in MARKER.findall(chunk.text[chunk.own_start :])})
        if label and found:
            expected = str(found[0]) if found[0] == found[-1] else f"{found[0]}-{found[-1]}"
            if label != expected:
                wrong.append((label, chunk.text))
    return wrong, sum(1 for label in labels if label)


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


def test_a_transcript_whose_clock_starts_again_gets_no_wrong_page(tmp_path):
    """Two sittings in one file: their first minutes share a time window, so chunks no
    longer follow the document's order."""

    def sitting(first_page, pages_long):
        out = []
        for k in range(pages_long):
            lines = []
            for m in range(4):
                t = (k * 4 + m) * 60
                lines += [f"{t // 60}:{t % 60:02d}", f"P{first_page + k}L{m} says item {m}."]
            out.append(lines)
        return out

    wrong, labelled = _misplaced(_pdf(tmp_path, sitting(1, 2) + sitting(3, 1)))
    assert wrong == []
    assert labelled


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


def _rows(top, cells):
    out = []
    for k, (a, b, c) in enumerate(cells):
        y = top - 18 * k
        out += [(72, y, a), (250, y, b), (400, y, c)]
    return out


def test_identical_rows_on_both_sides_of_a_page_break_keep_their_own_pages(tmp_path):
    """Every row is the same text, so only order can tell them apart: a chunk of
    page-2 rows must not be placed among the page-3 copies."""
    same = ("Bolt M8", "10", "1.20")
    page2 = _rows(740, [("Item", "Qty", "Price"), ("Nut M6", "4", "0.30")] + [same] * 6)
    page3 = _rows(740, [same] * 6 + [("Washer", "2", "0.05")])
    path = _pdf(tmp_path, [["Intro prose on page one."], page2, page3])
    chunks, labels = _chunks(path, count_tokens=estimate_tokens)
    # Data rows in document order. On page 3 the first row is read as the header of a
    # new table, so it is context, not a row of its own.
    row_pages = [2] * 7 + [3] * 6
    seen = 0
    checked = 0
    for chunk, label in zip(chunks, labels, strict=True):
        rows = [
            ln
            for ln in chunk.text[chunk.own_start :].split("\n")
            if ln.startswith("|") and "---" not in ln
        ]
        if not rows:
            continue
        span = sorted(set(row_pages[seen : seen + len(rows)]))
        seen += len(rows)
        assert label == "-".join(str(p) for p in dict.fromkeys([span[0], span[-1]])), (
            label,
            chunk.text,
        )
        checked += 1
    assert seen == len(row_pages) and checked >= 4


def test_a_page_of_bare_spaces_does_not_shift_the_pages_after_it(tmp_path):
    """markitdown strips each line's trailing whitespace, so the form feed that sits on
    a line of spaces is lost with them: counting form feeds puts Gamma on page 2."""
    path = _pdf(tmp_path, [["Alpha on page one."], " ", ["Gamma on page three."]])
    assert parse_file(path).markdown.count("\f") == 1
    got = _chunk_pages(path)
    assert _page_of(got, "Alpha") == "1"
    assert _page_of(got, "Gamma") == "3"


NBSP = "\xa0"


def test_a_block_too_many_and_one_too_few_do_not_cancel_out(tmp_path):
    """A row of no-break spaces on a table page splits that page in two, and a `.1` left
    at the foot of a page is merged into the next page's first line: one block more and
    one fewer, so the counts still agree while every block between them is a page off."""
    table_page = [(72, 740, "P3L0 heading line"), (72, 720, NBSP)] + [
        (x, y, text) for x, y, text in TABLE if y >= 660
    ]
    path = _pdf(
        tmp_path,
        [
            ["P1L1 first page text.", ".1"],
            ["P2L1 the intent of this request."],
            table_page,
            ["P4L1 end."],
        ],
    )
    wrong, _ = _misplaced(path)
    assert wrong == []


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
    real = pages.page_prints(path)
    monkeypatch.setattr(pages, "page_prints", lambda p: [real[0], real[2], real[2]])
    assert parse_file(path).page_starts == ()
    assert {label for _, label in _chunk_pages(path)} == {""}


def test_a_block_that_is_not_its_page_s_text_gives_no_numbers(tmp_path, monkeypatch):
    path = _pdf(tmp_path, [["Alpha on page one."], " ", ["Gamma on page three."]])
    real = pages.page_prints(path)
    monkeypatch.setattr(pages, "page_prints", lambda p: [real[0], real[1], real[0]])
    assert parse_file(path).page_starts == ()


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
    real = pages.page_prints(path)
    monkeypatch.setattr(pages, "page_prints", lambda p: [real[0], real[0], real[2], real[3]])
    got = _chunk_pages(path, load_config({}, cwd=tmp_path))
    assert _page_of(got, "again") == ""
    assert _page_of(got, "page index 1") == "2"
    assert _page_of(got, "page index 3") == "4"


# --- chunk_pages: chunks against the text they were cut from ------------------------


def _starts(markdown: str) -> pages.PageStarts:
    """Page 1 from the start, and the next page after every form feed."""
    feeds = [i + 1 for i, ch in enumerate(markdown) if ch == "\f"]
    return ((0, 1), *((at, n) for n, at in enumerate(feeds, start=2)))


def _chunk(own: str, context: str = "") -> Chunk:
    return Chunk(text=context + own, parent_index=0, own_start=len(context))


def test_context_a_chunk_repeats_does_not_widen_its_pages():
    """A table's header row, repeated in front of rows from a later page, must not pull
    the chunk back to the header's page."""
    header = "| Name | Qty |\n| --- | --- |\n"
    markdown = f"{header}| a | 1 |\n| b | 2 |\n\f| c | 3 |\n| d | 4 |"
    chunks = [_chunk("| a | 1 |\n| b | 2 |", header), _chunk("| c | 3 |\n| d | 4 |", header)]
    assert pages.chunk_pages(markdown, _starts(markdown), chunks) == ["1", "2"]


def test_a_row_is_not_placed_at_a_later_copy_reached_through_the_header():
    header = "| A | B |\n| --- | --- |\n"
    markdown = f"{header}| x | 1 |\n| x | 1 |\n\n\f| x | 1 |\n| --- | --- |\n| x | 1 |"
    chunks = [
        _chunk("| x | 1 |", header),
        _chunk("| x | 1 |", header),
        _chunk("| x | 1 |", "| x | 1 |\n| --- | --- |\n"),
    ]
    assert pages.chunk_pages(markdown, _starts(markdown), chunks) == ["1", "1", "2"]


def test_a_breadcrumb_from_an_earlier_page_does_not_widen_a_chunk():
    markdown = "Intro words on page one.\n\n\f# Part\n\n\f## Scope\n\nClause text on page three."
    chunks = [
        _chunk("Intro words on page one."),
        _chunk("Clause text on page three.", "Part > Scope\n\n"),
    ]
    assert pages.chunk_pages(markdown, _starts(markdown), chunks) == ["1", "3"]


def test_a_chunk_whose_own_text_is_not_in_the_document_gets_no_page():
    """A heading with nothing under it becomes a chunk of its cleaned words, which the
    document does not hold verbatim. It is left without a page, and the chunks after it
    are still placed."""
    markdown = "First page text here.\n\n\f# **2** **Reserve**\n\n\fThird page text here."
    chunks = [_chunk("First page text here."), _chunk("2 Reserve"), _chunk("Third page text here.")]
    assert pages.chunk_pages(markdown, _starts(markdown), chunks) == ["1", "", "3"]


def test_own_text_is_only_ever_matched_as_whole_words():
    """`Страница 38` is not in `Страница 380`."""
    markdown = "Alpha.\n\n\fСтраница 380"
    chunks = [_chunk("Alpha."), _chunk("Страница 38")]
    assert pages.chunk_pages(markdown, _starts(markdown), chunks) == ["1", ""]


def test_own_text_far_past_where_the_chunk_before_it_ended_is_not_taken():
    filler = "\n\n".join(f"Uncovered paragraph {i} with some words in it." for i in range(400))
    markdown = f"Alpha opens.\n\n{filler}\n\n\fOmega closes."
    chunks = [_chunk("Alpha opens."), _chunk("Omega closes.")]
    assert pages.chunk_pages(markdown, _starts(markdown), chunks) == ["1", ""]


def test_a_chunk_not_found_ahead_is_not_placed_at_a_copy_behind():
    filler = "\n\n".join(f"Uncovered paragraph {i} with some words in it." for i in range(400))
    markdown = f"Signed by both parties.\n\nAlpha opens.\n\n{filler}\n\n\fSigned by both parties."
    chunks = [
        _chunk("Signed by both parties."),
        _chunk("Alpha opens."),
        _chunk("Signed by both parties."),
    ]
    assert pages.chunk_pages(markdown, _starts(markdown), chunks) == ["1", "1", ""]


def test_a_chunk_is_found_after_the_one_before_it_not_at_an_earlier_copy():
    markdown = "Confidential notice.\n\nBody one.\n\n\fConfidential notice.\n\nBody two."
    chunks = [
        _chunk(text)
        for text in ("Confidential notice.", "Body one.", "Confidential notice.", "Body two.")
    ]
    assert pages.chunk_pages(markdown, _starts(markdown), chunks) == ["1", "1", "2", "2"]


def test_out_of_order_chunks_are_placed_only_where_their_text_is_unique():
    """A transcript whose clock starts again: the second chunk lies before the first, and
    the third's words are in the document twice."""
    turns = "\n".join(f"{m}:00\nturn {m} of the first sitting." for m in range(6))
    again = "\n".join(f"{m}:00\nturn {m} again." for m in range(6))
    markdown = f"{turns}\n\f{again}\n0:30\nturn 0 of the first sitting."
    chunks = [
        _chunk("0:30\nturn 0 of the first sitting."),
        _chunk("1:00\nturn 1 of the first sitting."),
        _chunk("turn 0 of the first sitting."),
    ]
    assert pages.chunk_pages(markdown, _starts(markdown), chunks) == ["2", "1", ""]


def test_text_on_an_unknown_page_gets_no_label():
    markdown = "Unknown origin words.\n\nKnown page words."
    starts = ((0, pages.UNKNOWN), (markdown.index("Known"), 7))
    chunks = [_chunk("Unknown origin words."), _chunk("Known page words.")]
    assert pages.chunk_pages(markdown, starts, chunks) == ["", "7"]


def test_a_joined_part_without_pages_does_not_take_the_page_of_the_part_before_it():
    text, starts = pages.joined([("Known words.", ((0, 4),)), ("Unknown words.", ())])
    assert text == "Known words.\n\nUnknown words."
    chunks = [_chunk("Known words."), _chunk("Unknown words.")]
    assert pages.chunk_pages(text, starts, chunks) == ["4", ""]


def test_no_page_starts_means_no_labels():
    assert pages.chunk_pages("Some text.", (), [_chunk("Some text.")]) == [""]
