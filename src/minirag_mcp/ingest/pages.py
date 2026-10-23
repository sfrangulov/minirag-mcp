"""PDF page numbers: where each page starts in a PDF's converted text, and which pages
each chunk cut from that text came from.

A wrong page is worse than none: a reader sent to page 12 for a passage on page 40
stops trusting every page after it. Wherever the evidence is short of certain, the
answer is "no page".
"""

from __future__ import annotations

import logging
import re
import unicodedata
from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTChar, LTContainer, LTItem
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from minirag_mcp.chunker import Chunk, in_document_order

if TYPE_CHECKING:
    from minirag_mcp.store import SourceInfo

logger = logging.getLogger(__name__)

# (offset, page) pairs in offset order: the text from `offset` up to the next pair is on
# `page`, counted from 1. Text before the first pair, or under a pair whose page is
# UNKNOWN, has no known page.
PageStarts = tuple[tuple[int, int], ...]
UNKNOWN = 0

_FEEDS = re.compile(r"\f+")
_BLANK_LINES = re.compile(r"\n{2,}")
# Characters markitdown adds to a page's text: the pipes and dashes of a Markdown table.
_TABLE_MARKUP = frozenset("|-")
# How far past the end of the chunk before it a chunk's own text may start, in
# non-whitespace characters, on top of four times its own length and the length of any
# chunks in between that could not be placed. What legitimately lies between two
# consecutive chunks is what neither of them holds as its own: headings carried in a
# breadcrumb, a table's header row.
_WINDOW = 4096


def predates_pages(info: SourceInfo) -> bool:
    """A PDF file indexed before page numbers were stored: none of its results carry
    one until it is read again."""
    return (
        info.source_type == "file"
        and info.source.lower().endswith(".pdf")
        and not info.pages_recorded
    )


def page_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in PDFPage.get_pages(fh))


def _chars(item: LTItem):
    if isinstance(item, LTChar):
        yield item.get_text()
    elif isinstance(item, LTContainer):
        for child in item:
            yield from _chars(child)


def _print(texts) -> Counter[str]:
    """What a page's text is made of, whatever order it was read in.

    Decomposed rather than composed, so a combining mark counts the same whichever
    character an extractor's reading order put in front of it; compatibility
    decomposition, so a ligature pdfplumber expands counts the same as the glyph.
    """
    return Counter(
        ch
        for text in texts
        for ch in unicodedata.normalize("NFKD", text)
        if not ch.isspace() and ch not in _TABLE_MARKUP
    )


def page_prints(path: Path) -> list[Counter[str]]:
    """Every page's characters, as `_print` counts them; empty for a page with no text.

    Characters only, without pdfminer's layout analysis: what a page holds does not
    depend on how its lines are grouped, and the grouping is the expensive part.
    """
    rsrc = PDFResourceManager()
    device = PDFPageAggregator(rsrc, laparams=None)
    interpreter = PDFPageInterpreter(rsrc, device)
    out: list[Counter[str]] = []
    with path.open("rb") as fh:
        for page in PDFPage.get_pages(fh):
            interpreter.process_page(page)
            out.append(_print(_chars(device.get_result())))
    return out


def pdf_page_starts(path: Path, markdown: str) -> PageStarts:
    """Where each page of the PDF at `path` starts in `markdown`, its markitdown text.

    () when that cannot be told for certain: the document is then stored without page
    numbers rather than with wrong ones.
    """
    try:
        return _pdf_page_starts(path, markdown)
    except Exception as e:  # pdfminer on an odd file; the conversion itself succeeded
        logger.warning("no page numbers for %s: %s", path, e)
        return ()


def _pdf_page_starts(path: Path, markdown: str) -> PageStarts:
    """markitdown converts a PDF one of two ways. Prose goes through pdfminer, which
    ends every page with a form feed. A document with a table on any page goes through
    pdfplumber a page at a time, and the pages, which hold no blank line of their own,
    are joined by one. Neither route leaves anything for a page with no text. And the
    route's own clean-up moves text across those boundaries: trailing whitespace is
    stripped from every line, which deletes a form feed whose line holds only spaces and
    turns a table row of no-break spaces into a blank line, and a lone `.1` at the foot
    of a page is joined to the first line of the next.

    So the separators are trusted outright only when every page boundary survived, and
    otherwise each block between them must hold exactly the characters of the page it
    is matched to.
    """
    if not markdown.strip():
        return ()
    count = page_count(path)
    if count == 1:
        return ((0, 1),)
    feeds = [m.end() for m in re.finditer("\f", markdown)]
    if len(feeds) == count - 1:
        return ((0, 1), *((at, page) for page, at in enumerate(feeds, start=2)))
    prints = page_prints(path)
    with_text = [index for index, chars in enumerate(prints) if chars]
    # One page of text has no boundary to find, and through pdfminer it can hold any
    # number of blank lines.
    separator = _FEEDS if feeds or len(with_text) == 1 else _BLANK_LINES
    blocks: list[tuple[int, int]] = []
    at = 0
    for m in separator.finditer(markdown):
        if markdown[at : m.start()].strip():
            blocks.append((at, m.start()))
        at = m.end()
    if markdown[at:].strip():
        blocks.append((at, len(markdown)))
    if len(blocks) != len(with_text):
        logger.warning(
            "no page numbers for %s: %d text block(s) for %d page(s) with text",
            path,
            len(blocks),
            len(with_text),
        )
        return ()
    for (start, end), index in zip(blocks, with_text, strict=True):
        if _print([markdown[start:end]]) != prints[index]:
            logger.warning(
                "no page numbers for %s: the text matched to page %d is not that page's",
                path,
                index + 1,
            )
            return ()
    return tuple((start, index + 1) for (start, _), index in zip(blocks, with_text, strict=True))


def stripped(text: str, starts: PageStarts) -> PageStarts:
    """`starts` for `text.strip()`."""
    shift = len(text) - len(text.lstrip())
    end = len(text.strip())
    return tuple((max(0, at - shift), page) for at, page in starts if at - shift < end)


def joined(parts: Sequence[tuple[str, PageStarts]], sep: str = "\n\n") -> tuple[str, PageStarts]:
    """Texts joined by `sep`, each carrying its own page starts along.

    A part with no starts of its own is marked UNKNOWN, so it does not read as the page
    of the part before it.
    """
    out: list[tuple[int, int]] = []
    at = 0
    for text, starts in parts:
        if not starts or starts[0][0] > 0:
            out.append((at, UNKNOWN))
        out += [(at + offset, page) for offset, page in starts]
        at += len(text) + len(sep)
    return sep.join(text for text, _ in parts), tuple(out)


class _Flat:
    """`markdown` without its whitespace, with where its words begin and end, and which
    page each character is on."""

    def __init__(self, markdown: str, starts: PageStarts) -> None:
        words = markdown.split()
        self.text = "".join(words)
        self.edges = bytearray(len(self.text) + 1)
        self.edges[0] = 1
        at = 0
        for word in words:
            at += len(word)
            self.edges[at] = 1
        self.keys: list[int] = []
        self.numbers: list[int] = []
        previous = consumed = 0
        for offset, page in sorted(starts, key=lambda s: s[0]):
            consumed += len("".join(markdown[previous:offset].split()))
            previous = offset
            self.keys.append(consumed)
            self.numbers.append(page)

    def find(self, own: str, lo: int, hi: int) -> int:
        """The first place in [lo, hi) where `own` starts and ends on a word boundary."""
        at = self.text.find(own, lo, hi)
        while at >= 0 and not (self.edges[at] and self.edges[at + len(own)]):
            at = self.text.find(own, at + 1, hi)
        return at

    def label(self, lo: int, hi: int) -> str:
        first = bisect_right(self.keys, lo) - 1
        last = bisect_right(self.keys, hi - 1) - 1
        if first < 0:
            return ""
        touched = self.numbers[first : last + 1]
        if UNKNOWN in touched:
            return ""
        low, high = min(touched), max(touched)
        return str(low) if low == high else f"{low}-{high}"


def chunk_pages(markdown: str, starts: PageStarts, chunks: Sequence[Chunk]) -> list[str]:
    """The pages each of `chunks`, cut from `markdown`, came from: "12", "12-13", or ""
    where that is not certain.

    A chunk's own text is the document's text but for whitespace, so it is looked for
    exactly, as whole words. Chunks normally follow the document, so each is looked for
    from where the one before it ended, and no further than a chunk can have drifted. A
    transcript whose clock goes back is cut out of order; there a chunk is placed only
    where its text occurs once in the whole document.
    """
    if not starts:
        return [""] * len(chunks)
    flat = _Flat(markdown, starts)
    ordered = in_document_order(markdown)
    out: list[str] = []
    cursor = 0
    unplaced = 0
    for chunk in chunks:
        own = "".join(chunk.text[chunk.own_start :].split())
        if not own:
            out.append("")
            continue
        if ordered:
            end = min(len(flat.text), cursor + _WINDOW + 4 * len(own) + unplaced)
            at = flat.find(own, cursor, end)
        else:
            at = flat.find(own, 0, len(flat.text))
            if at >= 0 and flat.find(own, at + 1, len(flat.text)) >= 0:
                at = -1
        if at < 0:
            out.append("")
            unplaced += len(own)
            continue
        out.append(flat.label(at, at + len(own)))
        if ordered:
            cursor = at + len(own)
            unplaced = 0
    return out
