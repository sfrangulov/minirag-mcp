"""PDF page numbers: where each page starts in a PDF's converted text, and which pages
each chunk cut from that text came from."""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTChar, LTContainer, LTItem
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

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
# How far past the end of the previous chunk a chunk is looked for before the whole
# text is searched, in non-whitespace characters, on top of four times its own length.
# What lies between two consecutive chunks is what neither of them carries: headings
# that went into a breadcrumb, a section that had nothing but a heading.
_WINDOW = 4096
# The shortest tail of a chunk that is taken as found. Shorter runs turn up by chance.
_MIN_RUN = 8


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


def _has_text(item: LTItem) -> bool:
    if isinstance(item, LTChar):
        return bool(item.get_text().strip())
    if isinstance(item, LTContainer):
        return any(_has_text(child) for child in item)
    return False


def text_pages(path: Path) -> list[int]:
    """The zero-based pages that carry a character other than whitespace.

    Characters only, without pdfminer's layout analysis: whether a page has text does
    not depend on how its lines are grouped, and the grouping is the expensive part.
    """
    rsrc = PDFResourceManager()
    device = PDFPageAggregator(rsrc, laparams=None)
    interpreter = PDFPageInterpreter(rsrc, device)
    out: list[int] = []
    with path.open("rb") as fh:
        for index, page in enumerate(PDFPage.get_pages(fh)):
            interpreter.process_page(page)
            if _has_text(device.get_result()):
                out.append(index)
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
    are joined by one. Neither route leaves anything for a page with no text, and
    markitdown's trailing-whitespace cleanup deletes a form feed whose line has nothing
    else on it, which is what a page holding only spaces leaves behind.

    So the separators are trusted outright only when every page boundary survived,
    and otherwise matched one to one against the pages that carry text.
    """
    if not markdown.strip():
        return ()
    count = page_count(path)
    if count == 1:
        return ((0, 1),)
    feeds = [m.end() for m in re.finditer("\f", markdown)]
    if len(feeds) == count - 1:
        return ((0, 1), *((at, page) for page, at in enumerate(feeds, start=2)))
    with_text = text_pages(path)
    # One page of text has no boundary to find, and through pdfminer it can hold any
    # number of blank lines.
    separator = _FEEDS if feeds or len(with_text) == 1 else _BLANK_LINES
    segments: list[int] = []
    at = 0
    for m in separator.finditer(markdown):
        if markdown[at : m.start()].strip():
            segments.append(at)
        at = m.end()
    if markdown[at:].strip():
        segments.append(at)
    if len(segments) != len(with_text):
        logger.warning(
            "no page numbers for %s: %d text block(s) for %d page(s) with text",
            path,
            len(segments),
            len(with_text),
        )
        return ()
    return tuple((at, index + 1) for at, index in zip(segments, with_text, strict=True))


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


def _label(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}-{last}"


def chunk_pages(markdown: str, starts: PageStarts, texts: Sequence[str]) -> list[str]:
    """The pages each of `texts`, chunks cut in order from `markdown`, came from: "12",
    "12-13", or "" where that is not known.

    The chunker changes nothing in a chunk's own text but whitespace, and puts any
    context it repeats — the document title, a heading breadcrumb, a table's header
    row — in front of it. So with whitespace removed, the end of every chunk is a run
    of the document's own characters, and it is looked for from where the chunk before
    it ended. A chunk whose own text is too short to find that way lies between its
    neighbours, and gets the pages of what lies between them.
    """
    if not starts:
        return [""] * len(texts)
    flat = "".join(markdown.split())
    keys: list[int] = []
    numbers: list[int] = []
    previous = consumed = 0
    for offset, page in sorted(starts, key=lambda s: s[0]):
        consumed += len("".join(markdown[previous:offset].split()))
        previous = offset
        keys.append(consumed)
        numbers.append(page)

    def label(lo: int, hi: int) -> str:
        if lo >= hi:
            return ""
        first = bisect_right(keys, lo) - 1
        last = bisect_right(keys, hi - 1) - 1
        if first < 0:
            return ""
        touched = numbers[first : last + 1]
        if UNKNOWN in touched:
            return ""
        return _label(min(touched), max(touched))

    spans: list[tuple[int, int] | None] = []
    before: list[int] = []
    cursor = 0
    for text in texts:
        before.append(cursor)
        span = _locate("".join(text.split()), flat, cursor)
        spans.append(span)
        if span is not None:
            cursor = max(cursor, span[1])
    out: list[str] = []
    for i, span in enumerate(spans):
        if span is None:
            after = next((s[0] for s in spans[i + 1 :] if s is not None), len(flat))
            span = (before[i], after)
        out.append(label(*span))
    return out


def _locate(own: str, flat: str, cursor: int) -> tuple[int, int] | None:
    """Where in `flat` the longest tail of `own` is, looked for from `cursor` on."""
    if not own:
        return None
    end = min(len(flat), cursor + _WINDOW + 4 * len(own))
    at = flat.find(own, cursor, end)
    if at >= 0:
        return at, at + len(own)
    best = None
    lo, hi = _MIN_RUN, len(own) - 1
    while lo <= hi:
        size = (lo + hi) // 2
        at = flat.find(own[-size:], cursor, end)
        if at >= 0:
            best, lo = (at, at + size), size + 1
        else:
            hi = size - 1
    if best is not None:
        return best
    at = flat.find(own, cursor)
    if at < 0:
        at = flat.find(own)
    return (at, at + len(own)) if at >= 0 else None
