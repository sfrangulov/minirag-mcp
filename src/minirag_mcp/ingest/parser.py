"""markitdown wrapper: files, HTML strings, URLs -> Markdown + title."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from markitdown import MarkItDown

SUPPORTED_EXTENSIONS = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".csv",
        ".epub",
        ".ipynb",
    }
)

_H1_RE = re.compile(r"^#\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^(```+|~~~+)")

# Words that say nothing about a document: the names cameras, scanners, office suites
# and file managers hand out ("Untitled 1", "IMG_20260807", "Копия документа (2)"), the
# generic document kinds, and the joiners that glue them together ("Copy of Report").
# Matched per token, so a real name that merely uses one of these words ("Document
# Management Policy") is still informative.
_GENERIC_TOKENS = frozenset(
    {
        "untitled",
        "document",
        "documents",
        "doc",
        "docs",
        "docx",
        "index",
        "readme",
        "new",
        "copy",
        "of",
        "scan",
        "image",
        "img",
        "file",
        "dsc",
        "screenshot",
        "report",
        # Russian equivalents, with the inflected forms filenames actually carry
        "снимок",
        "документ",
        "документа",
        "документов",
        "копия",
        "копии",
        "безымянный",
    }
)
# Section headings whole document sets share. Measured on a 558-document corpus: 76
# documents opened with "1 Общие положения" and 69 with "Лист изменений", so the first
# H1 named the section, not the document.
_BOILERPLATE_HEADINGS = frozenset(
    {
        "общие положения",
        "назначение",
        "введение",
        "назначение и область применения",
        "лист изменений",
        "история изменений",
        "содержание",
        "оглавление",
        "термины и определения",
        "список сокращений",
        "general provisions",
        "introduction",
        "overview",
        "purpose",
        "scope",
        "table of contents",
        "contents",
        "revision history",
        "change log",
        "definitions",
        "abbreviations",
    }
)
# Word tokens: a run of letters or a run of digits, so "Document1" splits into
# "document" + "1" and "DSC00042" into "dsc" + "00042".
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+")
# One level of section numbering: "1.", "2)", "IV." — arabic or roman, and only when a
# separator follows, so "introduction" is never read as the roman numeral "i".
_SECTION_NUMBER_RE = re.compile(r"^(?:\d+|[ivxlcdm]+)(?:[.)]\s*|\s+)")
# Markdown emphasis and punctuation wrapping a heading: "**1. Общие положения**"
_HEADING_TRIM = "*_`~#[]() \t.:;,!-–—"
# An inline image, which markitdown emits for a heading that is only a picture
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# A leftover extension in a stem, as "report.pdf.pdf" leaves "report.pdf"
_EXTENSION_SUFFIX_RE = re.compile(
    r"\.(?:" + "|".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS)) + r")$",
    re.IGNORECASE,
)
_SEPARATORS_RE = re.compile(r"[\s_\-.]+")
_SPACE_RE = re.compile(r"\s+")
_MIN_INFORMATIVE_STEM = 4

_converter: MarkItDown | None = None


class ParserError(Exception):
    pass


@dataclass(frozen=True)
class ParsedDoc:
    markdown: str
    title: str
    # False when `title` is a stand-in identifier (the source URL, "Untitled") rather
    # than a title the document or the caller actually supplied. Callers that put the
    # title into indexed text must not do so for a stand-in.
    has_title: bool = True


def _md() -> MarkItDown:
    global _converter
    if _converter is None:
        _converter = MarkItDown(enable_plugins=False)
    return _converter


def _first_h1(markdown: str) -> str | None:
    """First ATX H1 outside fenced code blocks.

    A `# ` line inside a fence is code (a shell or Python comment), not a title.
    """
    fence: str | None = None
    for line in markdown.split("\n"):
        stripped = line.lstrip()
        if fence is None:
            match = _FENCE_RE.match(stripped)
            if match:
                fence = match.group(1)
                continue
            heading = _H1_RE.match(line)
            if heading:
                return heading.group(1).strip()
        elif stripped.startswith(fence[0] * len(fence)):
            fence = None
    return None


def _is_informative_stem(stem: str) -> bool:
    """Whether a filename stem is worth showing as the document's title.

    Real document sets name files after their subject ("И-112_ЗПС_Хранение ТМЗ на
    складах"), which is often the only place the document code and topic appear at
    all. Machine-issued names ("scan_001", "IMG_20260807_123456", "Копия документа
    (2)") say nothing, so they must not stand in for a heading.

    The judgement is per token, not on the whole string: "New Document" and
    "Document1" are as machine-issued as "document", while "Document Management
    Policy" is a real name that merely uses the word. A stem is uninformative when it
    is too short to carry meaning, or when — once pure-digit tokens (counters, dates,
    sequence numbers) are dropped — nothing is left but generic words.
    """
    bare = _SEPARATORS_RE.sub("", stem).strip()
    if len(bare) < _MIN_INFORMATIVE_STEM:
        return False
    tokens = [t for t in _TOKEN_RE.findall(stem.casefold()) if not t.isdigit()]
    return any(t not in _GENERIC_TOKENS for t in tokens)


def _is_boilerplate_heading(heading: str) -> bool:
    """Whether a heading names a boilerplate section rather than the document.

    Office document sets open with the same first section — "1. Общие положения",
    "Лист изменений" — so that H1 identifies the section, not the document, and is
    identical across the whole set. The match is on the whole normalised heading, so
    "Общие положения о премировании работников" (a real title) is left alone.
    """
    text = _SPACE_RE.sub(" ", heading.casefold()).strip(_HEADING_TRIM)
    while True:
        stripped = _SECTION_NUMBER_RE.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return text.strip(_HEADING_TRIM) in _BOILERPLATE_HEADINGS


def _is_usable_heading(heading: str) -> bool:
    """Whether an H1 can stand as the document's title.

    Two headings cannot: a boilerplate section name, which the whole document set
    shares, and a heading with no words in it at all — markitdown renders a heading
    that holds only a picture as "![](data:image/x-emf;base64...)", which names
    nothing (5 documents in the measured corpus).
    """
    if _is_boilerplate_heading(heading):
        return False
    return bool(_TOKEN_RE.search(_IMAGE_RE.sub("", heading)))


def _strip_extension_suffix(stem: str) -> str:
    """Drop an extension the stem still carries: "report.pdf.pdf" → "report.pdf" → "report"."""
    return _EXTENSION_SUFFIX_RE.sub("", stem) or stem


def _display_stem(stem: str) -> str:
    """Underscores to spaces, whitespace collapsed; every other character kept.

    These filenames carry meaningful punctuation ("И-112_ЗПС_..." → "И-112 ЗПС ...");
    stripping it would destroy the document code the name exists to convey.
    """
    return _SPACE_RE.sub(" ", stem.replace("_", " ")).strip()


def find_title(markdown: str, explicit: str | None) -> str | None:
    """The title the caller gave or the document carries — None when it has neither.

    Distinguishing "no title" from "a fallback title" is what lets callers avoid
    treating a source id or a URL as if the document were named that.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    return _first_h1(markdown)


def extract_title(markdown: str, explicit: str | None, fallback: str) -> str:
    return find_title(markdown, explicit) or fallback


def _file_title(markdown: str, explicit: str | None, stem: str) -> str:
    """Title for a file: converter metadata → a usable H1 → informative filename stem
    → first H1 → stem.

    A heading the author wrote is the best title available, so it wins by default; see
    `_is_usable_heading` for the two kinds that cannot. The exception is the measured
    one: office document sets share a boilerplate first section ("1. Общие положения"),
    and there the filename is what names the document. markitdown supplies real
    metadata only for formats that carry it (HTML, EPUB); when it does, that beats both.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    stem = _strip_extension_suffix(stem)
    h1 = _first_h1(markdown)
    if h1 and _is_usable_heading(h1):
        return h1
    if _is_informative_stem(stem):
        return _display_stem(stem)
    if h1:
        return h1
    return _display_stem(stem) or stem


def parse_file(path: Path) -> ParsedDoc:
    try:
        result = _md().convert(str(path))
    except Exception as e:  # markitdown may raise lib-specific errors too
        raise ParserError(f"Failed to convert {path}: {e}") from e
    markdown = result.markdown or ""
    return ParsedDoc(markdown=markdown, title=_file_title(markdown, result.title, path.stem))


def parse_html(html: str, title: str | None = None) -> ParsedDoc:
    try:
        result = _md().convert_stream(io.BytesIO(html.encode("utf-8")), file_extension=".html")
    except Exception as e:
        raise ParserError(f"Failed to convert HTML: {e}") from e
    markdown = result.markdown or ""
    found = find_title(markdown, title or result.title)
    return ParsedDoc(markdown=markdown, title=found or "Untitled", has_title=found is not None)


def parse_url(url: str) -> ParsedDoc:
    try:
        result = _md().convert_url(url)
    except Exception as e:
        raise ParserError(f"Failed to fetch/convert {url}: {e}") from e
    markdown = result.markdown or ""
    found = find_title(markdown, result.title)
    return ParsedDoc(markdown=markdown, title=found or url, has_title=found is not None)
