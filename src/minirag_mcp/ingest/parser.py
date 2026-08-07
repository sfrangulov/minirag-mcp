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

# Filenames that say nothing about the document. Compared against the stem with
# separators stripped, so "read_me" and "New Copy" collapse onto these too.
_GENERIC_STEMS = frozenset(
    {"untitled", "document", "doc", "index", "readme", "new", "copy", "scan", "image"}
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
    all. Machine-issued names ("scan", "IMG", "20260807") say nothing, so they must
    not displace a real heading. A stem is informative unless it is too short to
    carry meaning, purely numeric once separators drop out, or one of a few generic
    names — everything else is assumed to be deliberate.
    """
    bare = _SEPARATORS_RE.sub("", stem).strip()
    if len(bare) < _MIN_INFORMATIVE_STEM:
        return False
    if bare.isdigit():
        return False
    return bare.casefold() not in _GENERIC_STEMS


def _display_stem(stem: str) -> str:
    """Underscores to spaces, whitespace collapsed; every other character kept.

    These filenames carry meaningful punctuation ("И-112_ЗПС_..." → "И-112 ЗПС ...");
    stripping it would destroy the document code the name exists to convey.
    """
    return _SPACE_RE.sub(" ", stem.replace("_", " ")).strip()


def extract_title(markdown: str, explicit: str | None, fallback: str) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    h1 = _first_h1(markdown)
    if h1:
        return h1
    return fallback


def _file_title(markdown: str, explicit: str | None, stem: str) -> str:
    """Title for a file: converter metadata → informative filename → first H1 → stem.

    The filename outranks the H1 because office documents open with boilerplate —
    every one of them starts "1. Общие положения" — while the filename names the
    document. markitdown supplies real metadata only for formats that carry it
    (HTML, EPUB); when it does, that beats the filename.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    if _is_informative_stem(stem):
        return _display_stem(stem)
    h1 = _first_h1(markdown)
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
    return ParsedDoc(
        markdown=markdown, title=extract_title(markdown, title or result.title, "Untitled")
    )


def parse_url(url: str) -> ParsedDoc:
    try:
        result = _md().convert_url(url)
    except Exception as e:
        raise ParserError(f"Failed to fetch/convert {url}: {e}") from e
    markdown = result.markdown or ""
    return ParsedDoc(markdown=markdown, title=extract_title(markdown, result.title, url))
