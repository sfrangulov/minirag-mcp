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


def extract_title(markdown: str, explicit: str | None, fallback: str) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    h1 = _first_h1(markdown)
    if h1:
        return h1
    return fallback


def parse_file(path: Path) -> ParsedDoc:
    try:
        result = _md().convert(str(path))
    except Exception as e:  # markitdown may raise lib-specific errors too
        raise ParserError(f"Failed to convert {path}: {e}") from e
    markdown = result.markdown or ""
    return ParsedDoc(markdown=markdown, title=extract_title(markdown, result.title, path.stem))


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
