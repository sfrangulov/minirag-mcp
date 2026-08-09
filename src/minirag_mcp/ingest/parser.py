"""markitdown wrapper: files, HTML strings, URLs -> Markdown + title."""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from markitdown import MarkItDown
from requests.adapters import HTTPAdapter

from minirag_mcp import ocr
from minirag_mcp.config import Config
from minirag_mcp.security import SecurityError, check_url

logger = logging.getLogger(__name__)

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

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"})


def supported_extensions() -> frozenset[str]:
    """The scan whitelist. Images are documents only when OCR can read them —
    without it they would ingest as nothing, and most images in a docs tree
    are illustrations, so their absence is silence, not an error."""
    if ocr.available():
        return SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS
    return SUPPORTED_EXTENSIONS


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
# The same, restricted to an inlined `data:` URI — the picture itself, not a link to
# one — with the alt text captured so it can be kept.
_DATA_URI_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*data:[^)]*\)")
# A leftover extension in a stem, as "report.pdf.pdf" leaves "report.pdf". Built from
# the static union of both extension sets — not `supported_extensions()`, whose answer
# depends on `ocr.available()` and can change between calls; this regex is compiled once.
_EXTENSION_SUFFIX_RE = re.compile(
    r"\.(?:"
    + "|".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS))
    + r")$",
    re.IGNORECASE,
)
_SEPARATORS_RE = re.compile(r"[\s_\-.]+")
_SPACE_RE = re.compile(r"\s+")
_MIN_INFORMATIVE_STEM = 4

# The header markitdown puts on its own session; kept when we supply ours instead.
_ACCEPT_HEADER = "text/markdown, text/html;q=0.9, text/plain;q=0.8, */*;q=0.1"
# requests defaults to 30 redirects. Every hop is a fresh chance for a redirect to
# land somewhere the guard has to refuse, and no document needs more than a handful.
MAX_REDIRECTS = 5

_converters: dict[bool, MarkItDown] = {}


class ParserError(Exception):
    pass


class BlockedFetchError(SecurityError):
    """A URL the transport guard refused — the first request or any redirect hop.

    Carries the URL that was actually blocked, which is what lets `parse_url` say
    whether the refusal happened on the URL the caller asked for or on a redirect
    target it was sent to.
    """

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"{reason} (blocked URL: {url})")
        self.url = url
        self.reason = reason


@dataclass(frozen=True)
class ParsedDoc:
    markdown: str
    title: str
    # False when `title` is a stand-in identifier (the source URL, "Untitled") rather
    # than a title the document or the caller actually supplied. Callers that put the
    # title into indexed text must not do so for a stand-in.
    has_title: bool = True
    # Non-empty when OCR produced some or all of `markdown` (engine name).
    ocr_engine: str = ""


class _GuardedHTTPAdapter(HTTPAdapter):
    """A transport that re-runs the URL check on every request it is handed.

    Validating only the URL the caller supplied leaves the fetch itself unguarded: a
    permitted public host answering `302 -> http://169.254.169.254/` would have its
    redirect followed, and cloud-metadata content would reach the index. requests
    calls `send()` once per hop, so a check here necessarily sees every redirect
    target as well as the original URL.
    """

    def __init__(self, *, allow_private: bool, **kwargs) -> None:
        self._allow_private = allow_private
        super().__init__(**kwargs)

    def send(self, request, **kwargs):
        try:
            check_url(request.url, allow_private=self._allow_private)
        except SecurityError as e:
            raise BlockedFetchError(request.url, str(e)) from e
        return super().send(request, **kwargs)


def _guarded_session(allow_private: bool) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": _ACCEPT_HEADER})
    adapter = _GuardedHTTPAdapter(allow_private=allow_private)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.max_redirects = MAX_REDIRECTS
    return session


def _md(allow_private: bool = False) -> MarkItDown:
    """The converter, cached per host-rule setting.

    The guard is baked into the instance's session, so one cached converter would
    freeze whichever `allow_private` was in force when it was first built. Keying the
    cache on the flag keeps the answer current; the flag is a bool, so at most two
    converters ever exist. Callers that never fetch (files, HTML strings) take the
    default instance — its session is unused.
    """
    converter = _converters.get(allow_private)
    if converter is None:
        converter = MarkItDown(
            enable_plugins=False, requests_session=_guarded_session(allow_private)
        )
        _converters[allow_private] = converter
    return converter


def _lines_with_fence_state(markdown: str) -> Iterator[tuple[str, bool]]:
    """Every line, paired with whether it belongs to a fenced code block.

    The opening and closing fence lines count as inside: they are part of the block.
    """
    fence: str | None = None
    for line in markdown.split("\n"):
        stripped = line.lstrip()
        if fence is None:
            match = _FENCE_RE.match(stripped)
            if match:
                fence = match.group(1)
                yield line, True
            else:
                yield line, False
        else:
            yield line, True
            if stripped.startswith(fence[0] * len(fence)):
                fence = None


def _first_h1(markdown: str) -> str | None:
    """First ATX H1 outside fenced code blocks.

    A `# ` line inside a fence is code (a shell or Python comment), not a title.
    """
    for line, in_fence in _lines_with_fence_state(markdown):
        if in_fence:
            continue
        heading = _H1_RE.match(line)
        if heading:
            return heading.group(1).strip()
    return None


def strip_data_uri_images(markdown: str) -> str:
    """Drop embedded-image placeholders, keeping their alt text.

    markitdown renders an embedded picture as `![alt](data:image/png;base64,...)`:
    4442 of 52231 chunks on a measured 558-document corpus carried one, and they are
    noise in search results and in read_file output alike. Alt text is content the
    author wrote, so it survives as plain text; a placeholder without any goes whole.

    Only `data:` URIs are stripped — an image link to a path or an http URL is a
    reference the reader can follow. Fenced code blocks are left untouched, where a
    data: URI is the subject of the document rather than decoration. A document that
    is nothing but placeholders keeps them: removing decoration must not remove the
    document.
    """
    stripped = "\n".join(
        line if in_fence else _DATA_URI_IMAGE_RE.sub(lambda m: m.group(1).strip(), line)
        for line, in_fence in _lines_with_fence_state(markdown)
    )
    return markdown if markdown.strip() and not stripped.strip() else stripped


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


def _stem_title(stem: str) -> str:
    """The last-resort title, from a stem `_strip_extension_suffix` has already been
    applied to. The raw stem survives only when normalisation leaves nothing at all."""
    return _display_stem(stem) or stem


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
    return _stem_title(stem)


def _converted_markdown(result) -> str:
    """The converter's Markdown, cleaned. Every entry point goes through here."""
    return strip_data_uri_images(result.markdown or "")


def parse_file(path: Path, config: Config | None = None) -> ParsedDoc:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        if not ocr.available():
            raise ParserError(f"{path} is an image, which needs OCR; {ocr.INSTALL_HINT}")
        if config is None:
            raise ParserError(f"OCR for {path} needs a Config (internal error)")
        try:
            text = ocr.ocr_image(path, config)
        except ocr.OcrError as e:
            raise ParserError(str(e)) from e
        # Not `_file_title`: it prefers an authored ATX heading to the filename, and a
        # recognized page has none — a `# ` line in OCR output is scanner noise.
        title = _stem_title(_strip_extension_suffix(path.stem))
        return ParsedDoc(markdown=text, title=title, ocr_engine=ocr.ENGINE_RAPIDOCR)
    try:
        result = _md().convert(str(path))
    except Exception as e:  # markitdown may raise lib-specific errors too
        raise ParserError(f"Failed to convert {path}: {e}") from e
    markdown = _converted_markdown(result)
    ocr_engine = ""
    if config is not None and path.suffix.lower() == ".pdf":
        markdown, ocr_engine = _pdf_with_ocr_fallback(path, markdown, config)
    return ParsedDoc(
        markdown=markdown,
        title=_file_title(markdown, result.title, path.stem),
        ocr_engine=ocr_engine,
    )


def _pdf_with_ocr_fallback(path: Path, markdown: str, config: Config) -> tuple[str, str]:
    """Route near-empty PDF pages through OCR; keep text-layer pages as they are.

    Per-page, not whole-document: a text cover page over scanned pages must
    not mask them. Threshold 0 disables the check entirely.
    """
    threshold = config.ocr_min_chars_per_page
    if threshold <= 0:
        return markdown, ""
    if not ocr.available() and markdown.strip():
        # The per-page pass is a second full pdfminer parse of a file markitdown has
        # already parsed. With no OCR to route pages to, its only remaining use is
        # telling a file with no text at all from one with some, and a converter result
        # that carries text has answered that.
        return markdown, ""
    try:
        page_texts = ocr.pdf_page_texts(path)
    except Exception as e:
        # pdfminer failing on an exotic PDF must not break the markitdown result that
        # already succeeded, but a silently disabled OCR path must still be visible.
        logger.warning("pdf_page_texts failed for %s, OCR routing skipped: %s", path, e)
        return markdown, ""
    needs = [i for i, text in enumerate(page_texts) if len(text.strip()) < threshold]
    if not needs:
        return markdown, ""
    if not ocr.available():
        # Reached only with an empty conversion: the short-circuit above has already
        # returned every PDF markitdown got text out of, and that is what keeps a
        # certificate or a title page — a few real characters, under the threshold on
        # every page — from being refused. All that is left to tell apart here is a
        # file pdfminer still reads from one with no text layer at all.
        if len(needs) == len(page_texts):
            raise ParserError(
                f"{path} looks like a scanned PDF (no text layer); {ocr.INSTALL_HINT}"
            )
        return markdown, ""  # partial text beats refusal
    try:
        recognized = ocr.ocr_pdf(path, config, needs)
    except ocr.OcrError as e:
        raise ParserError(str(e)) from e
    # markitdown's markdown has no page boundaries to splice into: a scanned page's
    # recognized text is appended after the converted document rather than woven
    # back into page order, trading position for keeping the good pages' own
    # structure (tables, headings) intact instead of flattening the whole document
    # to raw per-page text the moment any single page needs OCR.
    ocr_text = "\n\n".join(text for i in needs if (text := recognized.get(i, "").strip()))
    if not ocr_text:
        # OCR finding nothing is only fatal when the converter found nothing either:
        # markitdown (pdfplumber) and pdfminer's per-page pass are separate
        # extractors and can disagree on an odd encoding or a degenerate text box,
        # so a markitdown result that already has content must survive even when
        # every page fell below the per-page threshold and OCR added nothing.
        if len(needs) == len(page_texts) and not markdown.strip():
            raise ParserError(f"{path}: OCR produced no text (page may be blank or unreadable)")
        return markdown, ""  # OCR added nothing; the converter already carried content
    merged = "\n\n".join(part for part in (markdown.strip(), ocr_text) if part)
    return merged, ocr.ENGINE_RAPIDOCR


def parse_html(html: str, title: str | None = None) -> ParsedDoc:
    try:
        result = _md().convert_stream(io.BytesIO(html.encode("utf-8")), file_extension=".html")
    except Exception as e:
        raise ParserError(f"Failed to convert HTML: {e}") from e
    markdown = _converted_markdown(result)
    found = find_title(markdown, title or result.title)
    return ParsedDoc(markdown=markdown, title=found or "Untitled", has_title=found is not None)


def _same_endpoint(a: str, b: str) -> bool:
    """Whether two URLs name the same host and port — chosen over a raw string
    compare because requests normalizes a URL before the guard ever sees it (a bare
    authority like "http://host" becomes "http://host/"), which a string compare would
    misreport as a redirect. Host and port are also the only part of the URL a
    redirect hop can actually change that matters here: where the request goes next.
    """
    pa, pb = urlparse(a), urlparse(b)
    return (pa.hostname, pa.port) == (pb.hostname, pb.port)


def parse_url(url: str, *, allow_private: bool = False) -> ParsedDoc:
    """Fetch and convert a URL, with the host rule enforced on every redirect hop.

    `allow_private` is the caller's current ALLOW_PRIVATE_URLS setting; it selects the
    converter whose session carries the matching guard.
    """
    try:
        result = _md(allow_private).convert_url(url)
    except BlockedFetchError as e:
        # The refusal must reach the user as a refusal, naming the host that was
        # blocked and — when it was not the URL they asked for — the redirect.
        if not _same_endpoint(e.url, url):
            raise ParserError(
                f"Refused to fetch {url}: it redirected to {e.url}, which this server "
                f"must not fetch. {e.reason}"
            ) from e
        raise ParserError(e.reason) from e
    except Exception as e:
        raise ParserError(f"Failed to fetch/convert {url}: {e}") from e
    markdown = _converted_markdown(result)
    found = find_title(markdown, result.title)
    return ParsedDoc(markdown=markdown, title=found or url, has_title=found is not None)
