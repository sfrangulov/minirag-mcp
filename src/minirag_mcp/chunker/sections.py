"""Structure-first sectioning: what a document's *parent sections* are.

A section is the unit a search hit is returned inside — a transcript time window, a
heading section, a slide, a paragraph. Each one is packed into one or more retrieval
units downstream (see `pack.py`); the units of a section share a `parent_id` so the
section can be rebuilt without storing its text twice.

The category is read off the converted Markdown, never off the file extension: the
same `.docx` extension covers meeting transcripts, numbered specifications and
instructions, and markitdown turns `.xlsx` and `.pptx` into ordinary Markdown too.
Detection is fail-safe — anything that does not clearly match a category falls to
`GENERIC`, which is the old structural splitter under a token budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from minirag_mcp.chunker.structural import split_markdown

TRANSCRIPT = "transcript"
SLIDES = "slides"
HEADINGS = "headings"
GENERIC = "generic"

# A transcript turn's timestamp: alone on its line, optionally preceded by a bolded
# speaker name (`**Dana Marx** 0:03`), which is the shape Teams exports use.
TIMESTAMP_LINE = re.compile(
    r"^[ \t]*(?:\*\*[^*\n]{1,80}\*\*[ \t]+)?((?:\d{1,2}:)?\d{1,2}:\d{2})[ \t]*$"
)
SLIDE_MARKER = re.compile(r"^[ \t]*<!--\s*Slide number:\s*(\d+)\s*-->[ \t]*$")
HEADING_LINE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*$")
FENCE = re.compile(r"^(```+|~~~+)")
TABLE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
TABLE_DELIMITER = re.compile(r"^[ \t]*\|(?:[ \t]*:?-+:?[ \t]*\|)+[ \t]*$")

# Measured against the live 558-document corpus: every one of the 107 real transcripts
# has 50.0%-51.7% of its non-blank lines matching TIMESTAMP_LINE (timestamps and speech
# strictly alternate), and every one of the other 452 documents has exactly 0.0%. The
# threshold sits in the middle of an empty gap, so it is not a tuned number — anything
# from 0.01 to 0.49 gives the identical partition. The floor on the absolute count stops
# a three-line document from being called a transcript on one stray `12:30`.
TRANSCRIPT_MIN_SHARE = 0.15
TRANSCRIPT_MIN_LINES = 5
TRANSCRIPT_WINDOW_SECONDS = 120
# Two markers, so a document that merely quotes the marker once is not a deck.
SLIDES_MIN_MARKERS = 2
HEADINGS_MIN = 2
# Cap on a heading section's size. Sections are what a caller gets back, and an
# unbounded one would hand back a whole 40-page chapter for a one-line hit. Chosen as
# roughly ten retrieval units of prose at the default budget.
SECTION_MAX_CHARS = 4000
_BREADCRUMB_JOIN = " > "
# Markdown emphasis and numbering punctuation wrapped around heading text
_HEADING_TRIM = "*_`~ \t"


@dataclass(frozen=True)
class Section:
    """One parent section.

    `prefixes` are context labels to put in front of every retrieval unit of this
    section, most informative first: the packer takes the first one that still leaves
    room for text. They are ordered so the fallback degrades gracefully (drop the
    document title, keep the timestamp) and always end with `""`.
    """

    prefixes: tuple[str, ...]
    body: str
    # heading chain, used only to drop a heading that has descendants but no text
    chain: tuple[str, ...] = ()


def _iter_lines(markdown: str):
    """Yield (line, inside_fenced_code) so headings inside code are not headings."""
    open_seq: str | None = None
    for line in markdown.split("\n"):
        fence = FENCE.match(line.lstrip())
        if open_seq is None:
            if fence:
                open_seq = fence.group(1)
                yield line, True
                continue
            yield line, False
        else:
            yield line, True
            stripped = line.lstrip()
            if stripped.startswith(open_seq[0]):
                run = len(stripped) - len(stripped.lstrip(open_seq[0]))
                if run >= len(open_seq):
                    open_seq = None


def detect_category(markdown: str) -> str:
    """Which structure this Markdown has. Unsure means GENERIC."""
    lines = list(_iter_lines(markdown))
    non_blank = [ln for ln, _ in lines if ln.strip()]
    if non_blank:
        stamps = sum(1 for ln in non_blank if TIMESTAMP_LINE.match(ln))
        if stamps >= TRANSCRIPT_MIN_LINES and stamps / len(non_blank) >= TRANSCRIPT_MIN_SHARE:
            return TRANSCRIPT
    if sum(1 for ln, _ in lines if SLIDE_MARKER.match(ln)) >= SLIDES_MIN_MARKERS:
        return SLIDES
    if sum(1 for ln, code in lines if not code and HEADING_LINE.match(ln)) >= HEADINGS_MIN:
        return HEADINGS
    return GENERIC


def _parse_timestamp(text: str) -> int:
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def format_timestamp(seconds: int) -> str:
    """`MM:SS` under an hour, `H:MM:SS` beyond — the notation the transcripts use.

    The design sketch wrote `HH:MM`, which cannot express a 120-second boundary: at
    minute resolution a window from 0 to 120 seconds and one from 0 to 2 seconds both
    render as `00:00-00:02`. Matching the source document's own notation also lets a
    reader take a label straight back to the recording.
    """
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _window_label(start: int, window: int) -> str:
    return f"[{format_timestamp(start)}–{format_timestamp(start + window)}]"


def _transcript_sections(markdown: str, title: str, window: int) -> list[Section]:
    """Group turns into fixed time windows.

    Turns are never split on the speaker: markitdown drops the speaker name in all but
    two of the corpus's transcripts, and the median turn is 90 characters — far too
    small to be a retrieval unit and far too small for the encoder to judge.
    """
    preamble: list[str] = []
    windows: dict[int, list[str]] = {}
    order: list[int] = []
    current: int | None = None
    for line in markdown.split("\n"):
        match = TIMESTAMP_LINE.match(line)
        if match:
            current = _parse_timestamp(match.group(1)) // window * window
            if current not in windows:
                windows[current] = []
                order.append(current)
        if current is None:
            preamble.append(line)
        else:
            windows[current].append(line)

    sections: list[Section] = []
    head = "\n".join(preamble).strip()
    if head:
        sections.append(Section(prefixes=("",), body=head))
    for start in order:
        body = "\n".join(windows[start]).strip()
        if not body:
            continue
        label = _window_label(start, window)
        prefixes = (f"{label} {title}".strip(), label, "")
        sections.append(Section(prefixes=prefixes, body=body))
    return sections


def _slide_sections(markdown: str) -> list[Section]:
    sections: list[Section] = []
    number: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        prefixes = (f"[Slide {number}]", "") if number is not None else ("",)
        sections.append(Section(prefixes=prefixes, body=body))

    for line in markdown.split("\n"):
        match = SLIDE_MARKER.match(line)
        if match:
            flush()
            number = match.group(1)
            continue
        buffer.append(line)
    flush()
    return sections


def clean_heading(text: str) -> str:
    """Heading text without its markup, for use in a breadcrumb.

    Corpus headings arrive as `# **2** **Общая схема...**` — markitdown keeps Word's
    bold runs — so the emphasis markers have to come off before the pieces are joined.
    """
    return re.sub(r"\s+", " ", text.replace("**", " ").replace("__", " ")).strip(_HEADING_TRIM)


def _split_oversized(body: str, limit: int) -> list[str]:
    """Cut a section body into `limit`-sized pieces at paragraph boundaries."""
    if len(body) <= limit:
        return [body]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for para in body.split("\n\n"):
        add = len(para) + (2 if current else 0)
        if current and size + add > limit:
            pieces.append("\n\n".join(current))
            current, size = [para], len(para)
        else:
            current.append(para)
            size += add
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _heading_sections(markdown: str) -> list[Section]:
    """One section per ATX heading, carrying the heading breadcrumb as its prefix."""
    raw: list[Section] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        chain = tuple(text for _, text in stack)
        full = _BREADCRUMB_JOIN.join(chain)
        if not chain:
            raw.append(Section(prefixes=("",), body=body, chain=chain))
            return
        # Degrade from the whole breadcrumb to the innermost heading alone: a deep
        # breadcrumb can be longer than the text it introduces.
        prefixes = tuple(dict.fromkeys((full, chain[-1], "")))
        raw.append(Section(prefixes=prefixes, body=body, chain=chain))

    for line, in_code in _iter_lines(markdown):
        match = None if in_code else HEADING_LINE.match(line)
        if match is None:
            buffer.append(line)
            continue
        flush()
        level, text = len(match.group(1)), clean_heading(match.group(2))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text or line.strip()))
    flush()

    kept: list[Section] = []
    for i, section in enumerate(raw):
        if not section.body.strip():
            # A heading with no text of its own is dropped when a nested heading
            # follows it, because its words survive inside that child's breadcrumb.
            # A childless empty heading is kept, so no heading text is ever lost.
            nxt = raw[i + 1] if i + 1 < len(raw) else None
            if nxt is not None and nxt.chain[: len(section.chain)] == section.chain:
                continue
            if not section.chain:
                continue
        for piece in _split_oversized(section.body, SECTION_MAX_CHARS):
            kept.append(Section(prefixes=section.prefixes, body=piece, chain=section.chain))
    return kept


def _generic_sections(markdown: str) -> list[Section]:
    """One section per structural block: the old splitter, without a character cap."""
    return [
        Section(prefixes=("",), body=block.text)
        for block in split_markdown(markdown, max_chars=None)
        if block.text.strip()
    ]


def split_sections(
    markdown: str,
    *,
    title: str = "",
    category: str | None = None,
    window: int = TRANSCRIPT_WINDOW_SECONDS,
) -> list[Section]:
    """Split Markdown into parent sections, choosing the strategy by detected shape."""
    category = category or detect_category(markdown)
    if category == TRANSCRIPT:
        sections = _transcript_sections(markdown, title, window)
    elif category == SLIDES:
        sections = _slide_sections(markdown)
    elif category == HEADINGS:
        sections = _heading_sections(markdown)
    else:
        sections = _generic_sections(markdown)
    sections = [s for s in sections if s.body.strip()]
    # Every strategy above can legitimately come up empty on odd input; falling back
    # here is what makes the whole detection step safe to be wrong.
    if not sections and markdown.strip():
        sections = _generic_sections(markdown)
    return sections
