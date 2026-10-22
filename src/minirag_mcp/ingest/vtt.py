"""WebVTT -> Markdown turns, in the shape the transcript sectioner recognises.

Adapted from webvtt-to-markdown 0.9.0 by GreenJesterAt
(https://github.com/GreenJesterAt/webvtt-to-markdown), Apache License 2.0 — see
LICENSES/webvtt-to-markdown.txt. Modified: its per-tool generation sources and their
scoring are folded into one speaker rule and one grouping rule; a turn is headed
`**Speaker** MM:SS` instead of `**(HH:MM:SS) Speaker**`; inline markup is dropped
instead of rendered; an unclosed voice tag and a header block carrying metadata lines
are accepted; a cue outside a Teams utterance keeps its place; decoding stops at
charset detection instead of falling back to fixed code pages.
"""

from __future__ import annotations

import codecs
import html
import re
from collections.abc import Iterator
from dataclasses import dataclass

from charset_normalizer import from_bytes

from minirag_mcp.chunker.sections import TRANSCRIPT_WINDOW_SECONDS, format_timestamp

_BOMS = (
    (codecs.BOM_UTF8, "utf-8"),
    # UTF-32-LE's mark begins with UTF-16-LE's, so it has to be tried first.
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)
# Older Teams and Stream exports leave the fields unpadded: `0:2:7.5 --> 0:2:9.0`.
_TIMING_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})\.\d{1,3}\s+-->\s+\S")
_VOICE_RE = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>]+)>")
# Only the tags WebVTT defines, and its inline timestamps: a `<` that opens none of
# them is text a generator failed to escape (`ставка <5% и >3%`).
_TAG_RE = re.compile(r"</?(?:[bciuv]|lang|ruby|rt)\b[^>\n]*>|<\d+(?::\d+)+\.\d+>")
# Teams numbers each utterance and every segment it splits one into:
# `<meeting id>/21-0`, `<meeting id>/21-1`. The key is everything before the segment.
_TEAMS_ID_RE = re.compile(r"^(.*/.+?)[-.]\d+$")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
# The sectioner's TIMESTAMP_LINE takes a bolded name only up to this length.
_MAX_SPEAKER = 80
# "Dana Marx: text" — how Zoom and other tools without voice tags name a speaker. The
# space after the colon is what keeps a clock time ("в 10:30") from being one.
_TEXTUAL_SPEAKER_RE = re.compile(rf"([^:\n]{{1,{_MAX_SPEAKER}}}?)\s*:\s+(.+)")
# How many cues decide whether "Name: text" is a speaker or just speech with a colon.
_SAMPLE_CUES = 50


class VttError(ValueError):
    pass


@dataclass(frozen=True)
class _Cue:
    identifier: str | None
    start: int
    text: str
    voice: str | None


@dataclass(frozen=True)
class _Note:
    lines: tuple[str, ...]


@dataclass
class _Turn:
    start: int
    speaker: str | None
    texts: list[str]


def to_markdown(data: bytes) -> str:
    """One paragraph per turn: a `**Speaker** MM:SS` line (a bare timestamp when the
    file names no speaker), then what was said, its cue line breaks undone."""
    rendered = [r for r in map(_render, _stitch(_group(_parse(_decode(data))))) if r]
    return "\n\n".join(rendered) + "\n" if rendered else ""


def _decode(data: bytes) -> str:
    try:
        for bom, encoding in _BOMS:
            if data.startswith(bom):
                return data[len(bom) :].decode(encoding)
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # A guess with no language behind it is a code page picked by byte statistics
    # alone, which on a short file is routinely the wrong one and decodes to nonsense.
    guess = from_bytes(data).best()
    if guess is None or not guess.languages:
        raise VttError("cannot tell what encoding this file is in; save it as UTF-8")
    return str(guess).lstrip("\ufeff")


def _blocks(text: str) -> Iterator[list[str]]:
    block: list[str] = []
    for line in text.splitlines():
        if line.strip():
            block.append(line)
        elif block:
            yield block
            block = []
    if block:
        yield block


def _parse(text: str) -> list[_Cue | _Note]:
    parsed: list[_Cue | _Note] = []
    for number, lines in enumerate(_blocks(text)):
        if number == 0 and lines[0].startswith("WEBVTT"):
            lines = _cue_under_header(lines)
            if not lines:
                continue
        head = lines[0]
        if head == "NOTE" or head.startswith(("NOTE ", "NOTE\t")):
            first = head[4:].strip()
            parsed.append(_Note(tuple(([first] if first else []) + lines[1:])))
            continue
        if head in ("STYLE", "REGION"):
            continue
        cue = _parse_cue(lines)
        if _plain(cue.text):
            parsed.append(cue)
    return parsed


def _cue_under_header(lines: list[str]) -> list[str]:
    """A cue no blank line separates from the header: from its timing line, or from
    the line above when that is not the `WEBVTT` line itself."""
    timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
    return [] if timing is None else lines[max(timing - 1, 1) :]


def _parse_cue(lines: list[str]) -> _Cue:
    if "-->" in lines[0]:
        identifier, timing, payload = None, lines[0], lines[1:]
    elif len(lines) > 1 and "-->" in lines[1]:
        identifier, timing, payload = lines[0], lines[1], lines[2:]
    else:
        raise VttError(f"a block that is neither a cue nor a note: {lines[0][:80]!r}")
    match = _TIMING_RE.match(timing)
    if match is None:
        raise VttError(f"malformed cue timing: {timing[:80]!r}")
    hours, minutes, seconds = match.groups()
    text = "\n".join(payload)
    voice = _VOICE_RE.search(text)
    return _Cue(
        identifier=identifier,
        start=int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds),
        text=text,
        voice=html.unescape(voice.group(1)).strip() if voice else None,
    )


def _plain(text: str) -> str:
    """Markup goes before entities are decoded: `&lt;b&gt;` is text that spells a tag."""
    lines = (html.unescape(_TAG_RE.sub("", line)).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _textual_speakers(cues: list[_Cue]) -> bool:
    sample = cues[:_SAMPLE_CUES]
    return len(sample) >= 2 and all(
        cue.voice is None and _TEXTUAL_SPEAKER_RE.fullmatch(_plain(cue.text)) for cue in sample
    )


def _group(items: list[_Cue | _Note]) -> list[_Turn | _Note]:
    """One turn per cue, except that the segments of one Teams utterance share one —
    the first segment's — even when another speaker's cue lands between them."""
    textual = _textual_speakers([item for item in items if isinstance(item, _Cue)])
    grouped: list[_Turn | _Note] = []
    utterances: dict[str, _Turn] = {}
    for item in items:
        if isinstance(item, _Note):
            grouped.append(item)
            continue
        speaker, text = item.voice, _plain(item.text)
        if textual and speaker is None:
            match = _TEXTUAL_SPEAKER_RE.fullmatch(text)
            if match:
                speaker, text = match.group(1).strip(), match.group(2).strip()
        key_match = _TEAMS_ID_RE.match(item.identifier or "")
        key = key_match.group(1) if key_match else None
        same = utterances.get(key) if key else None
        if same is not None and same.speaker == speaker:
            same.texts.append(text)
            continue
        turn = _Turn(start=item.start, speaker=speaker, texts=[text])
        if key:
            utterances[key] = turn
        grouped.append(turn)
    return grouped


def _has_list_line(turn: _Turn) -> bool:
    return any(_LIST_LINE_RE.match(line) for text in turn.texts for line in text.split("\n"))


def _stitch(items: list[_Turn | _Note]) -> list[_Turn | _Note]:
    """Adjacent turns of one named speaker become one, unless either holds a list or
    they fall in different transcript windows: a merged turn keeps only its first
    timestamp, so the later speech would be filed under the earlier window."""
    stitched: list[_Turn | _Note] = []
    for item in items:
        previous = stitched[-1] if stitched else None
        if (
            isinstance(item, _Turn)
            and isinstance(previous, _Turn)
            and item.speaker is not None
            and item.speaker == previous.speaker
            and item.start // TRANSCRIPT_WINDOW_SECONDS
            == previous.start // TRANSCRIPT_WINDOW_SECONDS
            and not _has_list_line(previous)
            and not _has_list_line(item)
        ):
            previous.texts.extend(item.texts)
        else:
            stitched.append(item)
    return stitched


def _render(item: _Turn | _Note) -> str:
    if isinstance(item, _Note):
        return "\n".join(f"> {line}" for line in item.lines)
    lines = [line for text in item.texts for line in text.split("\n") if line]
    body = "\n".join(lines) if _has_list_line(item) else " ".join(lines)
    stamp = format_timestamp(item.start)
    name = " ".join((item.speaker or "").replace("*", "").split())
    if not name:
        return f"{stamp}\n{body}"
    if len(name) > _MAX_SPEAKER:
        return f"{stamp}\n{item.speaker}: {body}"
    return f"**{name}** {stamp}\n{body}"
