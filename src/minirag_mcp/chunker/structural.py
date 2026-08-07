"""Stage 1 chunking: split Markdown into structural blocks.

Units are paragraphs. Headings glue to the paragraph that follows them so a
heading never dangles alone. Fenced code blocks are atomic and never split,
regardless of size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"^(```+|~~~+)")
_HEADING_RE = re.compile(r"^#{1,6}\s")


@dataclass(frozen=True)
class Block:
    text: str
    is_code: bool


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    words = text.split(" ")
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            pieces.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def split_markdown(markdown: str, *, max_chars: int = 1500) -> list[Block]:
    lines = markdown.split("\n")
    blocks: list[Block] = []
    para: list[str] = []          # accumulating non-code lines
    pending_heading: str | None = None

    def flush_para() -> None:
        nonlocal pending_heading
        text = "\n".join(para).strip()
        para.clear()
        if not text:
            return
        if pending_heading is not None:
            text = f"{pending_heading}\n\n{text}"
            pending_heading = None
        for piece in _split_long_paragraph(text, max_chars):
            blocks.append(Block(text=piece, is_code=False))

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE_RE.match(line.lstrip())
        if fence:
            flush_para()
            open_seq = fence.group(1)
            code = [line]
            i += 1
            while i < len(lines):
                code.append(lines[i])
                lstripped = lines[i].lstrip()
                if (
                    lstripped.startswith(open_seq[0])
                    and len(lstripped) - len(lstripped.lstrip(open_seq[0]))
                    >= len(open_seq)
                ):
                    break
                i += 1
            if pending_heading is not None:
                code.insert(0, "")
                code.insert(0, pending_heading)
                pending_heading = None
            blocks.append(Block(text="\n".join(code).strip(), is_code=True))
        elif _HEADING_RE.match(line):
            flush_para()
            if pending_heading is not None:
                pending_heading = pending_heading + "\n\n" + line.strip()
            else:
                pending_heading = line.strip()
        elif line.strip() == "":
            flush_para()
        else:
            para.append(line)
        i += 1

    flush_para()
    if pending_heading is not None:
        blocks.append(Block(text=pending_heading, is_code=False))
    return blocks
