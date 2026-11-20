"""Struck-through text, and keeping it marked when a unit boundary cuts it.

markitdown writes struck-through text — a `.docx` run under `<w:strike/>`, HTML `<s>`
and `<del>` — as `~~text~~`, and `.md` files carry the same GFM markers. The text is
kept, because a deleted clause still records what was removed, but only the markers say
it is not current. A run is marked once, at its ends, so a unit cut from its middle
would hold deleted text with no marker at all; the packer closes the run at the end of
one unit and re-opens it at the start of the next, and for that it has to know which
markers pair.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NamedTuple

MARKER = "~~"
# A tilde a cut leaves beside a marker is written escaped: `~~~` is no marker, and at the
# start of a line it opens a code fence when the stored text is chunked again.
ESCAPE = "\\"
# A backslash escape, a delimiter run of tildes, or the backticks that open a code span:
# a code span outranks every other inline construct, so a `~~` inside one is literal.
_INLINE = re.compile(r"\\.|`+|~+", re.S)
_CELL_EDGE = re.compile(r"(?<!\\)\|")

# (offset of a paired `~~`, +1 when it opens a run and -1 when it closes one)
Mark = tuple[int, int]


def pairs(text: str) -> list[Mark]:
    """The `~~` markers of one paragraph that pair, in order.

    GFM's rule without its punctuation clauses: exactly two tildes, none of them escaped;
    a marker followed by non-whitespace may open, one preceded by non-whitespace may
    close, and a closer takes the nearest open one. A marker left without a partner is a
    literal `~~`, and marks nothing.
    """
    waiting: list[int] = []
    marks: list[Mark] = []
    at = 0
    while found := _INLINE.search(text, at):
        delimiter, at = found.group(), found.end()
        if delimiter[0] == ESCAPE:
            continue
        if delimiter[0] == "`":
            closing = re.compile(f"(?<!`){delimiter}(?!`)").search(text, at)
            at = closing.end() if closing else at
            continue
        if delimiter != MARKER:
            continue
        start = found.start()
        if waiting and start and not text[start - 1].isspace():
            marks += [(waiting.pop(), 1), (start, -1)]
        elif at < len(text) and not text[at].isspace():
            waiting.append(start)
    return sorted(marks)


def row_pairs(row: str) -> list[Mark]:
    """`pairs` for a table row, whose cells are paragraphs of their own."""
    marks: list[Mark] = []
    start = 0
    for edge in [*(m.start() for m in _CELL_EDGE.finditer(row)), len(row)]:
        marks += [(start + at, side) for at, side in pairs(row[start:edge])]
        start = edge + 1
    return marks


class Cut(NamedTuple):
    """A span of marked text as a unit holds it. `reopen` re-opens the runs the span
    starts inside and goes before the unit's own text; `own` closes the runs it ends
    inside, and is then not `verbatim`."""

    reopen: str
    own: str
    verbatim: bool


class Marked(NamedTuple):
    """Text with its paired markers located, so a span can be cut out of it marked.

    `struck` is how many runs are already open where the text starts: the text can be
    one piece of a paragraph, and its markers were paired over the whole paragraph.
    """

    text: str
    struck: int = 0
    marks: Sequence[Mark] = ()

    def open_at(self, at: int) -> int:
        return self.struck + sum(side for offset, side in self.marks if offset < at)

    def cut(self, start: int, end: int) -> Cut:
        own = self.text[start:end]
        first = start + len(own) - len(own.lstrip())
        last = start + len(own.rstrip())
        # One marker however deep the nesting, and none where the document's own already
        # stands: `~~~~` is no marker at all.
        reopen = self.open_at(start) > 0 and (first, 1) not in self.marks
        closing = self.open_at(end) > 0 and (last - len(MARKER), -1) not in self.marks
        if reopen:
            own = own.lstrip()
        if closing:
            own = own.rstrip()
            if own.endswith("~") and not own.endswith(ESCAPE + "~"):
                own = own[:-1] + ESCAPE + "~"
            own += MARKER
        lead = MARKER + ESCAPE if own.startswith("~") else MARKER
        return Cut(lead if reopen else "", own, not closing)

    def uncuttable(self) -> frozenset[int]:
        """Offsets no cut may fall at: inside a marker, or between a marker and the text
        it opens or closes on, where it would stop being one."""
        out: set[int] = set()
        for offset, side in self.marks:
            out |= {offset + 1, offset + 2} if side > 0 else {offset, offset + 1}
        return frozenset(out)
