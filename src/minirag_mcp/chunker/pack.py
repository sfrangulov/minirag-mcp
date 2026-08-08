"""Packing a section into retrieval units that fit the token budget.

The budget is the whole point of the redesign: the encoder truncates at 128 tokens, so
anything past that is not merely unranked, it is *invisible* — measured on the live
index, 22.8% of all indexed tokens were being discarded before they reached the model.
A unit is therefore grown greedily until one more piece would cross the budget.

One exception, and it is deliberate: a fenced code block is atomic. Splitting code
mid-block produces fragments that are wrong rather than merely partial, so an
over-budget fence is emitted whole and the encoder truncates it. Everything else —
prose, table rows, even a single word longer than the budget — is split until it fits.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from minirag_mcp.chunker.sections import TABLE_DELIMITER, TABLE_ROW, TIMESTAMP_LINE, Section
from minirag_mcp.chunker.structural import split_markdown
from minirag_mcp.chunker.tokens import CountTokens

TEXT = "text"
ROW = "row"
CODE = "code"

# Room a prefix must leave for the text it introduces. A breadcrumb or a table header
# that would swallow the budget is dropped in favour of a shorter one.
MIN_BODY_TOKENS = 24
# How many leading lines `join_parent` will consider shared. A heading breadcrumb plus
# its blank line plus a table's two header lines is exactly four.
MAX_SHARED_LINES = 4
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class Atom:
    """The smallest piece the packer will not voluntarily split.

    `header` is the table header rows repeated in front of every unit built from this
    atom, so a row chunk still says what its columns mean. `sep` is what joins this
    atom to the one before it inside a unit.
    """

    text: str
    kind: str
    header: str = ""
    sep: str = " "


def _table_header(lines: Sequence[str]) -> int:
    """Number of leading lines forming the header of a table run, 0 if it is not one.

    A run of pipe-delimited lines is only a table when a delimiter row follows the
    header; without one the pipes are just punctuation and the lines stay prose.
    """
    if len(lines) >= 2 and TABLE_DELIMITER.match(lines[1]):
        return 2
    return 0


def _text_atoms(lines: Sequence[str]) -> list[Atom]:
    """Sentences, remembering the line and paragraph breaks that separated them.

    A block can contain a blank line — `split_markdown` glues a heading to the
    paragraph below it that way — and losing it would run the heading into the text.
    """
    atoms: list[Atom] = []
    pending: str | None = None
    for line in lines:
        if not line.strip():
            if atoms:
                pending = "\n\n"
            continue
        for j, part in enumerate(p for p in _SENTENCE_END.split(line) if p.strip()):
            sep = " " if j else (pending or ("\n" if atoms else " "))
            atoms.append(Atom(text=part, kind=TEXT, sep=sep))
            pending = None
    return atoms


def _block_atoms(block_text: str) -> list[Atom]:
    atoms: list[Atom] = []
    lines = block_text.split("\n")
    i = 0
    while i < len(lines):
        if TABLE_ROW.match(lines[i]):
            j = i
            while j < len(lines) and TABLE_ROW.match(lines[j]):
                j += 1
            run = lines[i:j]
            head = _table_header(run)
            if head:
                header = "\n".join(run[:head])
                atoms += [Atom(text=row, kind=ROW, header=header, sep="\n") for row in run[head:]]
            else:
                atoms += _text_atoms(run)
            i = j
        else:
            j = i
            while j < len(lines) and not TABLE_ROW.match(lines[j]):
                j += 1
            atoms += _text_atoms(lines[i:j])
            i = j
    return atoms


def section_atoms(body: str) -> list[Atom]:
    """Break a section body into atoms, keeping code atomic and tables row-aligned.

    Blocks stay separated by a blank line when they land in the same unit, so a
    transcript's turns and a spec's paragraphs read the way they did on the page.
    """
    atoms: list[Atom] = []
    for block in split_markdown(body, max_chars=None):
        if block.is_code:
            fresh = [Atom(text=block.text, kind=CODE, sep="\n\n")]
        else:
            fresh = _block_atoms(block.text)
        if not fresh:
            continue
        if atoms:
            fresh[0] = Atom(fresh[0].text, fresh[0].kind, fresh[0].header, "\n\n")
        atoms += fresh
    return atoms


def _render(prefix: str, header: str, body: str) -> str:
    parts = []
    if prefix:
        parts.append(prefix + "\n\n")
    if header:
        parts.append(header + "\n")
    parts.append(body)
    return "".join(parts)


def _join(atoms: Sequence[Atom]) -> str:
    return atoms[0].text + "".join(a.sep + a.text for a in atoms[1:])


def _choose_prefix(prefixes: Sequence[str], count_tokens: CountTokens, budget: int) -> str:
    """The most informative prefix that still leaves room for real text."""
    for prefix in prefixes:
        if not prefix or count_tokens(_render(prefix, "", "")) <= budget - MIN_BODY_TOKENS:
            return prefix
    return ""


def _greedy(pieces: Sequence[str], glue: str, wrap, count_tokens: CountTokens, budget: int):
    """Pack `pieces` into the fewest `glue`-joined groups whose wrapped form fits."""
    out: list[str] = []
    current: list[str] = []
    for piece in pieces:
        trial = current + [piece]
        if current and count_tokens(wrap(glue.join(trial))) > budget:
            out.append(glue.join(current))
            current = [piece]
        else:
            current = trial
    if current:
        out.append(glue.join(current))
    return out


def _force_split(text: str, wrap, count_tokens: CountTokens, budget: int) -> list[str]:
    """Last resort for one piece that does not fit: split at whitespace, then anywhere.

    This is what a single table row longer than the whole budget goes through. The row
    is never cut for the *reader* — the parent section still holds it intact — but its
    retrieval units have to fit the encoder, and a truncated vector would drop the tail
    entirely rather than merely index it separately.
    """
    out: list[str] = []
    for piece in _greedy(text.split(" "), " ", wrap, count_tokens, budget):
        if count_tokens(wrap(piece)) <= budget and piece:
            out.append(piece)
        else:
            out += _greedy(list(piece), "", wrap, count_tokens, budget)
    return [p for p in out if p.strip()]


def _resolve_headers(
    atoms: Sequence[Atom], prefix: str, count_tokens: CountTokens, budget: int
) -> list[Atom]:
    """Decide, per table, whether its header row can be repeated in every unit.

    A header too wide to repeat is not simply discarded — that would delete the table's
    column names from the index. It is demoted to an ordinary first row, so it is
    stored exactly once and the section still reads as a table.
    """
    out: list[Atom] = []
    i = 0
    while i < len(atoms):
        atom = atoms[i]
        if atom.kind != ROW or not atom.header:
            out.append(atom)
            i += 1
            continue
        header = atom.header
        j = i
        while j < len(atoms) and atoms[j].kind == ROW and atoms[j].header == header:
            j += 1
        if count_tokens(_render(prefix, header, "")) <= budget - MIN_BODY_TOKENS:
            out += list(atoms[i:j])
        else:
            demoted = [Atom(line, ROW, "", "\n") for line in header.split("\n")]
            demoted[0] = Atom(demoted[0].text, ROW, "", atom.sep)
            out += demoted + [Atom(a.text, ROW, "", "\n") for a in atoms[i:j]]
        i = j
    return out


def pack_section(section: Section, count_tokens: CountTokens, budget: int) -> list[str]:
    """Retrieval units for one section, every one within `budget` tokens.

    The only unit that may exceed it is a fenced code block, which is atomic.
    """
    atoms = section_atoms(section.body)
    if not atoms:
        return []
    prefix = _choose_prefix(section.prefixes, count_tokens, budget)
    atoms = _resolve_headers(atoms, prefix, count_tokens, budget)
    units: list[str] = []
    run: list[Atom] = []

    def flush() -> None:
        if not run:
            return
        header = run[0].header
        body = _join(run)
        rendered = _render(prefix, header, body)
        if run[0].kind == CODE or count_tokens(rendered) <= budget:
            units.append(rendered)
        else:
            units.extend(
                _render(prefix, header, piece)
                for piece in _force_split(
                    body, lambda t: _render(prefix, header, t), count_tokens, budget
                )
            )
        run.clear()

    for atom in atoms:
        if run and (atom.kind != run[0].kind or atom.header != run[0].header):
            flush()
        if atom.kind == CODE:
            run.append(atom)
            flush()
            continue
        if run:
            trial = _render(prefix, run[0].header, _join(run + [atom]))
            if count_tokens(trial) > budget:
                flush()
        run.append(atom)
    flush()
    return units


def join_parent(texts: Sequence[str]) -> str:
    """Rebuild a section from the retrieval units stored for it, in order.

    The units of one section repeat a shared header — a heading breadcrumb, a time
    window label, a table's header row — because each one has to carry that context
    into its own vector. Concatenating them verbatim would repeat the header once per
    unit, and for a table it would plant a header row in the middle of the rows. So a
    unit's leading lines are dropped when the unit before it already ended up carrying
    them, and consecutive table rows are rejoined without a blank line between them.
    """
    kept = [t for t in texts if t.strip()]
    if not kept:
        return ""
    out = kept[0]
    previous = kept[0].split("\n")
    for text in kept[1:]:
        lines = text.split("\n")
        limit = min(len(previous), len(lines) - 1, MAX_SHARED_LINES)
        shared = 0
        while shared < limit and previous[shared] == lines[shared]:
            shared += 1
        body = lines[shared:]
        while body and not body[0].strip():
            body.pop(0)
        previous = lines
        if not body:
            continue
        out = out.rstrip("\n")
        tail = out.rsplit("\n", 1)[-1]
        # A table has to stay contiguous, and a transcript timestamp belongs on the
        # line above the words it introduces; anything else reads better spaced out.
        contiguous = (TABLE_ROW.match(tail) and TABLE_ROW.match(body[0])) or TIMESTAMP_LINE.match(
            tail
        )
        glue = "\n" if contiguous else "\n\n"
        out = out + glue + "\n".join(body)
    return out
