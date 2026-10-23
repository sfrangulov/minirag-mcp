"""One definition of what a literal term matches, for both Python and SQL filtering.

`find_mentions` promises every document that contains a term, and its "none" is a claim
of absence, so the rule has to be the same in the SQL that narrows the table and in the
Python that decides. SQL cannot see where a word starts or collapse whitespace, so its
clause is a superset of the rule; `Term.search` is the rule itself.

The rule: a term matches where it begins a word, and its end is open, so an inflected
form is found from its stem ("акт" finds "акта", never "контакт"). A term in quotes is
closed at the end too, matching that word and no other. Case, ё against е, and Unicode
composition are ignored; whitespace inside a phrase matches any run of it.

An open end is the default because the costlier error is a false "none": a stem that
also catches "актуально" overcounts visibly, and the word forms it matched are reported
so the caller can narrow it, while a missed inflection is silent.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from minirag_mcp.scope import sql_str

_YO, _E = "ё", "е"
# What an NFD "ё" is left as once lowered: е plus a combining diaeresis.
_NFD_YO = _E + "\u0308"


def fold(s: str) -> str:
    """`s` with case, ё/е and composition erased, at the length of its NFC form.

    Lowered per character because `str.lower` turns "İ" into two characters, which would
    shift every span found after it out of line with the text it was found in.
    """
    nfc = unicodedata.normalize("NFC", s)
    return "".join(low if len(low := c.lower()) == 1 else c for c in nfc).replace(_YO, _E)


_QUOTES = {'"': '"', "«": "»", "“": "”"}
# A letter or digit. Unlike `\w`, not an underscore: document codes separate words with it.
_ALNUM = r"[^\W_]"


@dataclass(frozen=True)
class Term:
    text: str
    words: tuple[str, ...]
    exact: bool
    # Matches the whole word form: an open-ended term runs on to the end of its word.
    pattern: re.Pattern[str]

    def search(self, text: str) -> tuple[int, int] | None:
        """The first match as a span of `text` in NFC, or None."""
        m = self.pattern.search(fold(text))
        return m.span() if m else None


@dataclass(frozen=True)
class Match:
    span: tuple[int, int]  # the first occurrence, in the text's NFC form
    forms: frozenset[str]  # every word form matched, folded, whitespace collapsed


def matches(terms: Sequence[Term], text: str) -> list[Match | None]:
    """Each of `terms` in `text`, folding `text` once."""
    folded = fold(text)
    out: list[Match | None] = []
    for t in terms:
        found = list(t.pattern.finditer(folded))
        out.append(
            Match(found[0].span(), frozenset(" ".join(m.group(0).split()) for m in found))
            if found
            else None
        )
    return out


def term(raw: str) -> Term:
    text = raw.strip()
    inner, exact = text, False
    if len(text) >= 2 and _QUOTES.get(text[0]) == text[-1]:
        inner, exact = text[1:-1], True
    words = tuple(fold(inner).split())
    if not words:
        raise ValueError("a term must not be empty")
    body = r"\s+".join(re.escape(w) for w in words)
    start = f"(?<!{_ALNUM})" if words[0][0].isalnum() else ""
    if not exact:
        end = f"{_ALNUM}*"
    else:
        end = f"(?!{_ALNUM})" if words[-1][-1].isalnum() else ""
    return Term(text, words, exact, re.compile(start + body + end))


def _sql_fold(column: str) -> str:
    return f"replace(replace(lower({column}), '{_YO}', '{_E}'), '{_NFD_YO}', '{_E}')"


def sql_clause(terms: Sequence[Term], columns: Sequence[str]) -> str:
    """Rows that may contain any of `terms` in any of `columns`.

    Every word of a term has to occur in one column, spelled either composed or
    decomposed: macOS stores filenames decomposed, and titles come from filenames.
    """

    def word(column: str, w: str) -> str:
        spellings = sorted({w, unicodedata.normalize("NFD", w)})
        return " OR ".join(f"contains({_sql_fold(column)}, '{sql_str(s)}')" for s in spellings)

    return " OR ".join(
        "(" + " AND ".join(f"({word(column, w)})" for w in t.words) + ")"
        for t in terms
        for column in columns
    )
