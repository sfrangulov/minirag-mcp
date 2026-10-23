"""One definition of what a literal term matches, for both Python and SQL filtering.

`find_mentions` promises every document that contains a term, and its "none" is a claim
of absence, so the rule has to be the same in the SQL that narrows the table and in the
Python that decides. SQL cannot see where a word starts or collapse whitespace, so its
clause is a superset of the rule; `matches` is the rule itself.

The rule: a term matches where it begins a word, and its end is open, so an inflected
form is found from its stem ("акт" finds "акта", never "контакт"). A term in quotes is
closed at the end too, matching that word and no other. Case, ё against е, and Unicode
composition are ignored; whitespace inside a phrase matches any run of it.

An open end is the default because the costlier error is a false "none": a stem that
also catches "актуально" overcounts visibly, and the word forms it matched are reported
so the caller can narrow it, while a missed inflection is silent.

What the text was stored as matters as much as the term. markitdown writes DOCX, HTML
and XLSX through markdownify, which escapes `_` and `*` with a backslash, and Word keeps
optional hyphens and zero-width marks inside words; both are ignored here. Text stored
in neither composed nor decomposed form, such as CJK compatibility ideographs, is the
one case the SQL side can miss.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from minirag_mcp.scope import sql_str

# Word's optional hyphen and the zero-width marks: present in stored text, never typed.
_IGNORABLE = "\u00ad\u200b\u200c\u200d\u2060\ufeff"
# Characters the slow path drops: the ignorables, a markdown escape's backslash, and
# the combining dot "İ" leaves behind when lowered.
_DROPPED = re.compile(f"[{_IGNORABLE}]|\\\\(?=[_*])|i\u0307")

_QUOTES = {'"': '"', "'": "'", "«": "»", "“": "”"}
# A letter or digit. Unlike `\w`, not an underscore: document codes separate words with it.
_ALNUM = r"[^\W_]"
MAX_FORM_CHARS = 60


def _same(s: str) -> str:
    # Not str.translate: with a mapping table it is about 300 times slower on Cyrillic.
    return s.replace("ё", "е").replace("ς", "σ")


def _fold_map(s: str) -> tuple[str, list[int] | None]:
    """`s` folded, and where each folded character came from in `s`'s NFC form.

    The map is None when folding kept every character in place, which is the common
    case and the fast one.
    """
    nfc = unicodedata.normalize("NFC", s)
    low = nfc.lower()
    if len(low) == len(nfc) and not _DROPPED.search(nfc):
        return _same(low), None
    out: list[str] = []
    where: list[int] = []
    for k, c in enumerate(nfc):
        if c in _IGNORABLE or (c == "\\" and nfc[k + 1 : k + 2] in ("_", "*")):
            continue
        for ch in c.lower():
            if ch == "\u0307" and out and out[-1] == "i":
                continue
            out.append(ch)
            where.append(k)
    return _same("".join(out)), where


def fold(s: str) -> str:
    """`s` with case, ё/е, composition, markdown escapes and invisible marks erased."""
    return _fold_map(s)[0]


@dataclass(frozen=True)
class Term:
    text: str
    words: tuple[str, ...]
    exact: bool
    # Matches the whole word form: an open-ended term runs on to the end of its word.
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class Match:
    span: tuple[int, int]  # the first occurrence, in the text's NFC form
    forms: frozenset[str]  # every word form matched, folded, whitespace collapsed


def _form(s: str) -> str:
    s = " ".join(s.split())
    return s if len(s) <= MAX_FORM_CHARS else s[:MAX_FORM_CHARS] + "…"


def matches(terms: Sequence[Term], text: str) -> list[Match | None]:
    """Each of `terms` in `text`, folding `text` once."""
    folded, where = _fold_map(text)
    out: list[Match | None] = []
    for t in terms:
        found = list(t.pattern.finditer(folded))
        if not found:
            out.append(None)
            continue
        a, b = found[0].span()
        span = (a, b) if where is None else (where[a], where[b - 1] + 1)
        out.append(Match(span, frozenset(_form(m.group(0)) for m in found)))
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


# Rust regex syntax: what has to be escaped, and what a folded character stands for in
# `lower()` of the stored text, which keeps ё, a final ς, İ's combining dot, markdown
# escapes and invisible marks that `fold` erased.
_RX_META = frozenset("\\.+*?()|[]{}^$#&-~")
_RX_GAP = "[" + "".join(f"\\x{{{ord(c):X}}}" for c in _IGNORABLE) + "]*"
_RX_CHAR = {
    "е": "(?:ё|е\\x{308}?)",
    "σ": "[σς]",
    "i": "i\\x{307}?",
    "_": "\\\\?_",
    "*": "\\\\?\\*",
}


def _rx_word(w: str) -> str:
    return _RX_GAP.join(_RX_CHAR.get(c) or ("\\" + c if c in _RX_META else c) for c in w)


def sql_subject(column: str) -> str:
    return f"lower({column})"


def sql_pattern(terms: Sequence[Term]) -> str:
    """A regex `sql_subject` matches for every row `matches` could accept.

    Any word of any term will do, spelled composed or decomposed: macOS stores filenames
    decomposed, and titles come from filenames.
    """
    spellings = sorted(
        {s for t in terms for w in t.words for s in (w, unicodedata.normalize("NFD", w))}
    )
    return "|".join(_rx_word(s) for s in spellings)


def sql_clause(terms: Sequence[Term], columns: Sequence[str]) -> str:
    """Rows that may contain any of `terms` in any of `columns`."""
    pattern = sql_str(sql_pattern(terms))
    return " OR ".join(f"regexp_like({sql_subject(c)}, '{pattern}')" for c in columns)
