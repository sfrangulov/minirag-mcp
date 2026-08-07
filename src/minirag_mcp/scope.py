"""One definition of what a scope prefix matches, for both Python and SQL filtering.

`scope` is the only mechanism that narrows what a caller can see, so the rule has to
be the same everywhere it is applied — in-process filtering of scan results, and the
`where` clause LanceDB prefilters both the vector and the FTS query with.
"""

from __future__ import annotations

from collections.abc import Sequence

SEP = "/"


def is_under(path: str, prefix: str) -> bool:
    """True when `path` is `prefix` itself, or something below it.

    The separator is always "/", never `os.sep`: source ids are not all filesystem
    paths. A data source id is an arbitrary string and a url source is a URL, whose
    separator is "/" on every platform.

    A trailing separator on `prefix` is normalised away, so "/a/b" and "/a/b/" name
    the same scope. Matching stops at a separator, so "/data/proj" covers
    "/data/proj/f.md" but never "/data/project-secret/f.md".
    """
    base = prefix.rstrip(SEP)
    return path == base or path.startswith(base + SEP)


def sql_str(s: str) -> str:
    """Escape a string for a single-quoted SQL literal.

    Lives here because the scope clause is built from it; `store` reuses it for its
    exact-source clauses rather than keeping a second copy of the rule.
    """
    return s.replace("'", "''")


def sql_clause(prefixes: Sequence[str]) -> str | None:
    """The `is_under` rule as a LanceDB filter over the `source` column.

    Returns None for an empty `prefixes` — no scope means no filter, not "match
    nothing". `starts_with` is a literal prefix test, not LIKE, so a source id
    containing `%` or `_` needs no further escaping.
    """
    if not prefixes:
        return None
    return " OR ".join(
        f"(source = '{sql_str(base)}' OR starts_with(source, '{sql_str(base + SEP)}'))"
        for base in (p.rstrip(SEP) for p in prefixes)
    )
