"""`displayPath` — what a Sources list is written from, beside the `source` it derives from.

The two fields are deliberately not the same string and are not interchangeable. `source`
is the identity key: `read_file`, `delete_file` and re-ingest all address a document by
it, so it stays the absolute path it has always been. `displayPath` exists only to be
printed, and the citation policy asks for it by name, copied verbatim.

Why the relative path and not the `title`: the title is derived — underscores become
spaces, the extension is dropped — so `И-112_ЗПС_Хранение ТМЗ.docx` reaches the user as
`И-112 ЗПС Хранение ТМЗ`, which is not the name of anything on disk. The relative path
carries the real filename, so what the user reads matches what they can find.

What these tests pin is that the field is always present, never empty, always a byte-wise
suffix of the `source` it came from when that source is under a root, and never silently
degraded to a bare filename or a re-cased one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minirag_mcp.results import aggregate_sources, display_path
from minirag_mcp.store import SearchResult

ROOT = "/Users/x/Documents/solidcore_data"

#: A real relative path from the indexed corpus: a numbered directory, a space in the
#: directory name, an underscore-and-space filename, Cyrillic, and a `.docx` extension —
#: every character class the `title` derivation would have flattened.
REAL_REL = "instructions/11. Учет затрат/И-112_ЗПС_Хранение ТМЗ.docx"


def hit(source: str, title: str = "T", chunk_index: int = 0) -> SearchResult:
    return SearchResult(
        text="body",
        source=source,
        title=title,
        chunk_index=chunk_index,
        score=1.0,
        distance=0.1,
    )


def test_every_source_carries_a_non_empty_display_path():
    """The policy says "copy `displayPath`", so a missing one is a citation that cannot
    be written. There is no fallback for the model to reach for, by design."""
    rows = aggregate_sources([hit(f"{ROOT}/a.md"), hit(f"{ROOT}/notes/b.md")], roots=[Path(ROOT)])
    assert [r["displayPath"] for r in rows] == ["a.md", "notes/b.md"]
    assert all(r["displayPath"] for r in rows)


def test_the_filename_survives_unmodified():
    """The whole reason the field is the path and not the `title`.

    `title` normalises underscores to spaces and strips the extension; a user looking for
    the file would be looking for a name that is not on disk. The path is copied byte for
    byte — separators, case, spaces, digits, extension.
    """
    source = f"{ROOT}/{REAL_REL}"
    (row,) = aggregate_sources([hit(source, "И-112 ЗПС Хранение ТМЗ")], roots=[Path(ROOT)])
    assert row["displayPath"] == REAL_REL
    assert row["displayPath"].endswith("И-112_ЗПС_Хранение ТМЗ.docx")
    assert row["source"] == source  # the identity key is untouched


def test_display_path_is_a_suffix_of_source():
    """Not a re-rendered path: the same bytes, with the root's prefix removed.

    Anything that round-tripped through a normaliser could change a separator or a
    Unicode composition and still look right in a terminal.
    """
    source = f"{ROOT}/{REAL_REL}"
    (row,) = aggregate_sources([hit(source)], roots=[Path(ROOT)])
    assert source.endswith(row["displayPath"])
    assert source == f"{ROOT}/{row['displayPath']}"


def test_the_root_that_contains_the_document_is_the_one_used():
    """Several roots are ordinary. Each document is measured against its own."""
    roots = [Path("/a/notes"), Path("/b/specs")]
    rows = aggregate_sources([hit("/b/specs/x/y.md"), hit("/a/notes/z.md")], roots=roots)
    assert [r["displayPath"] for r in rows] == ["x/y.md", "z.md"]


def test_nested_roots_resolve_to_the_innermost_one():
    """When two roots both contain the file, the more specific one owns it.

    Deterministic beats first-wins: the answer must not depend on the order the roots
    happened to be configured in.
    """
    roots = [Path("/a"), Path("/a/b")]
    (row,) = aggregate_sources([hit("/a/b/c.md")], roots=roots)
    assert row["displayPath"] == "c.md"
    assert (
        aggregate_sources([hit("/a/b/c.md")], roots=list(reversed(roots)))[0]["displayPath"]
        == "c.md"
    )


def test_a_source_outside_every_root_falls_back_to_itself():
    """A document indexed under a root that has since been reconfigured away.

    The absolute path is worse to read than a relative one and better than nothing;
    an empty string would leave the model to invent the line, which is the failure the
    field exists to remove.
    """
    (row,) = aggregate_sources([hit("/elsewhere/a.md")], roots=[Path(ROOT)])
    assert row["displayPath"] == "/elsewhere/a.md"


def test_no_roots_configured_falls_back_to_the_source():
    """`aggregate_sources` is called from the CLI too, and a caller may have none."""
    (row,) = aggregate_sources([hit("/a/b.md")])
    assert row["displayPath"] == "/a/b.md"


@pytest.mark.parametrize(
    "source",
    ["https://example.com/docs/onboarding", "notes-pasted-2026-08", "data:inline"],
)
def test_a_data_or_url_source_cites_the_id_it_was_ingested_under(source):
    """`data` and `url` items have no filesystem path at all — `source` is the id, and
    for a URL that id is the URL, which is already the most locating string there is."""
    (row,) = aggregate_sources([hit(source)], roots=[Path(ROOT)])
    assert row["displayPath"] == source


def test_a_root_prefix_that_is_not_a_path_boundary_is_not_stripped():
    """`/data` must not claim `/data_archive/x.md`. String-prefix logic gets this wrong."""
    (row,) = aggregate_sources([hit("/data_archive/x.md")], roots=[Path("/data")])
    assert row["displayPath"] == "/data_archive/x.md"


def test_the_root_itself_as_a_source_keeps_a_usable_string():
    """Degenerate but reachable: relpath would be `.`, which locates nothing."""
    (row,) = aggregate_sources([hit(ROOT)], roots=[Path(ROOT)])
    assert row["displayPath"] == ROOT


def test_display_path_helper_matches_what_aggregate_puts_on_the_row():
    """The helper is the whole derivation; `aggregate_sources` only calls it."""
    source = f"{ROOT}/{REAL_REL}"
    assert display_path(source, [Path(ROOT)]) == REAL_REL
    (row,) = aggregate_sources([hit(source)], roots=[Path(ROOT)])
    assert row["displayPath"] == display_path(source, [Path(ROOT)])


def test_aggregation_still_dedupes_in_rank_order():
    """The field is additive; the list it hangs off is unchanged."""
    rows = aggregate_sources(
        [hit(f"{ROOT}/b.md"), hit(f"{ROOT}/a.md"), hit(f"{ROOT}/b.md", chunk_index=1)],
        roots=[Path(ROOT)],
    )
    assert [r["source"] for r in rows] == [f"{ROOT}/b.md", f"{ROOT}/a.md"]
    assert rows[0]["hits"] == 2
    assert set(rows[0]) == {"source", "title", "hits", "displayPath"}
