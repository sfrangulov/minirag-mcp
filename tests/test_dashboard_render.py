import re
import time

from minirag_mcp.dashboard.data import FileRow, FolderStats, IndexView, RootStats
from minirag_mcp.dashboard.render import (
    render_folder,
    render_index,
    render_message,
    render_overview,
)

HOSTILE = "<script>alert(1)</script>\"'&"
KW = {"version": "2.0.0a2", "generated": "22.11.2026 14:20"}


def make_view(name="alpha", **over):
    base = dict(
        name=name,
        description="Strategy documents",
        state="ready",
        why="",
        documents=212,
        chunks=85318,
        roots=(
            RootStats(
                "/docs/a",
                212,
                85318,
                (
                    FolderStats(
                        "sub", "/docs/a/sub", 212, 85318, (FolderStats("x", "/docs/a/sub/x", 3, 9),)
                    ),
                ),
            ),
        ),
        facets={},
        latest=None,
        coverage=None,
        report_problem=None,
        exclude=("*.tmp",),
        max_table_rows=1000,
        model="bge",
        fts_language="english",
        reranker=None,
    )
    base.update(over)
    return IndexView(**base)


def report(outcome="succeeded", **over):
    latest = {
        "startedAt": "2026-11-22T14:00:00+00:00",
        "finishedAt": "2026-11-22T14:12:00+00:00",
        "ranBy": "cli",
        "scope": None,
        "outcome": outcome,
        "error": "disk full" if outcome == "failed" else None,
        "counts": {"ingested": 2},
    }
    latest.update(over)
    return latest


def test_a_file_name_with_markup_renders_as_text():
    view = make_view(
        description=HOSTILE,
        roots=(RootStats(HOSTILE, 1, 1, (FolderStats(HOSTILE, HOSTILE, 1, 1),)),),
        exclude=(HOSTILE,),
        coverage={"errors": [{"source": HOSTILE, "error": HOSTILE}], "largeTables": [HOSTILE]},
    )
    files = [FileRow(HOSTILE, HOSTILE, 1, {"type": HOSTILE}, None)]
    pages = [
        render_overview([view], **KW),
        render_index(view, **KW),
        render_folder(view, HOSTILE, files, **KW),
        render_message(HOSTILE, HOSTILE),
    ]
    for page in pages:
        assert "<script>" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_the_page_requests_nothing_off_the_machine():
    view = make_view(latest=report(), coverage={"errors": [], "largeTables": []})
    for page in (
        render_overview([view], **KW),
        render_index(view, **KW),
        render_folder(view, "/docs/a", [], **KW),
        render_message("t", "x"),
    ):
        assert not re.search(r"(?:src|href)\s*=\s*[\"']?\s*(?:https?:)?//", page)
        assert not re.search(r"url\(\s*[\"']?\s*(?:https?:)?//", page)
        assert "@import" not in page
        assert "http://" not in page and "https://" not in page


def test_the_version_keeps_its_case():
    page = render_overview([make_view()], **KW)
    assert "2.0.0a2" in page


def test_overview_lists_every_index_with_its_totals():
    page = render_overview(
        [make_view("alpha"), make_view("beta", documents=10, chunks=20)],
        **KW,
    )
    assert "/dashboard/index/alpha" in page and "/dashboard/index/beta" in page
    assert "2 indexes" in page
    assert "222 documents in 85 338 chunks" in page
    assert "85 318" in page


def test_an_index_without_a_report_says_so():
    page = render_overview([make_view()], **KW)
    assert "no report yet" in page


def test_a_failed_latest_is_shown_beside_the_coverage(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    coverage = {
        "startedAt": "2026-11-20T10:00:00+00:00",
        "finishedAt": "2026-11-20T10:05:00+00:00",
        "errors": [{"source": "/docs/a/bad.pdf", "error": "boom"}],
        "errorsTotal": 1,
        "largeTables": ["/docs/a/big.xlsx"],
    }
    page = render_index(make_view(latest=report("failed"), coverage=coverage), **KW)
    assert "The last sync failed:</b> disk full" in page
    assert "/docs/a/bad.pdf" in page and "boom" in page
    assert "/docs/a/big.xlsx" in page
    assert "20.11.2026 10:05" in page
    assert "ran later" in page


def test_the_css_is_inlined():
    page = render_message("t", "x")
    assert "<style>" in page and "--navy:#002B49" in page
    assert "<link" not in page


def test_kinds_block_needs_a_type_facet_and_folds_the_tail():
    assert "Kinds of documents" not in render_overview([make_view()], **KW)
    values = {f"k{i}": {"count": 10 - i} for i in range(8)}
    facets = {"type": {"values": values, "uncertain": 4}}
    page = render_overview([make_view(facets=facets)], **KW)
    assert "Kinds of documents" in page and "other kinds" in page
    assert "4 of those labels are uncertain" in page


def test_an_unavailable_index_shows_its_reason():
    page = render_index(make_view(state="unavailable", why="index.json is unreadable"), **KW)
    assert "unavailable" in page and "index.json is unreadable" in page


def test_folder_links_are_urlencoded_and_capped():
    kids = tuple(FolderStats(f"c{i}", f"/r/p/c{i}", 1, 1) for i in range(15))
    view = make_view(roots=(RootStats("/r", 1, 1, (FolderStats("a b", "/r/a b", 1, 1, kids),)),))
    page = render_index(view, **KW)
    assert "/dashboard/index/alpha/folder?path=%2Fr%2Fa%20b" in page
    assert "&hellip; 3 more folders" in page


def teardown_module():
    time.tzset()
