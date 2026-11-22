import re
import time
from urllib.parse import quote

from minirag_mcp.dashboard.data import FileRow, FolderStats, IndexView, RootStats
from minirag_mcp.dashboard.render import (
    _css,
    render_folder,
    render_index,
    render_message,
    render_overview,
)

HOSTILE = "<script>alert(1)</script>\"'&"
KW = {"version": "2.0.0a2", "generated": "22.11.2026 14:20"}
ICON = '<link rel="icon" href="data:,">'


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
    assert re.findall(r"<link[^>]*>", page) == [ICON]


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


def test_a_malformed_timestamp_in_the_report_is_escaped():
    bad = "<script>alert(1)</script>"
    view = make_view(
        latest=report(finishedAt=bad, startedAt=bad),
        coverage={"startedAt": "x", "finishedAt": bad, "errors": [], "largeTables": []},
    )
    files = [FileRow("/a", "t", 1, {}, bad)]
    for page in (
        render_overview([view], **KW),
        render_index(view, **KW),
        render_folder(view, "/a", files, **KW),
    ):
        assert "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in render_index(view, **KW)


def test_an_out_of_range_timestamp_in_the_report_is_shown_as_it_is():
    # Valid ISO 8601, but a year 0 in any local time zone: astimezone cannot represent it.
    stamp = "0001-01-01T00:00:00+23:59"
    view = make_view(
        latest=report(finishedAt=stamp, startedAt=stamp),
        coverage={"startedAt": stamp, "finishedAt": stamp, "errors": [], "largeTables": []},
    )
    files = [FileRow("/a", "t", 1, {}, stamp)]
    for page in (
        render_overview([view], **KW),
        render_index(view, **KW),
        render_folder(view, "/a", files, **KW),
    ):
        assert stamp in page


def test_failed_is_the_count_of_the_sync_and_notices_are_still_listed():
    notice = {"source": "/docs/a/locked", "error": "could not list this folder: permission denied"}
    coverage = {
        "startedAt": "2026-11-20T10:00:00+00:00",
        "finishedAt": "2026-11-20T10:05:00+00:00",
        "counts": {"ingested": 3, "failed": 0},
        "errors": [notice],
        "errorsTotal": 1,
        "largeTables": [],
    }
    view = make_view(latest=report(), coverage=coverage)
    overview, page = render_overview([view], **KW), render_index(view, **KW)
    assert '<span class="bad">' not in overview + page
    assert '<td class="num">0</td><td class="num">0</td><td class="num">1</td></tr></tbody>' in (
        overview
    )
    assert '<div class="fig">0</div><div class="cap">failed</div>' in page
    assert "<h3>Did not get in at the last full sync &middot; " in page
    assert "/docs/a/locked" in page and "permission denied" in page
    counted = dict(coverage, counts={"failed": 37})
    page = render_index(make_view(latest=report(), coverage=counted), **KW)
    assert '<div class="fig"><span class="bad">37</span></div>' in page


def test_every_page_names_an_icon_so_the_browser_asks_for_none():
    view = make_view()
    for page in (
        render_overview([view], **KW),
        render_index(view, **KW),
        render_folder(view, "/docs/a", [], **KW),
        render_message("t", "x"),
    ):
        assert page.count(ICON) == 1 and page.index(ICON) < page.index("</head>")


def test_paths_under_the_home_folder_are_shown_from_tilde(monkeypatch):
    home = "/Users/someone"
    monkeypatch.setenv("HOME", home)
    root = f"{home}/Library/CloudStorage/OneDrive-Client/Strategy"
    shown = "~/Library/CloudStorage/OneDrive-Client/Strategy"
    view = make_view(
        roots=(RootStats(root, 1, 1, (FolderStats("sub", f"{root}/sub", 1, 1),)),),
        coverage={
            "counts": {"failed": 1},
            "errors": [{"source": f"{root}/bad.pdf", "error": "boom"}],
            "largeTables": [f"{root}/big.xlsx"],
        },
    )
    files = [FileRow(f"{root}/sub/a.md", "A", 1, {}, None)]
    overview, index = render_overview([view], **KW), render_index(view, **KW)
    folder = render_folder(view, f"{root}/sub", files, **KW)
    for page in (overview, index, folder):
        assert home not in page
    assert f">{shown}</a>" in overview and f">{shown}</a>" in index
    assert f"<td>{shown}/bad.pdf</td>" in index and f"<td>{shown}/big.xlsx</td>" in index
    assert f"{shown}/sub/a.md</span>" in folder and f"<code>{shown}/sub</code>" in folder
    assert f"<title>{shown}/sub - minirag dashboard</title>" in folder
    assert f"path={quote(root, safe='')}" in overview
    assert f"path={quote(f'{root}/sub', safe='')}" in index


def test_only_the_home_folder_itself_becomes_tilde(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/someone")
    roots = tuple(RootStats(p, 1, 1, ()) for p in ("/Users/someone", "/Users/someone-else/x"))
    page = render_overview([make_view(roots=roots)], **KW)
    assert ">~</a>" in page and ">/Users/someone-else/x</a>" in page


def test_only_index_names_keep_to_one_line():
    css = _css()
    assert "td a{" not in css and "a.idx-name{white-space:nowrap}" in css
    page = render_overview([make_view()], **KW)
    assert '<td><a class="idx-name" href="/dashboard/index/alpha">alpha</a></td>' in page


def _rules(css: str) -> dict[str, str]:
    return {sel.strip(): body for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)}


def _colour(css: str, value: str) -> str:
    if value.startswith("var("):
        return re.search(rf"{re.escape(value[4:-1])}:(#[0-9A-Fa-f]{{6}})", css).group(1)
    return value


def _declarations(body: str) -> dict[str, str]:
    return dict(d.split(":", 1) for d in body.split(";") if d)


def _contrast_on_white(hex_colour: str) -> float:
    def channel(c: int) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    lum = 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    return 1.05 / (lum + 0.05)


def test_state_text_is_readable_and_amber_marks_a_warn_state():
    css = _css()
    rules = _rules(css)
    for state in (".state.ok", ".state.warn", ".state.bad"):
        colour = _colour(css, _declarations(rules[state])["color"])
        assert _contrast_on_white(colour) >= 4.5, (state, colour)
    assert _colour(css, _declarations(rules[".state.bad"])["color"]) == "#9E2A2B"
    marker = _declarations(rules[".state.warn::before"])
    assert (marker["width"], marker["height"]) == ("8px", "8px")
    assert _colour(css, marker["background"]) == "#CF7F00"
