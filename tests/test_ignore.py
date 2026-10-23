"""`.miniragignore`: gitignore rules per document root, applied wherever the disk is read."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client

import minirag_mcp.cli as cli
from minirag_mcp.config import load_config
from minirag_mcp.ingest.ignore import IGNORE_FILE
from minirag_mcp.ingest.scanner import scan_roots
from minirag_mcp.server import create_app


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "docs"
    for rel in ("a.md", "top.md", "sub/top.md", "drafts/d.md", "sub/drafts/e.md", "x.tmp.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n\nBody of {rel}.")
    (root / "keep.tmp.md").write_text("# keep\n\nKept.")
    return root


def rel_paths(entries, root):
    return sorted(str(e.path.relative_to(root)) for e in entries)


def rules(root, text):
    (root / IGNORE_FILE).write_text(text)


def test_without_a_file_nothing_is_excluded(root):
    ignored: list = []
    assert len(scan_roots([root], ignored=ignored)) == 7
    assert ignored == []


def test_gitignore_rules_exclude_files_and_folders(root):
    rules(root, "# private drafts\ndrafts/\n*.tmp.md\n!keep.tmp.md\n/top.md\n")
    ignored: list = []
    got = rel_paths(scan_roots([root], ignored=ignored), root)
    assert got == ["a.md", "keep.tmp.md", "sub/top.md"]
    # A folder counts once, and nothing under it is walked.
    assert sorted(str(p.relative_to(root)) for p in ignored) == [
        "drafts",
        "sub/drafts",
        "top.md",
        "x.tmp.md",
    ]


def test_each_root_has_its_own_rules(tmp_path, root):
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.md").write_text("# a\n\nOther a.")
    rules(root, "a.md\n")
    got = scan_roots([root, other])
    assert other / "a.md" in {e.path for e in got}
    assert root / "a.md" not in {e.path for e in got}


def test_an_unreadable_file_stops_its_own_root_and_no_other(tmp_path, root):
    other = tmp_path / "other"
    other.mkdir()
    (other / "o.md").write_text("# o\n\nOther.")
    rules(root, "drafts/\n")
    (root / IGNORE_FILE).chmod(0)
    blind: list = []
    try:
        got = scan_roots([root, other], blind=blind)
    finally:
        (root / IGNORE_FILE).chmod(0o600)
    assert [e.path for e in got] == [other / "o.md"]
    assert [b[0] for b in blind] == [root] and str(root / IGNORE_FILE) in blind[0][1]


def test_a_broken_link_as_the_rule_file_is_an_error_not_an_absence(root):
    (root / IGNORE_FILE).symlink_to(root / "nowhere")
    blind: list = []
    assert scan_roots([root], blind=blind) == []
    assert "link to a file that is not there" in blind[0][1]


def test_matching_ignores_a_bom_case_and_unicode_composition(root):
    import unicodedata

    (root / "Private").mkdir()
    (root / "Private" / "p.md").write_text("# p\n\nSecret.")
    nfd = unicodedata.normalize("NFD", "Счёт")
    (root / nfd).mkdir()
    (root / nfd / "s.md").write_text("# s\n\nSecret.")
    (root / IGNORE_FILE).write_bytes("\ufeffa.md\nprivate/\nсчёт/\n".encode())
    got = rel_paths(scan_roots([root]), root)
    assert "a.md" not in got
    assert not any(p.startswith(("Private", nfd)) for p in got)


def test_nested_roots_are_each_held_to_the_other_ones_rules(tmp_path):
    outer = tmp_path / "A"
    inner = outer / "B"
    for rel in ("B/private/p.md", "B/own/o.md", "B/kept.md"):
        (outer / rel).parent.mkdir(parents=True, exist_ok=True)
        (outer / rel).write_text(f"# {rel}\n\nText.")
    (outer / IGNORE_FILE).write_text("B/private/\n")
    (inner / IGNORE_FILE).write_text("own/\n")
    ignored: list = []
    got = {e.path for e in scan_roots([outer, inner], ignored=ignored)}
    assert got == {inner / "kept.md"}
    assert sorted(ignored) == [inner / "own", inner / "private"]


def test_a_symlink_to_an_excluded_file_is_left_out(root):
    rules(root, "drafts/\n")
    (root / "alias.md").symlink_to(root / "drafts" / "d.md")
    ignored: list = []
    got = rel_paths(scan_roots([root], ignored=ignored), root)
    assert "alias.md" not in got and root / "alias.md" in ignored


# --- the tools and commands ---------------------------------------------------------------


def run_cli(tokens, capsys):
    try:
        cli.app(tokens, result_action="return_value", exit_on_error=False)
        code = 0
    except SystemExit as e:
        code = e.code
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def env(root, tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    return {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}


def flags(env):
    return ["--base-dir", env["BASE_DIR"], "--db-path", env["DB_PATH"]]


def test_a_rule_added_later_takes_its_files_out_of_the_index(root, env, capsys):
    code, out, _ = run_cli(["sync", *flags(env), "--json"], capsys)
    assert code == 0 and json.loads(out)["counts"]["ingested"] == 7
    rules(root, "drafts/\n")
    code, out, _ = run_cli(["sync", *flags(env), "--json"], capsys)
    counts = json.loads(out)["counts"]
    assert counts["deleted"] == 2 and counts["ignored"] == 2
    code, out, _ = run_cli(["query", "Body of drafts", *flags(env), "--json"], capsys)
    assert not any("drafts" in s["source"] for s in json.loads(out)["sources"])


async def test_list_files_counts_what_was_excluded(root, env, capsys, fake_embedder):
    rules(root, "drafts/\n*.tmp.md\n")
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        listed = (await c.call_tool("list_files", {})).data
    code, out, _ = run_cli(["list", *flags(env), "--json"], capsys)
    assert listed["ignored"] == 4  # drafts, sub/drafts, x.tmp.md, keep.tmp.md
    assert not any("drafts" in f["source"] for f in listed["files"])
    assert code == 0 and json.loads(out) == listed


async def test_an_excluded_file_is_refused_by_name_in_both_shells(root, env, capsys, fake_embedder):
    rules(root, "drafts/\n")
    path = root / "drafts" / "d.md"
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        r = await c.call_tool("ingest_file", {"filePath": str(path)}, raise_on_error=False)
    code, _, err = run_cli(["ingest", str(path), *flags(env)], capsys)
    expected = f"{path} is excluded by {root / IGNORE_FILE}"
    assert r.is_error and expected in r.content[0].text
    assert code == 1 and expected in err


def test_ingesting_a_folder_honours_the_rules(root, env, capsys):
    rules(root, "drafts/\n")
    code, out, _ = run_cli(["ingest", str(root), *flags(env), "--json"], capsys)
    names = [i["source"] for i in json.loads(out)["ingested"]]
    assert code == 0 and len(names) == 5 and not any("drafts" in n for n in names)


async def test_status_names_each_ignore_file_and_its_rule_count(root, env, fake_embedder):
    rules(root, "# comment\n\ndrafts/\n*.tmp.md\n")
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        st = (await c.call_tool("status", {})).data
    assert st["ignoreFiles"] == [{"file": str(root / IGNORE_FILE), "rules": 2}]


def test_ingesting_a_subfolder_applies_the_roots_rules(root, env, capsys):
    rules(root, "sub/drafts/\n")
    code, out, _ = run_cli(["ingest", str(root / "sub"), *flags(env), "--json"], capsys)
    names = [i["source"] for i in json.loads(out)["ingested"]]
    assert code == 0 and names == [str(root / "sub" / "top.md")]


async def test_a_file_under_an_excluded_folder_stays_out_even_if_re_included(
    root, env, fake_embedder
):
    """git cannot re-include a file whose folder is excluded; neither does the scan,
    which never walks the folder, so neither may an explicit ingest."""
    rules(root, "drafts/\n!drafts/d.md\n")
    assert root / "drafts" / "d.md" not in {e.path for e in scan_roots([root])}
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        r = await c.call_tool(
            "ingest_file", {"filePath": str(root / "drafts" / "d.md")}, raise_on_error=False
        )
    assert r.is_error and "is excluded by" in r.content[0].text


def test_a_file_indexed_before_its_rule_goes_even_if_this_install_cannot_read_it(root, env, capsys):
    from minirag_mcp.store import ChunkRecord, Store

    png = root / "scan.png"
    png.write_bytes(b"\x89PNG")
    store = Store(Path(env["DB_PATH"]), dim=8)
    store.replace_source(
        str(png),
        [
            ChunkRecord(
                f"{png}#0",
                str(png),
                "file",
                "scan",
                0,
                "OCR text",
                [1.0] + [0.0] * 7,
                "h",
                png.stat().st_mtime,
                "2026-10-23T00:00:00+00:00",
            )
        ],
    )
    rules(root, "*.png\n")
    code, out, _ = run_cli(["sync", *flags(env), "--json"], capsys)
    counts = json.loads(out)["counts"]
    assert counts["deleted"] == 1 and counts["unreadable"] == 0
    assert store.get_source(str(png)) is None


def test_a_sync_with_an_unreadable_rule_file_fails_that_root_and_deletes_nothing(root, env, capsys):
    code, _, _ = run_cli(["sync", *flags(env)], capsys)
    rules(root, "drafts/\n")
    (root / IGNORE_FILE).chmod(0)
    try:
        code, out, err = run_cli(["sync", *flags(env), "--json"], capsys)
    finally:
        (root / IGNORE_FILE).chmod(0o600)
    payload = json.loads(out)
    assert code == 1 and payload["counts"]["failed"] >= 1 and payload["counts"]["deleted"] == 0
    assert str(root / IGNORE_FILE) in err


async def test_a_scope_inside_an_excluded_folder_is_refused_not_a_silent_no_op(
    root, env, capsys, fake_embedder
):
    rules(root, "drafts/\n")
    scope = root / "drafts"
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        r = await c.call_tool("sync_start", {"path": str(scope)}, raise_on_error=False)
        listed = await c.call_tool("list_files", {"scope": str(scope)}, raise_on_error=False)
    code, _, err = run_cli(["sync", str(scope), *flags(env)], capsys)
    for text in (r.content[0].text, listed.content[0].text, err):
        assert f"{scope} is excluded by {root / IGNORE_FILE}" in text
    assert r.is_error and listed.is_error and code == 1


def test_ingesting_an_excluded_folder_is_refused(root, env, capsys):
    rules(root, "drafts/\n")
    code, _, err = run_cli(["ingest", str(root / "drafts"), *flags(env)], capsys)
    assert code == 1 and f"is excluded by {root / IGNORE_FILE}" in err


async def test_counts_stay_inside_a_scope(root, env, capsys, fake_embedder):
    rules(root, "drafts/\n")
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        listed = (await c.call_tool("list_files", {"scope": str(root / "sub")})).data
    code, out, _ = run_cli(["sync", str(root / "sub"), *flags(env), "--json"], capsys)
    assert listed["ignored"] == 1 and json.loads(out)["counts"]["ignored"] == 1


async def test_status_reports_a_rule_file_it_cannot_read(root, env, fake_embedder):
    rules(root, "drafts/\n")
    (root / IGNORE_FILE).chmod(0)
    try:
        async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
            st = (await c.call_tool("status", {})).data
    finally:
        (root / IGNORE_FILE).chmod(0o600)
    (entry,) = st["ignoreFiles"]
    assert entry["file"] == str(root / IGNORE_FILE) and "cannot read" in entry["error"]
