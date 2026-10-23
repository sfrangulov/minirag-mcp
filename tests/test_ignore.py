"""`.miniragignore`: gitignore rules per document root, applied wherever the disk is read."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

import minirag_mcp.cli as cli
from minirag_mcp.config import load_config
from minirag_mcp.ingest.ignore import IGNORE_FILE, IgnoreFileError
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


def test_an_unreadable_file_fails_loudly_rather_than_index_what_it_hides(root):
    rules(root, "drafts/\n")
    (root / IGNORE_FILE).chmod(0)
    try:
        with pytest.raises(IgnoreFileError, match=str(root / IGNORE_FILE)):
            scan_roots([root])
    finally:
        (root / IGNORE_FILE).chmod(0o600)


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
