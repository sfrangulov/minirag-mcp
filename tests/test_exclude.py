"""RAG_EXCLUDE and an indexes.toml's `exclude`: gitignore patterns from minirag's
configuration, applied to every root."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

import minirag_mcp.cli as cli
from minirag_mcp.config import ConfigError, load_config
from minirag_mcp.indexes import load_server_config
from minirag_mcp.ingest.ignore import IGNORE_FILE
from minirag_mcp.ingest.scanner import scan_roots
from minirag_mcp.server import create_app


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "docs"
    for rel in ("a.md", "top.md", "sub/top.md", "drafts/d.md", "sub/drafts/e.md", "x.tmp.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n\nBody of {rel}.")
    return root


def rel_paths(entries, root):
    return sorted(str(e.path.relative_to(root)) for e in entries)


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
    return {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db"), "RAG_EXCLUDE": "drafts/"}


def flags(env):
    return ["--base-dir", env["BASE_DIR"], "--db-path", env["DB_PATH"]]


def test_unset_or_empty_means_no_patterns(tmp_path):
    assert load_config({"BASE_DIR": str(tmp_path)}).exclude == ()
    assert load_config({"BASE_DIR": str(tmp_path), "RAG_EXCLUDE": " ; ;"}).exclude == ()


def test_patterns_are_split_on_semicolons_and_trimmed(tmp_path):
    cfg = load_config({"BASE_DIR": str(tmp_path), "RAG_EXCLUDE": " drafts/ ;; *.tmp.md ;"})
    assert cfg.exclude == ("drafts/", "*.tmp.md")


def test_a_pattern_the_parser_rejects_is_a_config_error_naming_it(tmp_path):
    with pytest.raises(ConfigError, match=r"RAG_EXCLUDE.*\[z-a\]"):
        load_config({"BASE_DIR": str(tmp_path), "RAG_EXCLUDE": "drafts/;[z-a]"})


def test_a_pattern_excludes_a_subfolder_of_a_root_with_no_file(root):
    assert not (root / IGNORE_FILE).exists()
    ignored: list = []
    got = rel_paths(scan_roots([root], ignored=ignored, exclude=("drafts/",)), root)
    assert got == ["a.md", "sub/top.md", "top.md", "x.tmp.md"]
    assert sorted(str(p.relative_to(root)) for p in ignored) == ["drafts", "sub/drafts"]


def test_anchored_patterns_are_relative_to_each_root_and_unanchored_match_at_any_depth(root):
    got = rel_paths(scan_roots([root], exclude=("/top.md", "sub/drafts")), root)
    assert got == ["a.md", "drafts/d.md", "sub/top.md", "x.tmp.md"]
    got = rel_paths(scan_roots([root], exclude=("top.md",)), root)
    assert got == ["a.md", "drafts/d.md", "sub/drafts/e.md", "x.tmp.md"]


def test_one_pattern_applies_to_every_root(tmp_path, root):
    other = tmp_path / "other"
    (other / "drafts").mkdir(parents=True)
    (other / "drafts" / "o.md").write_text("# o\n\nOther.")
    (other / "k.md").write_text("# k\n\nKept.")
    got = {e.path for e in scan_roots([root, other], exclude=("drafts/",))}
    assert other / "k.md" in got and not any("drafts" in str(p) for p in got)


def test_the_root_file_comes_after_the_configured_patterns_and_can_re_include(root):
    (root / IGNORE_FILE).write_text("!x.tmp.md\n")
    got = rel_paths(scan_roots([root], exclude=("*.tmp.md", "drafts/")), root)
    assert got == ["a.md", "sub/top.md", "top.md", "x.tmp.md"]


def test_the_root_file_still_excludes_on_its_own(root):
    (root / IGNORE_FILE).write_text("top.md\n")
    got = rel_paths(scan_roots([root], exclude=("drafts/",)), root)
    assert got == ["a.md", "x.tmp.md"]


def test_matching_ignores_case_and_unicode_composition(tmp_path):
    root = tmp_path / "docs"
    for rel in ("Private/p.md", "Café/c.md", "a.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("# x\n\nBody.")
    got = rel_paths(scan_roots([root], exclude=("private/", "café/")), root)
    assert got == ["a.md"]


def test_a_previously_indexed_file_newly_excluded_leaves_the_index(root, env, capsys, monkeypatch):
    monkeypatch.delenv("RAG_EXCLUDE", raising=False)
    code, out, _ = run_cli(["sync", *flags(env), "--json"], capsys)
    assert code == 0 and json.loads(out)["counts"]["ingested"] == 6
    monkeypatch.setenv("RAG_EXCLUDE", env["RAG_EXCLUDE"])
    code, out, _ = run_cli(["sync", *flags(env), "--json"], capsys)
    counts = json.loads(out)["counts"]
    assert counts["deleted"] == 2 and counts["ignored"] == 2


async def test_an_excluded_file_is_refused_naming_the_variable_in_both_shells(
    root, env, capsys, fake_embedder, monkeypatch
):
    monkeypatch.setenv("RAG_EXCLUDE", env["RAG_EXCLUDE"])
    path = root / "drafts" / "d.md"
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        r = await c.call_tool("ingest_file", {"filePath": str(path)}, raise_on_error=False)
    code, _, err = run_cli(["ingest", str(path), *flags(env)], capsys)
    expected = f"{path} is excluded by RAG_EXCLUDE pattern 'drafts/'"
    assert r.is_error and expected in r.content[0].text
    assert code == 1 and expected in err


async def test_status_counts_the_patterns(root, env, fake_embedder):
    env["RAG_EXCLUDE"] = "drafts/; *.tmp.md ;/top.md"
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        st = (await c.call_tool("status", {})).data
    assert st["excludePatterns"] == 3
    del env["RAG_EXCLUDE"]
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        assert "excludePatterns" not in (await c.call_tool("status", {})).data


async def test_list_files_and_a_scoped_sync_honour_the_patterns(
    root, env, capsys, fake_embedder, monkeypatch
):
    monkeypatch.setenv("RAG_EXCLUDE", env["RAG_EXCLUDE"])
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        listed = (await c.call_tool("list_files", {})).data
        r = await c.call_tool("sync_start", {"path": str(root / "drafts")}, raise_on_error=False)
    code, _, err = run_cli(["sync", str(root / "drafts"), *flags(env)], capsys)
    assert listed["ignored"] == 2
    assert not any("drafts" in f["source"] for f in listed["files"])
    expected = f"{root / 'drafts'} is excluded by RAG_EXCLUDE pattern 'drafts/'"
    assert r.is_error and expected in r.content[0].text
    assert code == 1 and expected in err


def test_cli_status_counts_the_patterns_too(root, env, capsys, monkeypatch):
    monkeypatch.setenv("RAG_EXCLUDE", "drafts/;*.tmp.md")
    code, out, _ = run_cli(["status", *flags(env), "--json"], capsys)
    assert code == 0 and json.loads(out)["excludePatterns"] == 2


def test_a_pattern_only_its_lowercased_form_rejects_is_a_config_error_naming_it(tmp_path):
    with pytest.raises(ConfigError, match=r"RAG_EXCLUDE.*\[Z-a\]"):
        load_config({"BASE_DIR": str(tmp_path), "RAG_EXCLUDE": "[Z-a]"})


def test_an_item_starting_with_a_hash_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match=r"RAG_EXCLUDE.*'#notes'.*\\#"):
        load_config({"BASE_DIR": str(tmp_path), "RAG_EXCLUDE": "drafts/;#notes"})


def test_an_escaped_hash_matches_a_leading_hash(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    for name in ("#a.md", "b.md"):
        (root / name).write_text("# x\n\nBody.")
    assert rel_paths(scan_roots([root], exclude=("\\#a.md",)), root) == ["b.md"]


def test_an_empty_item_does_not_shift_which_pattern_is_named(tmp_path):
    from minirag_mcp.ingest import ignore

    (tmp_path / "drafts").mkdir()
    rules = ignore.load(tmp_path, exclude=("", "drafts/"))
    assert rules.source("drafts", True) == "RAG_EXCLUDE pattern 'drafts/'"


def test_the_list_note_credits_either_source(root, env, capsys, monkeypatch):
    monkeypatch.setenv("RAG_EXCLUDE", env["RAG_EXCLUDE"])
    code, _, err = run_cli(["list", *flags(env)], capsys)
    assert code == 0 and "excluded by .miniragignore or RAG_EXCLUDE" in err


def test_ingesting_a_folder_honours_the_patterns(root, env, capsys, monkeypatch):
    monkeypatch.setenv("RAG_EXCLUDE", env["RAG_EXCLUDE"])
    code, out, _ = run_cli(["ingest", str(root), *flags(env), "--json"], capsys)
    names = [i["source"] for i in json.loads(out)["ingested"]]
    assert code == 0 and len(names) == 4 and not any("drafts" in n for n in names)
    code, _, err = run_cli(["ingest", str(root / "drafts"), *flags(env)], capsys)
    assert code == 1 and "is excluded by RAG_EXCLUDE pattern 'drafts/'" in err


@pytest.fixture
def toml_index(root, tmp_path, monkeypatch, fake_embedder):
    """The index `docs` of an indexes.toml, which the CLI is made to load."""
    path = tmp_path / "config" / "indexes.toml"
    path.parent.mkdir()
    path.write_text(
        f'[index.docs]\nroots = ["{root}"]\ndb = "{tmp_path / "db"}"\nexclude = ["drafts/"]\n',
        encoding="utf-8",
    )
    cfg = load_server_config(path, {"HOME": str(tmp_path)}).indexes["docs"]
    monkeypatch.delenv("RAG_EXCLUDE", raising=False)
    monkeypatch.setattr(cli, "_load_cfg", lambda *args: cfg)
    monkeypatch.setattr(cli, "_make_embedder", lambda c: fake_embedder)
    return path, cfg


async def test_an_index_from_indexes_toml_names_its_exclude_key_in_both_shells(
    root, toml_index, capsys, fake_embedder
):
    path, cfg = toml_index
    drafts = root / "drafts"
    async with Client(create_app(cfg, embedder=fake_embedder)) as c:
        replies = [
            await c.call_tool(tool, args, raise_on_error=False)
            for tool, args in (
                ("ingest_file", {"filePath": str(drafts / "d.md")}),
                ("sync_start", {"path": str(drafts)}),
                ("list_files", {"scope": str(drafts)}),
            )
        ]
    errors = [r.content[0].text for r in replies if r.is_error]
    for tokens in (
        ["ingest", str(drafts / "d.md")],
        ["ingest", str(drafts)],
        ["sync", str(drafts)],
        ["list", "--scope", str(drafts)],
    ):
        code, _, err = run_cli(tokens, capsys)
        errors.append(err if code == 1 else f"exit {code}")
    by = f"excluded by {path} [index.docs] exclude pattern 'drafts/'"
    assert len(errors) == 7 and all(by in e and "RAG_EXCLUDE" not in e for e in errors)


def test_the_list_note_credits_the_exclude_key_of_an_indexes_toml(root, toml_index, capsys):
    path, _ = toml_index
    code, _, err = run_cli(["list"], capsys)
    assert code == 0 and f"excluded by .miniragignore or {path} [index.docs] exclude" in err
