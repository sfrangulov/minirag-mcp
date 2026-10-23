"""The error and empty-result paths, run through both shells.

Parity between the MCP server and the CLI is otherwise a convention, and two of these
paths had drifted (minirag-mcp-jd6): each case here asks both surfaces the same thing and
checks that the caller is told the same thing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastmcp import Client

import minirag_mcp.cli as cli
import minirag_mcp.embedder as embedder_mod
from minirag_mcp.config import load_config
from minirag_mcp.embedder import Embedder
from minirag_mcp.labels import Labeler
from minirag_mcp.server import create_app


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a.md").write_text("# Alpha\n\nAlpha body about tokens and auth.")
    return root


def run_cli(tokens, capsys) -> tuple[int, str, str]:
    try:
        cli.app(tokens, result_action="return_value", exit_on_error=False)
        code = 0
    except SystemExit as e:
        code = e.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


async def call_mcp(env: dict, tool: str, args: dict, embedder) -> tuple[bool, object]:
    cfg = load_config(env)
    async with Client(create_app(cfg, embedder=embedder)) as c:
        r = await c.call_tool(tool, args, raise_on_error=False)
    return r.is_error, (r.content[0].text if r.is_error else r.data)


def _flags(env: dict) -> list[str]:
    out = ["--base-dir", env["BASE_DIR"], "--db-path", env["DB_PATH"]]
    if "CACHE_DIR" in env:
        out += ["--cache-dir", env["CACHE_DIR"]]
    return out


async def test_an_empty_query_is_refused_by_both(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    is_error, text = await call_mcp(env, "query_documents", {"query": "   "}, fake_embedder)
    code, _, err = run_cli(["query", "   ", *_flags(env)], capsys)
    assert is_error and "query must not be empty" in text
    assert code == 1 and "query must not be empty" in err


async def test_an_unusable_db_path_is_an_index_error_in_both_status_reports(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    """`status` is what a broken setup is diagnosed with, so it answers in both shells,
    and names the setting to change rather than an errno."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("a file where the index folder should be")
    env = {"BASE_DIR": str(root), "DB_PATH": str(blocker)}
    is_error, st = await call_mcp(env, "status", {}, fake_embedder)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    assert not is_error and "DB_PATH" in st["indexError"]
    assert code == 0
    assert json.loads(out)["indexError"] == st["indexError"]


async def test_an_unusable_db_path_fails_a_query_the_same_way(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    env = {"BASE_DIR": str(root), "DB_PATH": str(blocker)}
    is_error, text = await call_mcp(env, "query_documents", {"query": "alpha"}, fake_embedder)
    code, _, err = run_cli(["query", "alpha", *_flags(env)], capsys)
    assert is_error and "DB_PATH" in text
    assert code == 1 and "DB_PATH" in err


async def test_an_empty_index_says_so_instead_of_returning_nothing(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    """`{"results": []}` from an index with nothing in it reads exactly like a corpus
    that does not cover the question."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    is_error, res = await call_mcp(env, "query_documents", {"query": "alpha"}, fake_embedder)
    code, out, _ = run_cli(["query", "alpha", *_flags(env), "--json"], capsys)
    assert not is_error and res["results"] == []
    assert "sync_start" in res["notice"]
    assert code == 0 and json.loads(out)["notice"] == res["notice"]


@pytest.mark.parametrize("scope", ["/somewhere/else", "sub"])
async def test_a_scope_no_indexed_document_is_under_says_so(
    scope, root, tmp_path, capsys, monkeypatch, fake_embedder
):
    """A relative scope is the likely mistake: `displayPath` is relative, and a scope is
    matched against the absolute `source`."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    cfg = load_config(env)
    async with Client(create_app(cfg, embedder=fake_embedder)) as c:
        await c.call_tool("ingest_file", {"filePath": str(root / "sub" / "a.md")})
        res = (await c.call_tool("query_documents", {"query": "alpha", "scope": scope})).data
    code, out, _ = run_cli(["query", "alpha", "--scope", scope, *_flags(env), "--json"], capsys)
    assert res["results"] == []
    assert scope in res["notice"] and "`source`" in res["notice"]
    assert code == 0 and json.loads(out)["notice"] == res["notice"]


async def test_a_scope_that_holds_documents_but_no_match_adds_no_notice(
    root, tmp_path, fake_embedder
):
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db"), "RAG_MAX_DISTANCE": "0"}
    cfg = load_config(env)
    async with Client(create_app(cfg, embedder=fake_embedder)) as c:
        await c.call_tool("ingest_file", {"filePath": str(root / "sub" / "a.md")})
        res = (
            await c.call_tool("query_documents", {"query": "zzz", "scope": str(root / "sub")})
        ).data
    assert res["results"] == []
    assert "notice" not in res


async def test_a_model_that_cannot_load_is_reported_by_both(root, tmp_path, capsys, monkeypatch):
    real = embedder_mod.TextEmbedding

    class Unloadable:
        # The registry lookup behind `dim` needs no download and must keep working.
        list_supported_models = staticmethod(real.list_supported_models)

        def __init__(self, *args, **kwargs):
            raise ValueError("Could not load model x from any source.")

    monkeypatch.setattr(embedder_mod, "TextEmbedding", Unloadable)
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: Embedder(cfg.model_name, cfg.cache_dir))
    env = {
        "BASE_DIR": str(root),
        "DB_PATH": str(tmp_path / "db"),
        "CACHE_DIR": str(tmp_path / "models"),
    }
    is_error, text = await call_mcp(env, "query_documents", {"query": "alpha"}, None)
    code, _, err = run_cli(["query", "alpha", *_flags(env)], capsys)
    assert is_error and "could not be loaded" in text
    assert code == 1 and "could not be loaded" in err


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a chmod 000 folder anyway")
async def test_a_folder_the_listing_cannot_see_into_is_reported_by_both(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    """minirag-mcp-k3w: the file under it stays indexed — `status` counts it and a search
    returns it — so a listing that leaves it out without a word contradicts both."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    hidden = root / "sub"
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        await c.call_tool("ingest_file", {"filePath": str(hidden / "a.md")})
        os.chmod(hidden, 0)
        try:
            listed = (await c.call_tool("list_files", {})).data
            code, out, err = run_cli(["list", *_flags(env), "--json"], capsys)
        finally:
            os.chmod(hidden, 0o755)

    by_source = {f["source"]: f for f in listed["files"]}
    assert by_source[str(hidden / "a.md")]["state"] == "unlisted"
    [folder] = listed["unlistedFolders"]
    assert folder["path"] == str(hidden)
    assert folder["indexedCount"] == 1
    assert "readable" in folder["message"]
    assert code == 0
    assert json.loads(out) == listed
    assert f"warn: {hidden}: {folder['message']}" in err


async def test_a_listing_with_nothing_hidden_reports_no_folders(root, tmp_path, fake_embedder):
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        listed = (await c.call_tool("list_files", {})).data
    assert listed["unlistedFolders"] == []
    assert [f["state"] for f in listed["files"]] == ["not_ingested"]


LABEL_TAXONOMY = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.type]
values.minutes = { description = "Minutes. Decisions." }
values.report = { description = "Report. Findings." }
"""


@pytest.fixture
def labelled_root(tmp_path):
    root = tmp_path / "docs"
    (root / "alpha").mkdir(parents=True)
    (root / ".minirag").mkdir()
    (root / ".minirag" / "classify.toml").write_text(LABEL_TAXONOMY)
    (root / "alpha" / "plan.md").write_text(
        "# Plan\n\nMinutes of the rollout plan. Date: 14 March 2026."
    )
    (root / "aaa-notes.md").write_text("# Notes\n\nReport on rollout notes. Date: 1 January 2026.")
    return root


async def _ingest_both(env, embedder, model_factory):
    """Label the corpus with the fake model, then forget that it was ever made, so a test
    can assert on what happens afterwards alone."""
    root = Path(env["BASE_DIR"])
    model_factory.answers = {"Minutes of": {"minutes": 1.0}}
    model_factory.default = {"report": 1.0}
    async with Client(
        create_app(load_config(env), embedder=embedder, model_factory=model_factory)
    ) as c:
        for rel in ("alpha/plan.md", "aaa-notes.md"):
            await c.call_tool("ingest_file", {"filePath": str(root / rel)})
    assert len(model_factory.created) >= 1
    model_factory.calls.clear()
    model_factory.created.clear()


async def test_a_label_filter_returns_the_same_sources_and_counts_from_both(
    labelled_root, tmp_path, capsys, monkeypatch, fake_embedder, model_factory
):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    env = {"BASE_DIR": str(labelled_root), "DB_PATH": str(tmp_path / "db")}
    await _ingest_both(env, fake_embedder, model_factory)
    plan = str(labelled_root / "alpha" / "plan.md")
    notes = str(labelled_root / "aaa-notes.md")
    async with Client(
        create_app(load_config(env), embedder=fake_embedder, model_factory=model_factory)
    ) as c:
        res = (
            await c.call_tool(
                "query_documents",
                {"query": "rollout", "labels": {"type": "minutes"}, "dateFrom": "2026-03-01"},
            )
        ).data
        listed = (
            await c.call_tool(
                "list_files", {"labels": {"type": ["minutes", "report"]}, "sort": "date"}
            )
        ).data
    code, out, _ = run_cli(
        ["query", "rollout", "--label", "type=minutes", "--date-from", "2026-03-01"]
        + [*_flags(env), "--json"],
        capsys,
    )
    code_l, out_l, _ = run_cli(
        ["list", "--label", "type=minutes", "--label", "type=report", "--sort", "date"]
        + [*_flags(env), "--json"],
        capsys,
    )
    assert code == 0 and code_l == 0
    cli_res, cli_listed = json.loads(out), json.loads(out_l)
    assert [s["source"] for s in res["sources"]] == [plan]
    assert [s["source"] for s in cli_res["sources"]] == [plan]
    assert cli_res["sources"] == res["sources"]
    assert cli_res["filterCounts"] == res["filterCounts"]
    assert cli_listed == listed
    assert [f["source"] for f in listed["files"]] == [plan, notes]  # newest first


def _last_error(err: str) -> str:
    return err.strip().splitlines()[-1].removeprefix("error: ")


async def test_a_filter_without_classify_toml_is_refused_the_same_way_by_both(
    root, tmp_path, capsys, monkeypatch, fake_embedder, model_factory
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    is_error, text = await call_mcp(
        env, "query_documents", {"query": "alpha", "labels": {"type": "spec"}}, fake_embedder
    )
    code, _, err = run_cli(["query", "alpha", "--label", "type=spec", *_flags(env)], capsys)
    is_error_l, text_l = await call_mcp(
        env, "list_files", {"dateFrom": "2026-01-01"}, fake_embedder
    )
    code_l, _, err_l = run_cli(["list", "--date-from", "2026-01-01", *_flags(env)], capsys)
    assert is_error and code == 1 and text.endswith(_last_error(err))
    assert is_error_l and code_l == 1 and text_l.endswith(_last_error(err_l))
    assert model_factory.calls == []


async def test_status_reports_the_same_labels_block_in_both(
    labelled_root, tmp_path, capsys, monkeypatch, fake_embedder, model_factory
):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    env = {"BASE_DIR": str(labelled_root), "DB_PATH": str(tmp_path / "db")}
    await _ingest_both(env, fake_embedder, model_factory)
    async with Client(
        create_app(load_config(env), embedder=fake_embedder, model_factory=model_factory)
    ) as c:
        st = (await c.call_tool("status", {})).data
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    assert code == 0
    assert st["labels"]["facets"]["type"]["values"]["minutes"]["count"] == 1
    assert json.loads(out)["labels"] == st["labels"]


async def test_status_has_no_labels_key_without_the_file(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    _, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    assert "labels" not in st and "labels" not in json.loads(out)


async def test_status_list_and_query_never_make_the_model(
    labelled_root, tmp_path, monkeypatch, fake_embedder, model_factory
):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    env = {"BASE_DIR": str(labelled_root), "DB_PATH": str(tmp_path / "db")}
    await _ingest_both(env, fake_embedder, model_factory)
    async with Client(
        create_app(load_config(env), embedder=fake_embedder, model_factory=model_factory)
    ) as c:
        await c.call_tool("status", {})
        listed = (await c.call_tool("list_files", {"labels": {"type": "minutes"}})).data
        res = (
            await c.call_tool(
                "query_documents", {"query": "rollout", "labels": {"type": "minutes"}}
            )
        ).data
    assert len(listed["files"]) == 1 and len(res["sources"]) == 1
    assert model_factory.calls == [] and model_factory.created == []


BROKEN_TAXONOMIES = {
    "default-is-a-list": "[facets.type]\n"
    'default = ["minutes"]\n'
    'values.minutes = { description = "Minutes." }\n',
    "home-of-an-unknown-user": LABEL_TAXONOMY.replace(
        "/models/fake.gguf", "~nosuchuser_zz/fake.gguf"
    ),
}


@pytest.mark.parametrize("text", BROKEN_TAXONOMIES.values(), ids=BROKEN_TAXONOMIES.keys())
async def test_a_taxonomy_that_fails_validation_breaks_no_tool_in_either_shell(
    labelled_root, tmp_path, capsys, monkeypatch, fake_embedder, model_factory, text
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    taxonomy = labelled_root / ".minirag" / "classify.toml"
    taxonomy.write_text(text)
    env = {"BASE_DIR": str(labelled_root), "DB_PATH": str(tmp_path / "db")}
    async with Client(
        create_app(load_config(env), embedder=fake_embedder, model_factory=model_factory)
    ) as c:
        await c.call_tool("ingest_data", {"data": "Rollout notes.", "source": "note-1"})
        st = (await c.call_tool("status", {})).data
        res = (await c.call_tool("query_documents", {"query": "rollout"})).data
        listed = (await c.call_tool("list_files", {})).data
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    code_q, out_q, _ = run_cli(["query", "rollout", *_flags(env), "--json"], capsys)
    code_l, out_l, _ = run_cli(["list", *_flags(env), "--json"], capsys)

    assert code == code_q == code_l == 0
    cli_st = json.loads(out)
    for status in (st, cli_st):
        assert "indexError" not in status
        assert status["labels"]["file"] == str(taxonomy)
        assert status["labels"]["error"].startswith(f"{taxonomy}: ")
    assert cli_st["labels"] == st["labels"]
    assert [s["source"] for s in res["sources"]] == ["note-1"]
    assert json.loads(out_q)["sources"] == res["sources"]
    assert json.loads(out_l)["files"] == listed["files"]
    assert model_factory.calls == []


async def test_a_labels_failure_in_status_is_a_labels_error_not_an_index_error(
    labelled_root, tmp_path, capsys, monkeypatch, fake_embedder, model_factory
):
    """The index is fine; saying otherwise sends the user to the wrong problem."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)

    def broken(self):
        raise RuntimeError("cannot tell the model state")

    monkeypatch.setattr(Labeler, "model_state", broken)
    env = {"BASE_DIR": str(labelled_root), "DB_PATH": str(tmp_path / "db")}
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)

    assert code == 0
    cli_st = json.loads(out)
    for status in (st, cli_st):
        assert "indexError" not in status
        assert status["chunkCount"] == 0
        assert status["labels"] == {
            "file": str(labelled_root / ".minirag" / "classify.toml"),
            "error": "RuntimeError: cannot tell the model state",
        }


async def test_find_returns_the_same_payload_through_both(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    (root / "sub" / "b.md").write_text("# Beta\n\nТокены и авторизация, Alpha.")
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    code, _, _ = run_cli(["sync", *_flags(env)], capsys)
    assert code == 0
    args = {"terms": ["alpha", "auth"], "match": "any"}
    is_error, mcp_payload = await call_mcp(env, "find_mentions", args, fake_embedder)
    code, out, _ = run_cli(["find", "alpha", "auth", "--any", *_flags(env), "--json"], capsys)
    assert not is_error and code == 0
    assert json.loads(out) == mcp_payload
    assert mcp_payload["matchedDocuments"] == 2


async def test_an_empty_term_is_refused_by_both(root, tmp_path, capsys, monkeypatch, fake_embedder):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    is_error, text = await call_mcp(env, "find_mentions", {"terms": " "}, fake_embedder)
    code, _, err = run_cli(["find", " ", *_flags(env)], capsys)
    assert is_error and "a term must not be empty" in text
    assert code == 1 and "a term must not be empty" in err
