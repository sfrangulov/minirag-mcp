"""The error and empty-result paths, run through both shells.

Parity between the MCP server and the CLI is otherwise a convention, and two of these
paths had drifted (minirag-mcp-jd6): each case here asks both surfaces the same thing and
checks that the caller is told the same thing.
"""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

import minirag_mcp.cli as cli
import minirag_mcp.embedder as embedder_mod
from minirag_mcp.config import load_config
from minirag_mcp.embedder import Embedder
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
