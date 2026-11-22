"""The error and empty-result paths, run through both shells.

Parity between the MCP server and the CLI is otherwise a convention, and two of these
paths had drifted (minirag-mcp-jd6): each case here asks both surfaces the same thing and
checks that the caller is told the same thing.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest
from fastmcp import Client

import minirag_mcp.cli as cli
import minirag_mcp.embedder as embedder_mod
from conftest import FakeEmbedder, stamp_identity
from index_helpers import file_registry, run_meta
from lookalike_texts import MINUTES, REGULATION_V1, REGULATION_V2
from minirag_mcp import ocr, ocr_vision
from minirag_mcp.config import default_db_path, load_config
from minirag_mcp.embedder import E5_LARGE, Embedder, EmbedderUnavailableError
from minirag_mcp.labels import Labeler
from minirag_mcp.labels.model import ModelError
from minirag_mcp.registry import UNRESTRICTED, Connection
from minirag_mcp.server import create_app
from minirag_mcp.store import DOCUMENT_TABLE, SIGNATURE_TABLE, Store


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


async def test_status_is_identical_in_both_shells(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    code, _, _ = run_cli(["sync", *_flags(env)], capsys)
    assert code == 0
    is_error, st = await call_mcp(env, "status", {}, fake_embedder)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    assert not is_error and code == 0
    assert (st["sourceCount"], "indexError" in st) == (1, False)
    assert json.loads(out) == st


async def test_status_reports_the_identity_the_same_in_both_shells(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    assert code == 0 and json.loads(out) == st
    assert (st["indexState"], st["indexGeneration"], st["chunkBudget"]) == (
        "ready",
        1,
        load_config(env).token_budget,
    )
    assert st["embedding"] == {
        "model": "fake",
        "prefixes": "none",
        "dim": 8,
        "backend": "in-process",
    }
    assert "rebuildReasons" not in st and "indexError" not in st


async def test_both_shells_refuse_a_1x_index_naming_the_v2_path(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    db = tmp_path / "old"
    Store(db, dim=fake_embedder.dim)
    before = sorted(p.relative_to(db) for p in db.rglob("*"))
    env = {"BASE_DIR": str(root), "DB_PATH": str(db)}
    v2 = str(default_db_path(load_config(env).roots))
    is_error, text = await call_mcp(env, "query_documents", {"query": "alpha"}, fake_embedder)
    code, _, err = run_cli(["query", "alpha", *_flags(env)], capsys)
    refusal = _last_error(err)
    assert is_error and code == 1
    assert v2 in refusal and "1.x" in refusal and text.endswith(refusal)
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    assert code == 0 and json.loads(out) == st
    assert st["indexState"] == "unavailable"
    assert refusal == f"Index 'default' is unavailable: {st['indexError']}"
    assert "chunkCount" not in st and "indexGeneration" not in st
    assert sorted(p.relative_to(db) for p in db.rglob("*")) == before


async def test_both_shells_refuse_an_ingest_into_a_pending_index_the_same_way(
    root, tmp_path, capsys, monkeypatch, fake_embedder, public_dns
):
    """It answers from its existing vectors and lets a document be deleted, but takes no
    ingest until an unscoped sync rebuilds it."""
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nNotes.", "R"))
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    (root / "sub" / "b.md").write_text("# Beta\n\nBeta body about tokens.")
    a_md, b_md = str(root / "sub" / "a.md"), str(root / "sub" / "b.md")
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    budget = load_config(env).token_budget
    bigger = budget + 90
    monkeypatch.setenv("CHUNK_TOKEN_BUDGET", str(bigger))
    env["CHUNK_TOKEN_BUDGET"] = str(bigger)
    reason = f"chunk budget {budget} -> {bigger} tokens"

    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        refused = [
            await c.call_tool("ingest_file", {"filePath": a_md}, raise_on_error=False),
            await c.call_tool("ingest_data", {"data": "x", "source": "s"}, raise_on_error=False),
            await c.call_tool("ingest_url", {"url": "https://example.com/p"}, raise_on_error=False),
        ]
        found = (await c.call_tool("query_documents", {"query": "alpha"})).data
        st = (await c.call_tool("status", {})).data
    cli_refusals = []
    for args in (["ingest", a_md], ["ingest-url", "https://example.com/q"]):
        code, _, err = run_cli([*args, *_flags(env)], capsys)
        assert code == 1
        cli_refusals.append(_last_error(err))
    code_s, out_s, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    code_q, out_q, _ = run_cli(["query", "beta", *_flags(env), "--json"], capsys)
    _, deleted = await call_mcp(env, "delete_file", {"filePath": a_md}, fake_embedder)
    code_d, out_d, _ = run_cli(["delete", b_md, *_flags(env), "--json"], capsys)

    [refusal] = set(cli_refusals)
    assert "index needs a rebuild: run an unscoped sync" in refusal and reason in refusal
    assert all(r.is_error and r.content[0].text.endswith(refusal) for r in refused)
    assert code_s == 0 and json.loads(out_s) == st
    assert (st["indexState"], st["rebuildReasons"]) == ("rebuild pending", [reason])
    assert st["chunkBudget"] == budget
    assert found["sources"][0]["source"] == a_md
    assert code_q == 0 and json.loads(out_q)["sources"][0]["source"] == b_md
    assert deleted["deletedChunks"] > 0
    assert code_d == 0 and json.loads(out_d)["deletedChunks"] > 0


async def _finished_job(client, job_id: str) -> dict:
    for _ in range(3000):
        status = (await client.call_tool("sync_status", {"jobId": job_id})).data
        if status["state"] in ("succeeded", "failed"):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("the sync job did not finish")


async def test_both_shells_report_the_rebuild(root, tmp_path, capsys, monkeypatch, fake_embedder):
    """An unscoped sync of a pending index rebuilds it, and both shells say so alike."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    envs = {
        shell: {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / shell)}
        for shell in ("cli", "mcp", "human")
    }
    for env in envs.values():
        assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    budget = load_config(envs["cli"]).token_budget
    bigger = budget + 90
    monkeypatch.setenv("CHUNK_TOKEN_BUDGET", str(bigger))
    for env in envs.values():
        env["CHUNK_TOKEN_BUDGET"] = str(bigger)
    reason = f"chunk budget {budget} -> {bigger} tokens"

    code, out, _ = run_cli(["sync", *_flags(envs["cli"]), "--json"], capsys)
    async with Client(create_app(load_config(envs["mcp"]), embedder=fake_embedder)) as c:
        job = (await c.call_tool("sync_start", {})).data["jobId"]
        status = await _finished_job(c, job)
    code_h, out_h, _ = run_cli(["sync", *_flags(envs["human"])], capsys)

    reported = json.loads(out)
    assert code == 0 and status["state"] == "succeeded", status["error"]
    assert reported["rebuild"] == status["rebuild"]
    assert status["rebuild"] == {
        "kind": "reingest",
        "reasons": [reason],
        "generation": 2,
        "reembedded": 0,
        "restored": 0,
    }
    assert reported["counts"] == status["counts"] and reported["errors"] == status["errors"]
    assert code_h == 0 and out_h.strip().endswith(f"; rebuilt (reingest): {reason}")
    code, out, _ = run_cli(["sync", *_flags(envs["cli"]), "--json"], capsys)
    assert code == 0 and json.loads(out)["rebuild"] is None
    _, st = await call_mcp(envs["mcp"], "status", {}, fake_embedder)
    assert (st["indexState"], st["indexGeneration"], st["chunkBudget"]) == ("ready", 2, bigger)


async def test_both_shells_refuse_a_scoped_sync_of_a_pending_index_the_same_way(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    bigger = load_config(env).token_budget + 90
    monkeypatch.setenv("CHUNK_TOKEN_BUDGET", str(bigger))
    env["CHUNK_TOKEN_BUDGET"] = str(bigger)
    a_md = str(root / "sub" / "a.md")

    code, out, err = run_cli(["sync", a_md, *_flags(env)], capsys)
    is_error, text = await call_mcp(env, "sync_start", {"path": a_md}, fake_embedder)

    refusal = _last_error(err)
    assert code == 1 and out == "" and is_error and text.endswith(refusal)
    assert refusal.startswith("Index 'default': ")
    assert "index needs a rebuild: run an unscoped sync" in refusal and a_md in refusal
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    assert (st["indexState"], st["indexGeneration"]) == ("rebuild pending", 1)


async def test_a_parallel_sync_counts_the_same_through_both_shells(
    root, tmp_path, capsys, monkeypatch, fake_embedder, inline_workers
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    # With the [ocr] extra installed, a parallel sync fetches Tesseract's data first.
    monkeypatch.setattr(ocr, "available", lambda: False)
    for i in range(7):
        (root / f"doc{i}.md").write_text(f"# Doc {i}\n\nBody of document {i}.")
    monkeypatch.setenv("RAG_SYNC_WORKERS", "2")
    envs = {
        shell: {
            "BASE_DIR": str(root),
            "DB_PATH": str(tmp_path / shell),
            "CACHE_DIR": str(tmp_path / "cache"),
            "RAG_SYNC_WORKERS": "2",
        }
        for shell in ("cli", "mcp")
    }

    code, out, _ = run_cli(["sync", *_flags(envs["cli"]), "--json"], capsys)
    async with Client(create_app(load_config(envs["mcp"]), embedder=fake_embedder)) as c:
        job = (await c.call_tool("sync_start", {})).data["jobId"]
        status = await _finished_job(c, job)

    cli_counts = json.loads(out)["counts"]
    assert code == 0 and status["state"] == "succeeded", status["error"]
    assert cli_counts == status["counts"] and cli_counts["ingested"] == 8
    assert inline_workers == [2, 2]


async def test_an_empty_query_is_refused_by_both(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    is_error, text = await call_mcp(env, "query_documents", {"query": "   "}, fake_embedder)
    code, _, err = run_cli(["query", "   ", *_flags(env)], capsys)
    assert is_error and "query must not be empty" in text
    assert code == 1 and "query must not be empty" in err


@pytest.mark.parametrize(
    "tool,args,command",
    [
        ("query_documents", {"query": ""}, ["query", ""]),
        ("find_mentions", {"terms": ["", "alpha"]}, ["find", "", "alpha"]),
    ],
)
async def test_an_empty_argument_reaches_the_core_through_the_entry_point(
    root, tmp_path, capsys, monkeypatch, fake_embedder, tool, args, command
):
    """The options around a command are read first, and an empty argument passes on to the
    command as it is, to be refused as the server refuses it."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    is_error, text = await call_mcp(env, tool, args, fake_embedder)
    code, out, err = run_meta([*command, *_flags(env), "--json"], capsys)
    assert is_error and (code, out) == (1, "") and text == _last_error(err)
    assert text in ("query must not be empty", "a term must not be empty")


@pytest.mark.parametrize(
    "tesseract,macos,bindings,below,expected",
    [
        (True, True, True, None, "vision"),
        (True, True, True, "70", "vision"),
        (True, True, False, None, "unavailable"),
        (True, True, True, "0", "off"),
        (True, True, False, "0", "off"),
        (True, False, False, None, None),
        (True, False, False, "0", None),
        (False, True, True, None, None),
    ],
)
async def test_status_names_the_second_reader_the_same_in_both_shells(
    root, tmp_path, capsys, monkeypatch, fake_embedder, tesseract, macos, bindings, below, expected
):
    """Present only where Apple Vision could be: on macOS, and only beside an OCR engine
    for it to second. RAG_OCR_SECOND_READER_BELOW=0 is the owner turning it off."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setattr(ocr, "available", lambda: tesseract)
    monkeypatch.setattr(ocr_vision, "supported", lambda: macos)
    monkeypatch.setattr(ocr_vision, "available", lambda: macos and bindings)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    if below is not None:
        env["RAG_OCR_SECOND_READER_BELOW"] = below
        monkeypatch.setenv("RAG_OCR_SECOND_READER_BELOW", below)
    is_error, st = await call_mcp(env, "status", {}, fake_embedder)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    assert not is_error and code == 0 and json.loads(out) == st
    assert st.get("ocrSecondReader", None) == expected


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


async def test_an_unknown_model_is_refused_the_same_way_by_both(root, tmp_path, capsys):
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db"), "MODEL_NAME": "x/unknown"}
    is_error, text = await call_mcp(env, "query_documents", {"query": "alpha"}, None)
    code, _, err = run_cli(["query", "alpha", *_flags(env), "--model-name", "x/unknown"], capsys)
    refusal = _last_error(err)
    assert is_error and code == 1 and text.endswith(refusal)
    assert refusal.startswith("Index 'default' is unavailable: Model 'x/unknown' is not in")
    assert refusal.endswith(" — see fastembed's supported models.")


async def test_an_index_recorded_at_another_width_is_refused_the_same_way_by_both(
    root, tmp_path, capsys
):
    env = {
        "BASE_DIR": str(root),
        "DB_PATH": str(tmp_path / "db"),
        "CACHE_DIR": str(tmp_path / "models"),
    }
    stamp_identity(load_config(env), FakeEmbedder(E5_LARGE, dim=16))
    is_error, text = await call_mcp(env, "query_documents", {"query": "alpha"}, None)
    code, _, err = run_cli(["query", "alpha", *_flags(env)], capsys)
    refusal = _last_error(err)
    assert is_error and code == 1 and text.endswith(refusal)
    assert f"built with {E5_LARGE}, at 16 dimensions, but it makes 1024-dimensional" in refusal


async def test_both_shells_report_the_embedding_backend(
    root, tmp_path, capsys, monkeypatch, fake_llama
):
    monkeypatch.setenv("EMBED_URL", fake_llama.url("/v1/embeddings"))
    monkeypatch.setenv("EMBED_ENDPOINT_MODEL", "e5")
    env = {
        "BASE_DIR": str(root),
        "DB_PATH": str(tmp_path / "db"),
        "CACHE_DIR": str(tmp_path / "models"),
        **{k: os.environ[k] for k in ("EMBED_URL", "EMBED_ENDPOINT_MODEL")},
    }
    is_error, st = await call_mcp(env, "status", {}, None)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    cli_st = json.loads(out)
    assert not is_error and code == 0 and "indexError" not in st
    assert st["embeddingBackend"] == "endpoint" and "embeddingWarning" not in st
    assert {k: v for k, v in cli_st.items() if k.startswith("embedding")} == {
        k: v for k, v in st.items() if k.startswith("embedding")
    }
    assert fake_llama.requests == []  # status asks the endpoint nothing


async def test_both_shells_pass_on_an_embedding_fallback(
    root, tmp_path, capsys, monkeypatch, public_dns
):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nNotes.", "R"))
    fallen = FakeEmbedder()
    fallen.backend, fallen.warning = "in-process (endpoint unreachable)", "W"
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fallen)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    async with Client(create_app(load_config(env), embedder=fallen)) as c:
        replies = [
            (await c.call_tool("ingest_file", {"filePath": str(root / "sub" / "a.md")})).data,
            (await c.call_tool("ingest_data", {"data": "x", "source": "s"})).data,
            (await c.call_tool("ingest_url", {"url": "https://example.com/p"})).data,
        ]
        found = (await c.call_tool("query_documents", {"query": "alpha"})).data
        st = (await c.call_tool("status", {})).data
    _, out, _ = run_cli(["ingest", str(root / "sub" / "a.md"), *_flags(env), "--json"], capsys)
    replies.append(json.loads(out)["ingested"][0])
    _, out, _ = run_cli(["ingest-url", "https://example.com/q", *_flags(env), "--json"], capsys)
    replies.append(json.loads(out))
    _, out, _ = run_cli(["query", "alpha", *_flags(env), "--json"], capsys)
    cli_found = json.loads(out)
    _, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    cli_st = json.loads(out)
    assert [r["warnings"] for r in replies] == [["W"]] * 5
    for payload in (found, cli_found):
        assert payload["embeddingBackend"] == "in-process (endpoint unreachable)"
        assert payload["warnings"] == ["W"]
    for status in (st, cli_st):
        assert status["embeddingBackend"] == "in-process (endpoint unreachable)"
        assert status["embeddingWarning"] == "W"


async def test_reranked_queries_are_identical_in_both_shells(
    root, tmp_path, capsys, monkeypatch, fake_embedder, fake_llama
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    (root / "sub" / "b.md").write_text("# Beta\n\nAlpha tokens and auth in a second note.")
    monkeypatch.setenv("RERANK_URL", fake_llama.url("/v1/rerank"))
    monkeypatch.setenv("RERANK_MODEL", "rr")
    env = {
        "BASE_DIR": str(root),
        "DB_PATH": str(tmp_path / "db"),
        **{k: os.environ[k] for k in ("RERANK_URL", "RERANK_MODEL")},
    }
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    args = {"query": "alpha tokens"}
    is_error, found = await call_mcp(env, "query_documents", args, fake_embedder)
    code, out, _ = run_cli(["query", "alpha tokens", *_flags(env), "--json"], capsys)
    assert not is_error and code == 0
    assert found["reranked"] is True and "rerankScore" in found["results"][0]
    assert json.loads(out) == found
    is_error, st = await call_mcp(env, "status", {}, fake_embedder)
    _, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    assert st["reranker"] == "endpoint" and json.loads(out) == st


async def test_a_skipped_reranker_is_reported_by_both_shells(
    root, tmp_path, capsys, monkeypatch, fake_embedder, dead_url
):
    """The human output of the CLI too: a reply's `warnings` are not left to --json."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setenv("RERANK_URL", dead_url + "/v1/rerank")
    monkeypatch.setenv("RERANK_MODEL", "rr")
    env = {
        "BASE_DIR": str(root),
        "DB_PATH": str(tmp_path / "db"),
        **{k: os.environ[k] for k in ("RERANK_URL", "RERANK_MODEL")},
    }
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    _, found = await call_mcp(env, "query_documents", {"query": "alpha"}, fake_embedder)
    code, out, err = run_cli(["query", "alpha", *_flags(env)], capsys)

    def unaddressed(text: str) -> str:
        return re.sub(r"0x[0-9a-f]+", "0x", text)

    [warning] = found["warnings"]
    assert warning.startswith("Reranking was skipped") and dead_url in warning
    assert code == 0 and "alpha" in out.lower()
    assert unaddressed(f"warn: {warning}") in [unaddressed(line) for line in err.splitlines()]


async def test_the_human_output_of_an_ingest_carries_its_warnings(
    root, tmp_path, capsys, monkeypatch, public_dns
):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nNotes.", "R"))
    fallen = FakeEmbedder()
    fallen.backend, fallen.warning = "in-process (endpoint unreachable)", "W"
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fallen)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    for args in (["ingest", str(root / "sub" / "a.md")], ["ingest-url", "https://example.com/q"]):
        code, _, err = run_cli([*args, *_flags(env)], capsys)
        assert code == 0 and err.splitlines().count("warn: W") == 1, args


async def test_both_shells_refuse_a_delete_while_the_index_is_rebuilt(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    from minirag_mcp.identity import begin_staging
    from minirag_mcp.lock import sync_lock

    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    a_md = str(root / "sub" / "a.md")
    with sync_lock(Path(env["DB_PATH"])):
        begin_staging(Path(env["DB_PATH"]))
        is_error, text = await call_mcp(env, "delete_file", {"filePath": a_md}, fake_embedder)
        code, out, err = run_cli(["delete", a_md, *_flags(env)], capsys)

    refusal = _last_error(err)
    assert is_error and code == 1 and out == ""
    assert "is being rebuilt into" in refusal and text == refusal
    assert refusal.startswith("Index 'default': ")


async def test_both_shells_refuse_an_ingest_while_the_index_is_rebuilt(
    root, tmp_path, capsys, monkeypatch, fake_embedder, public_dns
):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.identity import begin_staging
    from minirag_mcp.ingest.parser import ParsedDoc
    from minirag_mcp.lock import sync_lock

    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nNotes.", "R"))
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    a_md = str(root / "sub" / "a.md")
    with sync_lock(Path(env["DB_PATH"])):
        begin_staging(Path(env["DB_PATH"]))
        async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
            by_file = await c.call_tool("ingest_file", {"filePath": a_md}, raise_on_error=False)
            by_url = await c.call_tool(
                "ingest_url", {"url": "https://example.com/p"}, raise_on_error=False
            )
        code, out, _ = run_cli(["ingest", a_md, *_flags(env), "--json"], capsys)
        code_u, _, err_u = run_cli(["ingest-url", "https://example.com/q", *_flags(env)], capsys)

    [failure] = json.loads(out)["failed"]
    refusal = by_file.content[0].text
    assert by_file.is_error and code == 1 and failure["error"] == refusal
    assert by_url.is_error and code_u == 1 and by_url.content[0].text == _last_error(err_u)
    assert refusal.startswith("Index 'default': ") and "is being rebuilt into" in refusal
    assert by_url.content[0].text == refusal


async def test_a_rebuild_that_fails_in_its_job_is_reported_alike_and_unwrapped_by_both(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    """Only what refuses a sync before it starts names the index: a rebuild that fails in
    the job is the job's failure, and the index still answers from what it holds."""

    class Unverified(FakeEmbedder):
        prefixes = "e5"

        def reference_error(self, vector):
            return "the endpoint serves another model"

    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    envs = {shell: {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / shell)} for shell in "ab"}
    for env in envs.values():
        assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    other = Unverified()
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: other)

    code, _, err = run_cli(["sync", *_flags(envs["a"])], capsys)
    async with Client(create_app(load_config(envs["b"]), embedder=other)) as c:
        job = (await c.call_tool("sync_start", {})).data["jobId"]
        status = await _finished_job(c, job)

    assert code == 1 and status["state"] == "failed"
    assert _last_error(err) == status["error"] == "the endpoint serves another model"


async def test_a_path_outside_the_roots_is_refused_the_same_way_by_both(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    elsewhere = str(tmp_path / "elsewhere.md")
    is_error, text = await call_mcp(env, "read_file", {"filePath": elsewhere}, fake_embedder)
    code, _, err = run_cli(["read", elsewhere, *_flags(env)], capsys)
    assert is_error and code == 1 and text == _last_error(err)
    assert text.startswith(f"Path outside configured document roots: {elsewhere}.")
    assert "BASE_DIRS" in text


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

    monkeypatch.setattr(Labeler, "model_status", broken)
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


async def test_an_ingest_says_why_its_labels_are_missing_in_both_shells(
    labelled_root, tmp_path, capsys, monkeypatch, fake_embedder, model_factory
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    model_factory.error = ModelError("could not download fake.gguf: offline")
    env = {"BASE_DIR": str(labelled_root), "DB_PATH": str(tmp_path / "db")}
    async with Client(
        create_app(load_config(env), embedder=fake_embedder, model_factory=model_factory)
    ) as c:
        by_file = (
            await c.call_tool("ingest_file", {"filePath": str(labelled_root / "aaa-notes.md")})
        ).data
        by_data = (await c.call_tool("ingest_data", {"data": "Notes.", "source": "n-1"})).data
    code, out, _ = run_cli(
        ["ingest", str(labelled_root / "alpha" / "plan.md"), *_flags(env), "--json"], capsys
    )
    expected = (
        "stored without a type label: could not download fake.gguf: offline. "
        "A full sync asks again."
    )
    assert by_file["labelProblem"] == by_data["labelProblem"] == expected
    assert code == 0 and json.loads(out)["ingested"][0]["labelProblem"] == expected


async def test_an_ingest_without_labels_keeps_its_reply_shape(root, tmp_path, fake_embedder):
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    is_error, reply = await call_mcp(
        env, "ingest_file", {"filePath": str(root / "sub" / "a.md")}, fake_embedder
    )
    assert not is_error and set(reply) == {"index", "source", "chunkCount", "title"}


async def test_every_json_payload_of_the_cli_names_its_index(
    labelled_root, tmp_path, capsys, monkeypatch, fake_embedder, model_factory, public_dns
):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nNotes.", "R"))
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    model_factory.answers = {"Minutes of": {"minutes": 1.0}}
    model_factory.default = {"report": 1.0}
    env = {"BASE_DIR": str(labelled_root), "DB_PATH": str(tmp_path / "db")}
    plan = str(labelled_root / "alpha" / "plan.md")
    commands = [
        ["sync"],
        ["ingest", plan],
        ["ingest-url", "https://example.com/q"],
        ["query", "rollout"],
        ["find", "rollout"],
        ["read-neighbors", "--file-path", plan, "--chunk-index", "0"],
        ["read", plan],
        ["list"],
        ["status"],
        ["classify", "try", plan],
        ["delete", "--source", "https://example.com/q"],
    ]
    named = []
    for command in commands:
        code, out, err = run_cli([*command, *_flags(env), "--json"], capsys)
        assert code == 0, (command, err)
        named.append(json.loads(out).get("index"))
    assert named == ["default"] * len(commands)


async def test_ingest_url_says_why_its_labels_are_missing_in_both_shells(
    labelled_root, tmp_path, capsys, monkeypatch, fake_embedder, model_factory, public_dns
):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nNotes.", "R"))
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    model_factory.error = ModelError("no model here")
    env = {"BASE_DIR": str(labelled_root), "DB_PATH": str(tmp_path / "db")}
    async with Client(
        create_app(load_config(env), embedder=fake_embedder, model_factory=model_factory)
    ) as c:
        reply = (await c.call_tool("ingest_url", {"url": "https://example.com/p"})).data
    code, out, _ = run_cli(["ingest-url", "https://example.com/q", *_flags(env), "--json"], capsys)
    assert "no model here" in reply["labelProblem"]
    assert code == 0 and json.loads(out)["labelProblem"] == reply["labelProblem"]


@pytest.fixture
def lookalike_root(tmp_path):
    root = tmp_path / "docs"
    (root / "regs").mkdir(parents=True)
    (root / "archive").mkdir()
    (root / "regs" / "v1.md").write_text(REGULATION_V1)
    (root / "archive" / "v2.md").write_text(REGULATION_V2)
    (root / "minutes.md").write_text(MINUTES)
    return root


async def test_query_and_find_name_the_same_look_alikes_through_both(
    lookalike_root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(lookalike_root), "DB_PATH": str(tmp_path / "db")}
    code, _, _ = run_cli(["sync", *_flags(env)], capsys)
    assert code == 0
    scope = str(lookalike_root / "regs")
    _, res = await call_mcp(
        env, "query_documents", {"query": "заявки на закупку", "scope": scope}, fake_embedder
    )
    _, found = await call_mcp(env, "find_mentions", {"terms": "закупочной"}, fake_embedder)
    code_q, out_q, _ = run_cli(
        ["query", "заявки на закупку", "--scope", scope, *_flags(env), "--json"], capsys
    )
    code_f, out_f, _ = run_cli(["find", "закупочной", *_flags(env), "--json"], capsys)

    assert code_q == code_f == 0
    assert json.loads(out_q) == res and json.loads(out_f) == found
    [row] = res["sources"]
    [alike] = row["lookAlikes"]
    assert alike["displayPath"] == "archive/v2.md" and 0.5 <= alike["overlap"] <= 1
    assert {r["displayPath"]: len(r["lookAlikes"]) for r in found["documents"]} == {
        "regs/v1.md": 1,
        "archive/v2.md": 1,
    }


async def test_status_reports_unsigned_documents_the_same_way_in_both(
    lookalike_root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(lookalike_root), "DB_PATH": str(tmp_path / "db")}
    run_cli(["sync", *_flags(env)], capsys)
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    assert "unsignedSourceCount" not in st and "lookAlikesWarning" not in st

    Store(tmp_path / "db", dim=fake_embedder.dim)._db.drop_table(SIGNATURE_TABLE)
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    cli_st = json.loads(out)
    assert code == 0
    assert st["unsignedSourceCount"] == cli_st["unsignedSourceCount"] == 3
    assert st["lookAlikesWarning"] == cli_st["lookAlikesWarning"]

    code, out, _ = run_cli(["sync", *_flags(env), "--json"], capsys)
    assert code == 0 and json.loads(out)["counts"]["signed"] == 3
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    assert "unsignedSourceCount" not in st


async def test_status_reports_undocumented_documents_the_same_way_in_both(
    lookalike_root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(lookalike_root), "DB_PATH": str(tmp_path / "db")}
    run_cli(["sync", *_flags(env)], capsys)
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    assert "undocumentedSourceCount" not in st and "documentsWarning" not in st

    Store(tmp_path / "db", dim=fake_embedder.dim)._db.open_table(DOCUMENT_TABLE).delete("true")
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    cli_st = json.loads(out)
    assert code == 0
    assert st["undocumentedSourceCount"] == cli_st["undocumentedSourceCount"] == 3
    assert st["documentsWarning"] == cli_st["documentsWarning"]

    code, _, _ = run_cli(["sync", *_flags(env)], capsys)
    assert code == 0
    _, st = await call_mcp(env, "status", {}, fake_embedder)
    assert "undocumentedSourceCount" not in st


# --- file mode: indexes.toml, `--index` in the CLI and `index` in the server ----------


@pytest.fixture
def file_mode(indexes_file, monkeypatch, fake_embedder):
    """Both shells over `indexes_file`: the CLI through --config, the server through a
    registry of the same file, each embedding with the fake."""
    monkeypatch.setenv("HOME", indexes_file.env["HOME"])
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)

    def server(connection: Connection = UNRESTRICTED):
        registry = file_registry(indexes_file, lambda c, s: fake_embedder)
        return create_app(registry, connection=connection)

    return server


async def _call(app, tool: str, args: dict) -> tuple[bool, object]:
    async with Client(app) as c:
        r = await c.call_tool(tool, args, raise_on_error=False)
    return r.is_error, (r.content[0].text if r.is_error else r.data)


async def test_a_named_index_answers_the_same_through_both(indexes_file, capsys, file_mode):
    cfg = ["--config", str(indexes_file.path)]
    assert run_meta([*cfg, "--index", "beta", "sync"], capsys)[0] == 0
    is_error, mcp_payload = await _call(
        file_mode(Connection("alpha", None)),
        "query_documents",
        {"query": "bananas", "index": "beta"},
    )
    code, out, _ = run_meta([*cfg, "--index", "beta", "query", "bananas", "--json"], capsys)
    assert not is_error and code == 0
    assert json.loads(out) == mcp_payload and mcp_payload["index"] == "beta"
    assert mcp_payload["sources"]


async def test_an_unknown_index_is_refused_the_same_way_by_both(indexes_file, capsys, file_mode):
    cfg = ["--config", str(indexes_file.path)]
    is_error, text = await _call(file_mode(), "query_documents", {"query": "x", "index": "delta"})
    code, _, err = run_meta([*cfg, "--index", "delta", "query", "x"], capsys)
    assert is_error and code == 1 and text.endswith(_last_error(err))
    assert _last_error(err) == "Unknown index 'delta'. Reachable indexes: alpha, beta, gamma."


async def test_list_indexes_matches_the_indexes_command(indexes_file, capsys, file_mode):
    cfg = ["--config", str(indexes_file.path)]
    assert run_meta([*cfg, "--index", "alpha", "sync"], capsys)[0] == 0
    # A row names the model its index records, and opening beta and gamma to list them
    # records the fake's: the first listing names their configured models.
    assert run_meta([*cfg, "indexes"], capsys)[0] == 0
    is_error, mcp_payload = await _call(file_mode(), "list_indexes", {})
    code, out, _ = run_meta([*cfg, "indexes", "--json"], capsys)
    assert not is_error and code == 0
    assert json.loads(out) == mcp_payload
    assert [r["documentCount"] for r in mcp_payload["indexes"]] == [1, 0, 0]


async def test_status_of_a_named_index_is_identical_in_both_shells(indexes_file, capsys, file_mode):
    cfg = ["--config", str(indexes_file.path)]
    assert run_meta([*cfg, "--index", "gamma", "sync"], capsys)[0] == 0
    is_error, st = await _call(file_mode(), "status", {"index": "gamma"})
    code, out, _ = run_meta([*cfg, "--index", "gamma", "status", "--json"], capsys)
    assert not is_error and code == 0
    assert json.loads(out) == st and (st["index"], st["sourceCount"]) == ("gamma", 1)


@pytest.mark.parametrize("held_by", ["another index", "no index"])
async def test_a_path_outside_the_named_index_is_refused_the_same_way_by_both(
    indexes_file, capsys, file_mode, held_by
):
    """In file mode the refusal names the index, points at the index whose roots hold the
    file, and never names the environment's BASE_DIRS or a restart."""
    cfg = ["--config", str(indexes_file.path)]
    alpha = indexes_file.roots["alpha"]
    path = (
        indexes_file.roots["beta"] / "beta.md" if held_by == "another index" else indexes_file.path
    )
    is_error, text = await _call(
        file_mode(), "read_file", {"filePath": str(path), "index": "alpha"}
    )
    code, _, err = run_meta([*cfg, "--index", "alpha", "read", str(path)], capsys)
    assert is_error and code == 1 and text == _last_error(err)
    assert text.startswith(f"Index 'alpha': {path} is outside its roots ({alpha})")
    if held_by == "another index":
        assert "under those of index 'beta', which answers for it." in text
    else:
        assert (
            f"a folder is reached by adding it to an index's roots in {indexes_file.path}" in text
        )
    assert text.endswith("Other reachable indexes: beta, gamma.")
    assert "BASE_DIRS" not in text and "restart" not in text


async def test_an_index_of_the_file_recorded_at_another_width_is_refused_alike_by_both(
    indexes_file, capsys, monkeypatch
):
    """Without a stand-in each shell's handles build their own embedders, so the handle's
    refusal reaches both, as in the environment mode."""
    monkeypatch.setenv("HOME", indexes_file.env["HOME"])
    registry = file_registry(indexes_file, None)
    stamp_identity(registry.resolve(UNRESTRICTED, "beta").config, FakeEmbedder(E5_LARGE, dim=16))
    is_error, text = await _call(
        create_app(registry), "query_documents", {"query": "x", "index": "beta"}
    )
    tokens = ["--config", str(indexes_file.path), "--index", "beta", "query", "x"]
    code, _, err = run_meta(tokens, capsys)
    refusal = _last_error(err)
    assert is_error and code == 1 and text == refusal
    assert refusal.startswith("Index 'beta' is unavailable: ")
    assert f"built with {E5_LARGE}, at 16 dimensions, but it makes 1024-dimensional" in refusal


class _QueryFails(FakeEmbedder):
    """Opens its index, then cannot embed a query, as a model that cannot be loaded."""

    def embed_query(self, text: str) -> list[float]:
        raise EmbedderUnavailableError("Embedding model fake could not be loaded: gone")


@pytest.fixture
def failing_beta(indexes_file, monkeypatch):
    """Both shells over `indexes_file`, embedding with `_QueryFails`: the server, and the
    CLI's options that select beta."""
    monkeypatch.setenv("HOME", indexes_file.env["HOME"])
    embedder = _QueryFails()
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: embedder)
    app = create_app(file_registry(indexes_file, lambda c, s: embedder))
    return app, ["--config", str(indexes_file.path), "--index", "beta"]


async def test_a_model_failing_in_a_named_index_names_it_the_same_way_in_both(failing_beta, capsys):
    """With several indexes a failure that is not a refusal must still say which index
    failed."""
    app, beta = failing_beta
    is_error, text = await _call(app, "query_documents", {"query": "x", "index": "beta"})
    code, _, err = run_meta([*beta, "query", "x"], capsys)
    assert is_error and code == 1 and text == _last_error(err)
    assert text == "Index 'beta': Embedding model fake could not be loaded: gone"


async def test_a_file_failing_to_parse_in_a_named_index_names_it_the_same_way_in_both(
    failing_beta, indexes_file, capsys
):
    app, beta = failing_beta
    broken = indexes_file.roots["beta"] / "broken.vtt"
    broken.write_text("WEBVTT\n\nhello there\nno timing\n", encoding="utf-8")
    is_error, text = await _call(app, "ingest_file", {"filePath": str(broken), "index": "beta"})
    code, out, err = run_meta([*beta, "ingest", str(broken), "--json"], capsys)
    [failed] = json.loads(out)["failed"]
    assert is_error and code == 1 and text == failed["error"]
    assert f"warn: {broken}: {text}" in err
    assert text == (
        f"Index 'beta': Failed to convert {broken}: a block that is neither a cue nor a note: "
        "'hello there'"
    )


async def test_a_refused_search_in_a_named_index_names_it_the_same_way_in_both(
    failing_beta, capsys
):
    app, beta = failing_beta
    calls = [
        ("find_mentions", {"terms": " "}, ["find", " "]),
        ("list_files", {"dateFrom": "2026-01-01"}, ["list", "--date-from", "2026-01-01"]),
    ]
    for tool, args, command in calls:
        is_error, text = await _call(app, tool, {**args, "index": "beta"})
        code, _, err = run_meta([*beta, *command], capsys)
        assert is_error and code == 1 and text == _last_error(err), tool
        assert text.startswith("Index 'beta': "), text


async def test_last_sync_is_identical_in_mcp_and_cli(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    is_error, st = await call_mcp(env, "status", {}, fake_embedder)
    code, out, _ = run_cli(["status", *_flags(env), "--json"], capsys)
    assert not is_error and code == 0
    assert st["lastSync"]["ranBy"] == "cli" and st["lastSync"]["outcome"] == "succeeded"
    assert json.loads(out)["lastSync"] == st["lastSync"]
    code, out, _ = run_cli(["status", *_flags(env)], capsys)
    line = next(ln for ln in out.splitlines() if ln.startswith("last sync: "))
    assert "(cli) succeeded, 0 failed, 0 large tables" in line


async def test_last_sync_error_is_identical_in_mcp_and_cli(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    from minirag_mcp.sync_report import SyncRecord, record_sync

    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    bad = str(root / "sub" / "a.md")
    record = SyncRecord(
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:05+00:00",
        "cli",
        None,
        "succeeded",
        None,
        {"failed": 1},
        [{"source": bad, "error": "conversion failed"}],
        None,
    )
    assert record_sync(tmp_path / "db", record) is None
    is_error, listed = await call_mcp(env, "list_files", {}, fake_embedder)
    code, out, _ = run_cli(["list", *_flags(env), "--json"], capsys)
    assert not is_error and code == 0
    assert [f["lastSyncError"] for f in listed["files"]] == ["conversion failed"]
    assert json.loads(out) == listed
    _, human, _ = run_cli(["list", *_flags(env)], capsys)
    assert f"{bad} (" in human and "  ! conversion failed" in human.splitlines()[1:2][0]


async def test_a_real_sync_failure_is_listed_identically_in_mcp_and_cli(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    bad = root / "sub" / "empty.md"
    bad.write_text("")
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 1  # a failed file fails the sync
    is_error, listed = await call_mcp(env, "list_files", {}, fake_embedder)
    code, out, _ = run_cli(["list", *_flags(env), "--json"], capsys)
    assert not is_error and code == 0
    assert [f["source"] for f in listed["files"] if "lastSyncError" in f] == [str(bad)]
    assert json.loads(out) == listed


async def test_an_unreadable_report_is_reported_identically_by_list_in_both(
    root, tmp_path, capsys, monkeypatch, fake_embedder
):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    assert run_cli(["sync", *_flags(env)], capsys)[0] == 0
    (tmp_path / "db" / "last-sync.json").write_text("{truncated")
    is_error, listed = await call_mcp(env, "list_files", {}, fake_embedder)
    code, out, err = run_cli(["list", *_flags(env), "--json"], capsys)
    assert not is_error and code == 0
    why = listed["lastSyncReport"]["unreadable"]
    assert json.loads(out) == listed
    assert all("lastSyncError" not in f for f in listed["files"])
    _, _, err = run_cli(["list", *_flags(env)], capsys)
    assert f"warn: last-sync report unreadable: {why}" in err
