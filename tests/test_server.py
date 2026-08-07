import pytest
from fastmcp import Client

from minirag_mcp.config import load_config
from minirag_mcp.server import create_app

# pytest-asyncio runs in auto mode (see pyproject) — bare `async def` tests are collected as-is.


@pytest.fixture
def app(tmp_path, fake_embedder):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "auth.md").write_text(
        "# Auth Guide\n\nOAuth2 token authentication flow explained at length here."
    )
    (root / "sub").mkdir()
    (root / "sub" / "err.md").write_text(
        "# Errors\n\nThe API returns ERR_CONNECTION_REFUSED when the backend is down."
    )
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
    return create_app(cfg, embedder=fake_embedder), root


async def test_all_eleven_tools_listed(app):
    mcp, _ = app
    async with Client(mcp) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == {
        "sync_start", "sync_status", "ingest_file", "ingest_data", "ingest_url",
        "query_documents", "read_chunk_neighbors", "read_file", "list_files",
        "delete_file", "status",
    }


async def test_sync_then_query_then_neighbors(app):
    mcp, root = app
    async with Client(mcp) as c:
        job = (await c.call_tool("sync_start", {})).data
        for _ in range(100):
            st = (await c.call_tool("sync_status", {"jobId": job["jobId"]})).data
            if st["state"] in ("succeeded", "failed"):
                break
        assert st["state"] == "succeeded" and st["counts"]["ingested"] == 2

        res = (await c.call_tool("query_documents", {"query": "ERR_CONNECTION_REFUSED"})).data
        assert res["results"], "expected at least one search result"
        top = res["results"][0]
        assert {"text", "source", "title", "chunkIndex", "score"} <= set(top)
        assert res["sources"], "expected aggregated sources"
        assert {"source", "title", "hits"} <= set(res["sources"][0])
        assert res["sources"][0]["source"] == top["source"]  # rank order preserved

        nb = (await c.call_tool(
            "read_chunk_neighbors",
            {"filePath": top["source"], "chunkIndex": top["chunkIndex"]},
        )).data
        assert nb["chunks"]


async def test_ingest_file_read_list_delete(app):
    mcp, root = app
    f = root / "new.md"
    f.write_text("# New\n\nFresh content body that is long enough to index.")
    async with Client(mcp) as c:
        r = (await c.call_tool("ingest_file", {"filePath": str(f)})).data
        assert r["source"] == str(f) and r["chunkCount"] >= 1

        full = (await c.call_tool("read_file", {"filePath": str(f)})).data
        assert full["source"] == str(f) and full["sourceType"] == "file"
        assert "Fresh content body" in full["text"]
        assert full["chunkCount"] == r["chunkCount"]

        listed = (await c.call_tool("list_files", {})).data
        by_source = {x["source"]: x for x in listed["files"]}
        assert by_source[str(f)]["state"] == "ingested"
        # auth.md/err.md exist on disk but were not ingested in this test's client session
        assert any(x["state"] == "not_ingested" for x in listed["files"])

        d = (await c.call_tool("delete_file", {"filePath": str(f)})).data
        assert d["deletedChunks"] >= 1
        with pytest.raises(Exception, match="not"):
            await c.call_tool("read_file", {"filePath": str(f)})


async def test_ingest_file_outside_root_is_tool_error(app):
    mcp, _ = app
    async with Client(mcp) as c:
        with pytest.raises(Exception, match="outside"):
            await c.call_tool("ingest_file", {"filePath": "/etc/passwd"})


async def test_ingest_data_and_url(app, monkeypatch):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url: ParsedDoc("# R\n\nRemote body.", "R"))
    mcp, _ = app
    async with Client(mcp) as c:
        r = (await c.call_tool(
            "ingest_data",
            {"data": "# Note\n\nSaved note body.", "source": "note-1", "format": "markdown"},
        )).data
        assert r["source"] == "note-1"
        r2 = (await c.call_tool("ingest_url", {"url": "https://example.com/x"})).data
        assert r2["source"] == "https://example.com/x"
        with pytest.raises(Exception, match="http"):
            await c.call_tool("ingest_url", {"url": "file:///etc/passwd"})


async def test_status_reports_counts(app):
    mcp, root = app
    async with Client(mcp) as c:
        st = (await c.call_tool("status", {})).data
        assert st["roots"] == [str(root)]
        assert "chunkCount" in st and "model" in st


async def test_degraded_mode_status_alive_others_fail(tmp_path, fake_embedder):
    mcp = create_app(None, config_error="BASE_DIRS must be a JSON array", embedder=fake_embedder)
    async with Client(mcp) as c:
        st = (await c.call_tool("status", {})).data
        assert "BASE_DIRS" in st["configError"]
        with pytest.raises(Exception, match="BASE_DIRS"):
            await c.call_tool("list_files", {})
