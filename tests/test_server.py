import asyncio
import time

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from minirag_mcp import ocr
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import load_config
from minirag_mcp.lock import sync_lock
from minirag_mcp.server import create_app
from minirag_mcp.store import MAX_TOP_K, ChunkRecord, Store

# pytest-asyncio runs in auto mode (see pyproject) — bare `async def` tests are collected as-is.

SYNC_TIMEOUT_SECONDS = 30.0
SYNC_POLL_SECONDS = 0.02


async def await_sync(client: Client, job_id: str) -> dict:
    """Poll `sync_status` until the job reaches a terminal state, and return it.

    Bounded by wall-clock time rather than by an attempt count: sync runs on a
    background thread, so how many polls fit before it finishes is a property of
    the machine, not of the job. A fixed number of no-wait iterations passes on a
    fast laptop and races on a loaded CI runner — a bigger number would only move
    the threshold, not remove it. Yielding between polls also lets the event loop
    run, so this finishes as soon as the job does and costs one poll interval
    locally.
    """
    deadline = time.monotonic() + SYNC_TIMEOUT_SECONDS
    st: dict | None = None
    while True:
        st = (await client.call_tool("sync_status", {"jobId": job_id})).data
        if st["state"] in ("succeeded", "failed"):
            return st
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"sync job {job_id} did not finish within {SYNC_TIMEOUT_SECONDS:g}s; "
                f"last state={st['state']!r} counts={st['counts']} "
                f"errors={st['errors']} error={st['error']!r}"
            )
        await asyncio.sleep(SYNC_POLL_SECONDS)


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
        "sync_start",
        "sync_status",
        "ingest_file",
        "ingest_data",
        "ingest_url",
        "query_documents",
        "read_chunk_neighbors",
        "read_file",
        "list_files",
        "delete_file",
        "status",
    }


async def test_sync_then_query_then_neighbors(app):
    mcp, root = app
    async with Client(mcp) as c:
        job = (await c.call_tool("sync_start", {})).data
        st = await await_sync(c, job["jobId"])
        assert st["state"] == "succeeded", f"sync failed: {st['error']!r} {st['errors']}"
        assert st["counts"]["ingested"] == 2, st["counts"]

        res = (await c.call_tool("query_documents", {"query": "ERR_CONNECTION_REFUSED"})).data
        assert res["results"], "expected at least one search result"
        top = res["results"][0]
        assert {"text", "source", "title", "chunkIndex", "score"} <= set(top)
        assert res["sources"], "expected aggregated sources"
        assert {"source", "title", "hits", "displayPath"} <= set(res["sources"][0])
        assert res["sources"][0]["source"] == top["source"]  # rank order preserved
        # The citation policy tells the model to copy `displayPath`; this is the wire
        # end of that promise. A rename on either side leaves the policy pointing at a
        # field that isn't there, which no wording test would catch. And the file name
        # has to arrive intact — "without any modification" is the requirement.
        for s in res["sources"]:
            assert s["displayPath"], "a source with nothing to print in a Sources list"
            assert s["source"] == f"{root}/{s['displayPath']}"
            assert s["displayPath"].rsplit("/", 1)[-1] == s["source"].rsplit("/", 1)[-1]

        nb = (
            await c.call_tool(
                "read_chunk_neighbors",
                {"filePath": top["source"], "chunkIndex": top["chunkIndex"]},
            )
        ).data
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


async def test_sync_start_is_a_tool_error_while_another_process_syncs(app):
    """Contention reaches the MCP client as a ToolError, and no job is started.

    The lock is held from this process, which contends identically to a foreign
    one; tests/test_lock.py covers the genuinely cross-process side.
    """
    mcp, root = app
    async with Client(mcp) as c:
        with sync_lock(root / ".minirag" / "lancedb"):
            with pytest.raises(ToolError, match="syncing this index") as excinfo:
                await c.call_tool("sync_start", {})
            assert "DB_PATH" in str(excinfo.value)

        # The refusal left nothing behind: a sync works as soon as the rival is gone.
        job = (await c.call_tool("sync_start", {})).data
        assert (await await_sync(c, job["jobId"]))["state"] == "succeeded"


async def test_ingest_file_and_query_are_not_blocked_by_a_held_sync_lock(app):
    """The lock is scoped to sync: single-file ingest and search stay unblocked."""
    mcp, root = app
    async with Client(mcp) as c:
        with sync_lock(root / ".minirag" / "lancedb"):
            r = (await c.call_tool("ingest_file", {"filePath": str(root / "auth.md")})).data
            assert r["chunkCount"] >= 1
            res = (await c.call_tool("query_documents", {"query": "OAuth2 token"})).data
            assert res["results"], "expected search to work during a sync"


async def test_ingest_file_outside_root_is_tool_error(app):
    mcp, _ = app
    async with Client(mcp) as c:
        with pytest.raises(Exception, match="outside"):
            await c.call_tool("ingest_file", {"filePath": "/etc/passwd"})


async def test_ingest_data_and_url(app, monkeypatch, public_dns):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nRemote body.", "R"))
    mcp, _ = app
    async with Client(mcp) as c:
        r = (
            await c.call_tool(
                "ingest_data",
                {"data": "# Note\n\nSaved note body.", "source": "note-1", "format": "markdown"},
            )
        ).data
        assert r["source"] == "note-1"
        r2 = (await c.call_tool("ingest_url", {"url": "https://example.com/x"})).data
        assert r2["source"] == "https://example.com/x"
        with pytest.raises(Exception, match="http"):
            await c.call_tool("ingest_url", {"url": "file:///etc/passwd"})


async def test_list_files_reports_the_ocr_engine(app, monkeypatch):
    """The MCP half of the same user-visible fact the CLI prints as `[ocr:<engine>]`
    (tests/test_cli.py): how a document entered the index."""
    from minirag_mcp import ocr

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "recognized invoice text")
    mcp, root = app
    scan = root / "scan.png"
    scan.write_bytes(b"png bytes irrelevant, ocr is faked")
    async with Client(mcp) as c:
        await c.call_tool("ingest_file", {"filePath": str(scan)})
        await c.call_tool("ingest_file", {"filePath": str(root / "auth.md")})
        by_source = {f["source"]: f for f in (await c.call_tool("list_files", {})).data["files"]}
    assert by_source[str(scan)]["ocrEngine"] == "rapidocr"
    assert by_source[str(root / "auth.md")]["ocrEngine"] == ""


async def test_list_files_scope_stops_at_a_path_component_boundary(tmp_path, fake_embedder):
    """A scope of .../proj must not list .../project-secret, on disk or in the index."""
    root = tmp_path / "docs"
    (root / "proj").mkdir(parents=True)
    (root / "project-secret").mkdir()
    (root / "proj" / "a.md").write_text("# A\n\nProject body long enough to index.")
    (root / "project-secret" / "b.md").write_text("# B\n\nSecret body long enough to index.")
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
    mcp = create_app(cfg, embedder=fake_embedder)
    async with Client(mcp) as c:
        await c.call_tool("ingest_file", {"filePath": str(root / "project-secret" / "b.md")})
        listed = (await c.call_tool("list_files", {"scope": str(root / "proj")})).data
    assert [f["source"] for f in listed["files"]] == [str(root / "proj" / "a.md")]


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


async def test_read_file_distinguishes_failure_reasons(app):
    mcp, root = app
    missing = root / "nope.md"
    directory = root / "sub"
    unindexed = root / "auth.md"
    async with Client(mcp) as c:
        with pytest.raises(Exception, match="does not exist") as missing_exc:
            await c.call_tool("read_file", {"filePath": str(missing)})
        with pytest.raises(Exception, match="Not a file") as dir_exc:
            await c.call_tool("read_file", {"filePath": str(directory)})
        with pytest.raises(Exception, match="not found in index") as unindexed_exc:
            await c.call_tool("read_file", {"filePath": str(unindexed)})
    messages = {str(missing_exc.value), str(dir_exc.value), str(unindexed_exc.value)}
    assert len(messages) == 3, f"expected three distinct error messages, got {messages}"


async def test_all_tools_have_substantive_descriptions(app):
    mcp, _ = app
    async with Client(mcp) as c:
        tools = await c.list_tools()
    for t in tools:
        assert t.description and len(t.description) > 40, f"{t.name} description too thin"


async def test_query_top_k_is_clamped_and_must_be_positive(app, monkeypatch):
    """An oversized topK is capped rather than refused; the store never sees it raw."""
    seen: list[int] = []
    real = Store.search

    def spy(self, *args, **kwargs):
        seen.append(kwargs["top_k"])
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Store, "search", spy)
    mcp, root = app
    async with Client(mcp) as c:
        await c.call_tool("ingest_file", {"filePath": str(root / "auth.md")})
        res = (await c.call_tool("query_documents", {"query": "token", "topK": 10**8})).data
        assert res["results"], "a clamped query still returns results"
        with pytest.raises(Exception, match="topK must be at least 1"):
            await c.call_tool("query_documents", {"query": "token", "topK": 0})
        with pytest.raises(Exception, match="topK must be at least 1"):
            await c.call_tool("query_documents", {"query": "token", "topK": -5})
    assert seen == [MAX_TOP_K]  # clamped, and the two refused calls never reached the store


async def test_query_returns_each_parent_section_once(app):
    """`text` stays the passage that was ranked; the section around it is named by
    `parentId` and lives once in `parents`. Attaching it per hit resent the same
    section for every hit inside it — about a third of the returned text."""
    mcp, root = app
    (root / "spec.md").write_text(
        "# 1 Хранение\n\n"
        + " ".join(f"Правило {i} про хранение товарно-материальных запасов." for i in range(40)),
        encoding="utf-8",
    )
    async with Client(mcp) as c:
        await c.call_tool("ingest_file", {"filePath": str(root / "spec.md")})
        res = (await c.call_tool("query_documents", {"query": "хранение запасов"})).data

    hits = [r for r in res["results"] if r["source"].endswith("spec.md")]
    hit = hits[0]
    assert "parentText" not in hit, "the section is not repeated on the hit"
    assert hit["parentId"].startswith(str(root / "spec.md") + "#p")
    section = res["parents"][hit["parentId"]]
    assert hit["text"] in section
    assert len(section) > len(hit["text"]), "the section must be bigger than the hit"

    # several hits share one section, and it is serialized once for all of them
    shared = [r for r in hits if r["parentId"] == hit["parentId"]]
    assert len(shared) > 1, "need two hits in one section for this to prove anything"
    assert list(res["parents"]).count(hit["parentId"]) == 1
    assert len(res["parents"]) < len(res["results"])


async def test_read_file_reconstructs_the_document_rather_than_concatenating_chunks(app):
    """The tool is documented as returning the document. Joining raw chunk texts
    repeated the table's header row and the heading breadcrumb once per chunk —
    measured on the real corpus, +22% at the median and 2.64x at the tail — and planted
    a header row in the middle of the table's rows, which is not valid Markdown."""
    mcp, root = app
    header = "| № | Показатель | Периодичность |"
    delimiter = "| --- | --- | --- |"
    rows = [
        f"| {i} | Показатель номер {i} по форме статистической отчетности | месяц |"
        for i in range(30)
    ]
    doc = root / "table.md"
    doc.write_text(
        "# 1 Отчетность\n\n## 1.1 Формы\n\n" + "\n".join([header, delimiter, *rows]),
        encoding="utf-8",
    )
    async with Client(mcp) as c:
        await c.call_tool("ingest_file", {"filePath": str(doc)})
        full = (await c.call_tool("read_file", {"filePath": str(doc)})).data

    assert full["chunkCount"] > 3, "the table has to span several chunks to prove anything"
    text = full["text"]
    assert text.count(header) == 1, "the header row is the document's, not each chunk's"
    assert text.count(delimiter) == 1
    assert text.count("1 Отчетность > 1.1 Формы") == 1
    # every row survives, once, in order, and the table stays contiguous
    assert [ln for ln in text.split("\n") if ln.startswith("| ") and "---" not in ln] == [
        header,
        *rows,
    ]
    assert f"{delimiter}\n{rows[0]}" in text


async def test_status_names_the_ocr_engine_when_the_extra_is_installed(app, monkeypatch):
    mcp, _root = app
    monkeypatch.setattr(ocr, "available", lambda: True)
    async with Client(mcp) as c:
        st = (await c.call_tool("status", {})).data
    assert st["ocr"] == ocr.ENGINE_RAPIDOCR
    assert "ocrHint" not in st


async def test_status_says_ocr_is_unavailable_and_how_to_get_it(app, monkeypatch):
    """An agent that meets "this file needs OCR" has no other way to learn whether this
    install can ever read it, or what to tell the user to run."""
    mcp, _root = app
    monkeypatch.setattr(ocr, "available", lambda: False)
    async with Client(mcp) as c:
        st = (await c.call_tool("status", {})).data
    assert st["ocr"] == "unavailable"
    assert "minirag-mcp[ocr]" in st["ocrHint"]


async def test_status_reports_the_chunking_scheme(app):
    mcp, root = app
    async with Client(mcp) as c:
        await c.call_tool("ingest_file", {"filePath": str(root / "auth.md")})
        st = (await c.call_tool("status", {})).data
    assert st["chunkScheme"] == SCHEME_VERSION
    assert st["staleChunkCount"] == 0
    assert "schemeWarning" not in st


async def test_the_listing_and_status_agree_on_a_scheme_stale_source(tmp_path, fake_embedder):
    """`status` said the index was stale and to re-sync; `list_files` said the same
    file was "ingested". An agent decides what needs work from the listing, so the two
    have to tell it the same thing."""
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "auth.md"
    doc.write_text("# Auth Guide\n\nOAuth2 token authentication flow explained at length here.")
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)

    async with Client(create_app(cfg, embedder=fake_embedder)) as c:
        await c.call_tool("ingest_file", {"filePath": str(doc)})
        listed = (await c.call_tool("list_files", {})).data["files"]
    assert {f["source"]: f["state"] for f in listed}[str(doc)] == "ingested"

    # age the stored chunks by one scheme, exactly as an index built by an older
    # release would look, without touching the file on disk
    store = Store(cfg.db_path, dim=fake_embedder.dim)
    info = store.get_source(str(doc))
    store.replace_source(
        str(doc),
        [
            ChunkRecord(
                id=f"{r.source}#{r.chunk_index}",
                source=r.source,
                source_type="file",
                title=r.title,
                chunk_index=r.chunk_index,
                text=r.text,
                vector=[0.1] * fake_embedder.dim,
                file_hash=info.file_hash,
                mtime=info.mtime,
                ingested_at="2026-01-01T00:00:00+00:00",
                parent_id=r.parent_id,
                scheme_version=SCHEME_VERSION - 1,
            )
            for r in store.all_chunks(str(doc))
        ],
    )

    async with Client(create_app(cfg, embedder=fake_embedder)) as c:
        st = (await c.call_tool("status", {})).data
        listed = (await c.call_tool("list_files", {})).data["files"]

    assert st["staleChunkCount"] > 0 and "re-sync" in st["schemeWarning"].lower()
    assert {f["source"]: f["state"] for f in listed}[str(doc)] == "stale_scheme"


async def test_status_tells_the_user_to_re_sync_a_stale_index(tmp_path, fake_embedder):
    """The migration has to be *detected*. A stale index still answers queries, so
    nothing else would ever mention that its vectors describe truncated text."""
    root = tmp_path / "docs"
    root.mkdir()
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
    store = Store(cfg.db_path, dim=fake_embedder.dim)
    store.replace_source(
        "/old.md",
        [
            ChunkRecord(
                id="/old.md#0",
                source="/old.md",
                source_type="file",
                title="Old",
                chunk_index=0,
                text="кусок, нарезанный старой схемой",
                vector=[0.1] * fake_embedder.dim,
                file_hash="h",
                mtime=1.0,
                ingested_at="2026-01-01T00:00:00+00:00",
                parent_id="",
                scheme_version=SCHEME_VERSION - 1,
            )
        ],
    )
    async with Client(create_app(cfg, embedder=fake_embedder)) as c:
        st = (await c.call_tool("status", {})).data
    assert st["staleChunkCount"] == 1
    assert "re-sync" in st["schemeWarning"].lower()
