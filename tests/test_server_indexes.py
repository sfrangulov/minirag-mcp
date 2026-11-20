"""The MCP tools over several indexes: the `index` argument, `list_indexes`, and results
that name the index that answered."""

from __future__ import annotations

import asyncio
import pickle
import threading
import time

import pytest
from fastmcp import Client

from conftest import SYNC_POLL_SECONDS, SYNC_TIMEOUT_SECONDS, await_sync
from index_helpers import file_registry
from minirag_mcp import ocr, parallel
from minirag_mcp.embedder import E5_LARGE, MINILM
from minirag_mcp.identity import IndexState
from minirag_mcp.registry import Connection
from minirag_mcp.server import create_app


@pytest.fixture
def make(indexes_file, fake_embedder, index_states):
    def build(default="alpha", allow=None, factory=None):
        reg = file_registry(indexes_file, factory or (lambda c, s: fake_embedder))
        return create_app(reg, connection=Connection(default, allow))

    return build


def _text(result) -> str:
    assert result.is_error, result.data
    return result.content[0].text


async def test_every_tool_takes_an_optional_index(make):
    async with Client(make()) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert len(tools) == 13
    for name, t in tools.items():
        if name == "list_indexes":
            continue
        props = t.inputSchema["properties"]
        assert "index" in props and "index" not in t.inputSchema.get("required", []), name
        assert "list_indexes" in props["index"]["description"], name


async def test_a_call_without_index_uses_the_connection_default(make, indexes_file):
    alpha = indexes_file.roots["alpha"]
    async with Client(make()) as c:
        r = (await c.call_tool("ingest_file", {"filePath": str(alpha / "alpha.md")})).data
        found = (await c.call_tool("query_documents", {"query": "apricots"})).data
    assert r["index"] == "alpha" and found["index"] == "alpha"
    assert found["sources"]
    assert all(s["source"].startswith(f"{alpha}/") for s in found["sources"])


async def test_a_named_index_takes_reads_and_writes(make, indexes_file):
    alpha_md = indexes_file.roots["alpha"] / "alpha.md"
    beta_md = indexes_file.roots["beta"] / "beta.md"
    async with Client(make()) as c:
        await c.call_tool("ingest_file", {"filePath": str(alpha_md)})
        r = (await c.call_tool("ingest_file", {"filePath": str(beta_md), "index": "beta"})).data
        found = (await c.call_tool("query_documents", {"query": "bananas", "index": "beta"})).data
        default = (await c.call_tool("query_documents", {"query": "bananas"})).data
    assert r["index"] == "beta" and found["index"] == "beta"
    assert [s["source"] for s in found["sources"]] == [str(beta_md)]
    assert default["index"] == "alpha"
    assert [s["source"] for s in default["sources"]] == [str(alpha_md)]


# How each tool that takes a path is handed one.
PATH_ARGS = {
    "sync_start": lambda path: {"path": path},
    "ingest_file": lambda path: {"filePath": path},
    "read_chunk_neighbors": lambda path: {"filePath": path, "chunkIndex": 0},
    "read_file": lambda path: {"filePath": path},
    "delete_file": lambda path: {"filePath": path},
}


@pytest.mark.parametrize("tool", sorted(PATH_ARGS))
async def test_a_file_under_another_reachable_index_is_refused_naming_it(make, indexes_file, tool):
    beta_md = indexes_file.roots["beta"] / "beta.md"
    async with Client(make()) as c:
        r = await c.call_tool(tool, PATH_ARGS[tool](str(beta_md)), raise_on_error=False)
    assert _text(r) == (
        f"Index 'alpha': {beta_md} is outside its roots ({indexes_file.roots['alpha']}) but "
        "under those of index 'beta': pass index=\"beta\" to reach it. Other reachable "
        "indexes: beta, gamma."
    )


async def test_a_file_under_no_reachable_index_is_refused_naming_where_roots_are_set(
    make, indexes_file
):
    alpha = indexes_file.roots["alpha"]
    beta_md = indexes_file.roots["beta"] / "beta.md"
    async with Client(make(default="alpha", allow=frozenset())) as c:
        hidden = await c.call_tool("read_file", {"filePath": str(beta_md)}, raise_on_error=False)
    async with Client(make()) as c:
        nowhere = await c.call_tool(
            "ingest_file", {"filePath": str(indexes_file.path)}, raise_on_error=False
        )
    remedy = (
        "and no other index this connection may use holds it; a folder is reached by adding "
        f"it to an index's roots in {indexes_file.path}."
    )
    assert _text(hidden) == f"Index 'alpha': {beta_md} is outside its roots ({alpha}), {remedy}"
    assert _text(nowhere) == (
        f"Index 'alpha': {indexes_file.path} is outside its roots ({alpha}), {remedy} Other "
        "reachable indexes: beta, gamma."
    )


def _every_tool(path: str) -> dict[str, dict]:
    """Arguments each tool but list_indexes is called with: valid, but for the index."""
    return {
        "sync_start": {},
        "sync_status": {"jobId": "0"},
        "ingest_file": {"filePath": path},
        "ingest_data": {"data": "x", "source": "s"},
        "ingest_url": {"url": "https://example.com/x"},
        "query_documents": {"query": "x"},
        "find_mentions": {"terms": "x"},
        "read_chunk_neighbors": {"filePath": path, "chunkIndex": 0},
        "read_file": {"filePath": path},
        "list_files": {},
        "delete_file": {"filePath": path},
        "status": {},
    }


async def test_the_allow_list_test_calls_every_tool(make):
    async with Client(make()) as c:
        served = {t.name for t in await c.list_tools()}
    assert set(_every_tool("")) == served - {"list_indexes"}


@pytest.mark.parametrize("tool", sorted(_every_tool("")))
async def test_every_tool_refuses_an_index_outside_the_allow_list_and_never_opens_it(
    make, indexes_file, fake_embedder, tool
):
    made = []

    def factory(config, scheme):
        made.append(config.index_name)
        return fake_embedder

    gamma_md = indexes_file.roots["gamma"] / "gamma.md"
    args = {**_every_tool(str(gamma_md))[tool], "index": "gamma"}
    async with Client(make(allow=frozenset({"beta"}), factory=factory)) as c:
        r = await c.call_tool(tool, args, raise_on_error=False)
    assert _text(r) == (
        "Index 'gamma' is not allowed on this connection (allow=beta). Reachable indexes: "
        "alpha, beta."
    )
    assert "gamma" not in made and not (indexes_file.roots["gamma"] / ".minirag").exists()


async def test_a_write_during_a_rebuild_is_refused_naming_the_index_and_the_others(
    make, indexes_file
):
    from minirag_mcp.identity import begin_staging
    from minirag_mcp.lock import sync_lock

    beta_md = str(indexes_file.roots["beta"] / "beta.md")
    db = indexes_file.roots["beta"] / ".minirag" / "v2"
    async with Client(make()) as c:
        await c.call_tool("ingest_file", {"filePath": beta_md, "index": "beta"})
        with sync_lock(db):
            begin_staging(db)
            refused = [
                await c.call_tool(tool, {**args, "index": "beta"}, raise_on_error=False)
                for tool, args in (
                    ("ingest_file", {"filePath": beta_md}),
                    ("ingest_data", {"data": "x", "source": "s"}),
                    ("delete_file", {"filePath": beta_md}),
                )
            ]
    for r in refused:
        text = _text(r)
        assert text.startswith(f"Index 'beta': {db} is being rebuilt into "), text
        assert text.endswith(" sync finished. Other reachable indexes: alpha, gamma."), text


async def test_refusals_reach_the_client_naming_the_reachable_indexes(make):
    async with Client(make(default="alpha", allow=frozenset({"beta"}))) as c:
        r = await c.call_tool(
            "query_documents", {"query": "x", "index": "gamma"}, raise_on_error=False
        )
    assert _text(r).endswith(
        "Index 'gamma' is not allowed on this connection (allow=beta). Reachable indexes: "
        "alpha, beta."
    )


async def test_a_connection_without_a_default_must_name_an_index(make):
    refusal = (
        "This connection has no default index, so name one with `index`. Reachable indexes: "
        "alpha, beta, gamma."
    )
    async with Client(make(default=None)) as c:
        query = await c.call_tool("query_documents", {"query": "x"}, raise_on_error=False)
        status = await c.call_tool("status", {}, raise_on_error=False)
        named = (await c.call_tool("status", {"index": "alpha"})).data
    assert _text(query) == refusal and _text(status) == refusal
    assert named["index"] == "alpha" and named["indexState"] == "ready"


async def test_list_indexes_reports_each_index_and_what_this_connection_may_use(make, indexes_file):
    alpha_md = indexes_file.roots["alpha"] / "alpha.md"
    async with Client(make(default="alpha", allow=frozenset())) as c:
        await c.call_tool("ingest_file", {"filePath": str(alpha_md)})
        got = (await c.call_tool("list_indexes", {})).data
    assert got["default"] == "alpha" and "configError" not in got
    alpha, beta, gamma = got["indexes"]
    assert alpha["reachable"] and alpha["state"] == "ready" and "stateReason" not in alpha
    assert (alpha["name"], alpha["description"], alpha["model"], alpha["language"]) == (
        "alpha",
        "Alpha documents",
        "fake",
        "English",
    )
    assert (alpha["documentCount"], alpha["chunkCount"]) == (1, 1)
    assert (alpha["embeddingBackend"], alpha["reranker"], alpha["classifyBackend"]) == (
        "in-process",
        "off",
        "in-process",
    )
    assert beta == {
        "name": "beta",
        "description": "Beta documents",
        "model": E5_LARGE,
        "language": "Russian",
        "reachable": False,
    }
    assert gamma == {
        "name": "gamma",
        "description": "Gamma documents",
        "model": MINILM,
        "language": "English",
        "reachable": False,
    }
    assert not (indexes_file.roots["beta"] / ".minirag").exists()


async def test_a_listed_index_whose_db_is_a_file_names_its_configured_model(make, indexes_file):
    blocker = indexes_file.path.parent.parent / "blocker"
    blocker.write_text("a file where gamma's index folder should be")
    indexes_file.write(indexes_file.text + 'db = "~/blocker"\n')
    async with Client(make(default="alpha", allow=frozenset())) as c:
        hidden = (await c.call_tool("list_indexes", {})).data["indexes"][2]
    async with Client(make()) as c:
        shown = (await c.call_tool("list_indexes", {})).data["indexes"][2]
    assert hidden == {
        "name": "gamma",
        "description": "Gamma documents",
        "model": MINILM,
        "language": "English",
        "reachable": False,
    }
    assert (shown["model"], shown["state"]) == (MINILM, "unavailable")
    assert str(blocker) in shown["stateReason"] and "documentCount" not in shown


async def test_a_rebuild_pending_index_answers_with_a_warning_and_refuses_ingests(
    make, index_states
):
    reason = "chunk budget 250 -> 340 tokens"
    index_states["beta"] = (IndexState.REBUILD_PENDING, reason)
    async with Client(make()) as c:
        found = (await c.call_tool("query_documents", {"query": "bananas", "index": "beta"})).data
        refused = await c.call_tool(
            "ingest_data", {"data": "x", "source": "s", "index": "beta"}, raise_on_error=False
        )
        listed = (await c.call_tool("list_indexes", {})).data
        ready = (await c.call_tool("query_documents", {"query": "apricots"})).data
    assert (found["index"], found["indexState"]) == ("beta", "rebuild pending")
    assert found["indexWarning"] == (
        f"Index 'beta' differs from its configuration ({reason}); it answers from its "
        "existing vectors until an unscoped sync_start rebuilds it."
    )
    text = _text(refused)
    assert text.startswith("Index 'beta': ") and "needs a rebuild" in text and reason in text
    assert text.endswith("Other reachable indexes: alpha, gamma.")
    beta = listed["indexes"][1]
    assert (beta["state"], beta["stateReason"]) == ("rebuild pending", reason)
    assert ready["index"] == "alpha" and "indexState" not in ready and "indexWarning" not in ready


async def test_an_unavailable_index_is_refused_listed_and_diagnosable(make, fake_embedder):
    def factory(config, scheme):
        if config.index_name == "gamma":
            raise RuntimeError("probe mismatch")
        return fake_embedder

    async with Client(make(factory=factory)) as c:
        query = await c.call_tool(
            "query_documents", {"query": "x", "index": "gamma"}, raise_on_error=False
        )
        listed = (await c.call_tool("list_indexes", {})).data
        st = (await c.call_tool("status", {"index": "gamma"})).data
    assert "Index 'gamma' is unavailable: probe mismatch." in _text(query)
    assert listed["indexes"][2] == {
        "name": "gamma",
        "description": "Gamma documents",
        "model": MINILM,
        "language": "English",
        "reachable": True,
        "state": "unavailable",
        "stateReason": "probe mismatch",
    }
    assert (st["index"], st["indexState"], st["indexError"]) == (
        "gamma",
        "unavailable",
        "probe mismatch",
    )


async def test_sync_status_is_polled_in_the_index_the_job_runs_in(make):
    async with Client(make()) as c:
        job = (await c.call_tool("sync_start", {"index": "beta"})).data
        st = await await_sync(c, job["jobId"], index="beta")
        elsewhere = await c.call_tool("sync_status", {"jobId": job["jobId"]}, raise_on_error=False)
    assert job["index"] == "beta"
    assert (st["index"], st["state"], st["counts"]["ingested"]) == ("beta", "succeeded", 1)
    assert _text(elsewhere) == (
        f"Unknown sync job {job['jobId']!r} (only the latest job is retained) in index "
        "'alpha'; a job started in another index is polled with that `index`."
    )


async def test_a_scoped_sync_of_a_pending_index_is_refused_naming_the_others(
    make, indexes_file, index_states
):
    index_states["beta"] = (IndexState.REBUILD_PENDING, "chunk budget 250 -> 340 tokens")
    beta_md = indexes_file.roots["beta"] / "beta.md"
    async with Client(make()) as c:
        r = await c.call_tool(
            "sync_start", {"path": str(beta_md), "index": "beta"}, raise_on_error=False
        )
    text = _text(r)
    assert text.startswith("Index 'beta': ") and "needs a rebuild" in text
    assert text.endswith("Other reachable indexes: alpha, gamma.")


async def test_a_busy_sync_is_refused_naming_the_index_its_db_key_and_the_others(
    make, indexes_file
):
    from minirag_mcp.lock import sync_lock

    async with Client(make()) as c:
        with sync_lock(indexes_file.roots["beta"] / ".minirag" / "v2"):
            r = await c.call_tool("sync_start", {"index": "beta"}, raise_on_error=False)
    text = _text(r)
    assert text.startswith("Index 'beta': Another sync of this index")
    assert f"use a different db in {indexes_file.path} [index.beta]." in text
    assert "DB_PATH" not in text and text.endswith("Other reachable indexes: alpha, gamma.")


async def _poll(client: Client, job_id: str, until) -> dict:
    deadline = time.monotonic() + SYNC_TIMEOUT_SECONDS
    while True:
        st = (await client.call_tool("sync_status", {"jobId": job_id})).data
        if until(st) or time.monotonic() >= deadline:
            return st
        await asyncio.sleep(SYNC_POLL_SECONDS)


async def test_a_sync_waiting_for_another_indexs_workers_says_so(make, indexes_file, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    go = threading.Event()

    def in_this_process(factory, waiting, workers, in_flight):
        go.wait(SYNC_TIMEOUT_SECONDS)
        preparer = pickle.loads(factory)()
        while waiting:
            yield parallel.prepare_one(preparer, waiting.popleft())
        return []

    monkeypatch.setattr(parallel, "_run", in_this_process)
    alpha = indexes_file.roots["alpha"]
    for i in range(7):
        (alpha / f"doc{i}.md").write_text(f"# Doc {i}\n\nBody of document {i}.")
    indexes_file.write(
        indexes_file.text.replace(
            'roots = ["~/corpora/alpha"]\n', 'roots = ["~/corpora/alpha"]\nsync_workers = 2\n'
        )
    )
    async with Client(make()) as c:
        try:
            # What another index's sync holds while its workers run.
            parallel._POOL_GATE.acquire()
            try:
                job = (await c.call_tool("sync_start", {})).data["jobId"]
                waiting = await _poll(c, job, lambda st: st["waiting"])
            finally:
                parallel._POOL_GATE.release()
            started = await _poll(c, job, lambda st: not st["waiting"])
        finally:
            # Else the job's thread holds the gate, and every later pool waits for it.
            go.set()
        done = await await_sync(c, job)
    assert (waiting["state"], waiting["waiting"]) == ("running", parallel.POOL_WAIT)
    assert (started["state"], started["waiting"]) == ("running", None)
    assert (done["state"], done["waiting"], done["counts"]["ingested"]) == ("succeeded", None, 8)


async def test_the_sync_of_an_index_a_reload_replaced_is_still_reported(
    make, indexes_file, monkeypatch
):
    from minirag_mcp.ingest.pipeline import Pipeline

    entered, release = threading.Event(), threading.Event()
    real = Pipeline.ingest_file

    def held(self, path, **kw):
        entered.set()
        release.wait(SYNC_TIMEOUT_SECONDS)
        return real(self, path, **kw)

    monkeypatch.setattr(Pipeline, "ingest_file", held)
    async with Client(make()) as c:
        job = (await c.call_tool("sync_start", {})).data["jobId"]
        try:
            assert await asyncio.to_thread(entered.wait, SYNC_TIMEOUT_SECONDS)
            indexes_file.write(
                indexes_file.text.replace(
                    'roots = ["~/corpora/alpha"]\n',
                    'roots = ["~/corpora/alpha"]\nhybrid_weight = 0.5\n',
                )
            )
            st = (await c.call_tool("status", {})).data
            row = (await c.call_tool("list_indexes", {})).data["indexes"][0]
        finally:
            release.set()
        done = await await_sync(c, job)
    assert (st["hybridWeight"], st["indexState"]) == (0.5, "syncing")
    assert (row["state"], row["stateReason"]) == ("syncing", "a sync of this index is running")
    assert (done["index"], done["state"], done["counts"]["ingested"]) == ("alpha", "succeeded", 1)


async def test_an_edited_indexes_toml_reaches_the_client_without_a_restart(make, indexes_file):
    delta = indexes_file.path.parent.parent / "corpora" / "delta"
    delta.mkdir()
    (delta / "delta.md").write_text("# Delta\n\nDelta body about dates and dunes.")
    table = '\n[index.delta]\ndescription = "{}"\nroots = ["~/corpora/delta"]\n'
    async with Client(make()) as c:
        before = (await c.call_tool("list_indexes", {})).data
        indexes_file.write(indexes_file.text + table.format("Delta documents"))
        r = (
            await c.call_tool(
                "ingest_file", {"filePath": str(delta / "delta.md"), "index": "delta"}
            )
        ).data
        indexes_file.write(indexes_file.text + table.format("Dates and dunes"))
        after = (await c.call_tool("list_indexes", {})).data
    assert [row["name"] for row in before["indexes"]] == ["alpha", "beta", "gamma"]
    assert r["index"] == "delta" and r["chunkCount"] >= 1
    assert [row["name"] for row in after["indexes"]] == ["alpha", "beta", "gamma", "delta"]
    assert after["indexes"][3]["description"] == "Dates and dunes"


async def test_a_broken_edit_is_reported_and_the_old_indexes_keep_answering(make, indexes_file):
    alpha_md = indexes_file.roots["alpha"] / "alpha.md"
    async with Client(make()) as c:
        await c.call_tool("ingest_file", {"filePath": str(alpha_md)})
        indexes_file.write("[server]\nport 8765\n")
        st = (await c.call_tool("status", {})).data
        found = (await c.call_tool("query_documents", {"query": "apricots"})).data
        listed = (await c.call_tool("list_indexes", {})).data
    assert "line 2" in st["configError"] and st["index"] == "alpha"
    assert found["index"] == "alpha" and [s["source"] for s in found["sources"]] == [str(alpha_md)]
    assert "line 2" in listed["configError"]
    assert [row["name"] for row in listed["indexes"]] == ["alpha", "beta", "gamma"]
