import multiprocessing
import os
import pickle
import re
from functools import partial
from types import SimpleNamespace

import pytest

from minirag_mcp import parallel
from minirag_mcp.config import load_config
from minirag_mcp.embedder import E5_LARGE, Embedder
from minirag_mcp.endpoints import EmbeddingEndpoint, Endpoint, EndpointError
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.store import DOCUMENT_TABLE, SIGNATURE_TABLE, ChunkRecord, IndexBusyError, Store
from parallel_helpers import SLOW_ROUTER, Preparer, Unloadable, broken_factory

GiB = 2**30

SAMPLE = "".join(
    f"processor\t: {n}\nphysical id\t: {phys}\ncore id\t\t: {core}\n\n"
    for n, (phys, core) in enumerate([(p, c) for p in (0, 1) for c in (0, 1)] * 2)
)


@pytest.mark.parametrize(
    "cores,memory,footprint,expected",
    [
        (18, 64 * GiB, int(1.5 * GiB), 17),  # endpoint: the cores decide
        (18, 64 * GiB, int(3.6 * GiB), 8),  # in-process e5: memory decides
        (8, 16 * GiB, int(3.6 * GiB), 2),
        (8, None, int(3.6 * GiB), 7),  # memory unknown
        (1, 64 * GiB, GiB, 1),
        (4, GiB, 4 * GiB, 1),
    ],
)
def test_default_workers(cores, memory, footprint, expected):
    assert parallel.default_workers(cores, memory, footprint) == expected


@pytest.mark.parametrize(
    "configured,files,default,expected",
    [
        (None, 3, 8, 1),
        (None, 100, 8, 8),
        (2, 12, 8, 2),
        (16, 20, 8, 5),
        (4, 7, 8, 1),
        (1, 1000, 8, 1),
    ],
)
def test_worker_count(configured, files, default, expected):
    assert parallel.worker_count(configured, files, default=default) == expected


def test_worker_footprint_counts_the_model_only_when_it_runs_in_process(fake_embedder):
    e5 = SimpleNamespace(model_name="intfloat/multilingual-e5-large", backend="in-process")
    assert parallel.worker_footprint(e5) == parallel.WORKER_BASE_BYTES + 2_240_000_000
    e5.backend = "endpoint"
    assert parallel.worker_footprint(e5) == parallel.WORKER_BASE_BYTES
    assert parallel.worker_footprint(fake_embedder) == parallel.WORKER_BASE_BYTES


def test_linux_physical_cores_counts_distinct_cores():
    assert parallel._linux_physical_cores(SAMPLE) == 4


def test_linux_physical_cores_is_none_without_topology():
    assert parallel._linux_physical_cores("processor\t: 0\nmodel name\t: x\n") is None


def test_physical_cores_is_within_the_logical_count():
    assert 1 <= parallel.physical_cores() <= os.cpu_count()


def test_worker_threads_share_the_cores(monkeypatch):
    monkeypatch.setattr(parallel, "physical_cores", lambda: 8)
    assert [parallel.worker_threads(w) for w in (1, 3, 16)] == [8, 2, 1]


def test_total_memory_is_positive_or_unknown():
    memory = parallel.total_memory()
    assert memory is None or memory > 0


@pytest.fixture
def paths(tmp_path):
    files = []
    for i in range(20):
        f = tmp_path / f"{i:02d}.md"
        f.write_text(f"text {i}")
        files.append(f)
    return files


def test_every_path_comes_back_in_input_order(paths):
    got = list(parallel.prepare_files(Preparer, paths, workers=2))
    assert [g.path for g in got] == paths
    assert [g.document for g in got] == [f"text {i}" for i in range(20)]
    assert all(g.error is None for g in got)


def test_a_file_that_raises_fails_alone(paths):
    got = list(parallel.prepare_files(partial(Preparer, fail_on="05.md"), paths, workers=2))
    assert (got[5].document, got[5].error) == (None, "cannot read 05.md")
    assert all(g.error is None for i, g in enumerate(got) if i != 5)


def test_a_worker_that_dies_fails_only_its_own_file(paths):
    got = list(parallel.prepare_files(partial(Preparer, crash_on="07.md"), paths, workers=2))
    assert [g.path for g in got] == paths and got[7].error == parallel.WORKER_DIED
    assert [g.document for i, g in enumerate(got) if i != 7] == [
        f"text {i}" for i in range(20) if i != 7
    ]


def test_a_preparer_that_cannot_be_built_fails_each_file_with_its_reason(paths):
    got = list(parallel.prepare_files(broken_factory, paths[:3], workers=2))
    assert [g.error for g in got] == ["no model here"] * 3


def test_a_preparer_that_cannot_be_unpickled_fails_each_file_with_its_reason(paths):
    got = list(parallel.prepare_files(Unloadable(), paths[:3], workers=2))
    assert [g.error for g in got] == ["no model here"] * 3


def test_workers_never_write_to_stdout(paths, capfd):
    list(parallel.prepare_files(partial(Preparer, shout=True), paths[:4], workers=2))
    out = capfd.readouterr()
    assert "LEAK" not in out.out and "LEAK-fd" in out.err


def test_closing_early_stops_every_worker(paths):
    got = parallel.prepare_files(Preparer, paths, workers=2)
    assert next(got).document == "text 0"
    got.close()
    assert multiprocessing.active_children() == []


def test_an_endpoint_failure_comes_back_marked(paths):
    make = partial(Preparer, fail_on="01.md", endpoint_down_on="02.md")
    got = list(parallel.prepare_files(make, paths[:4], workers=2))
    assert [g.endpoint_failed for g in got] == [False, False, True, False]
    assert (got[2].document, got[2].error) == (None, SLOW_ROUTER)


def test_worker_pipeline_prepares_like_the_parent(tmp_path, fake_embedder):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    f = tmp_path / "doc.md"
    f.write_text("# T\n\n" + "Sentence about topic. " * 80)
    pipe = parallel.worker_pipeline(cfg, fake_embedder, 3)
    assert pipe.store is None and pipe.labels is None and pipe.before_store is None
    assert fake_embedder.threads == 3
    assert pipe.prepare_file(f, rules_checked=True) == Pipeline(
        None, fake_embedder, cfg
    ).prepare_file(f)


def test_a_document_prepared_in_a_worker_is_stored_as_the_parent_prepares_it(
    tmp_path, fake_embedder
):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    f = tmp_path / "doc.md"
    f.write_text("# T\n\n" + "Sentence about topic. " * 80)
    make = partial(parallel.worker_pipeline, cfg, fake_embedder, 1)
    [got] = parallel.prepare_files(make, [f], workers=1)
    parent = Pipeline(Store(tmp_path / "db", dim=fake_embedder.dim), fake_embedder, cfg)
    assert got.error is None and got.document == parent.prepare_file(f)
    assert not hasattr(fake_embedder, "threads")  # the worker changed its own copy
    parent.store_prepared(got.document)
    rows = parent.store._table.search().where(f"source = '{f}'").limit(10_000).to_list()
    rows.sort(key=lambda r: r["chunk_index"])
    assert [r["text"] for r in rows] == [c.text for c in got.document.chunks]
    assert len(rows) == len(got.document.vectors) > 1
    for row, vector in zip(rows, got.document.vectors, strict=True):
        assert list(row["vector"]) == pytest.approx(vector)


@pytest.fixture
def dead_worker(tmp_path, dead_url, stub_text_embedding, tokenizer_file):
    """A worker's pipeline over a pickled copy of an embedder whose endpoint is down, and
    a file for it to prepare."""
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    f = tmp_path / "doc.md"
    f.write_text("# T\n\nSome text about a topic.")
    endpoint = EmbeddingEndpoint(Endpoint(dead_url + "/v1/embeddings", "e5"))
    parent = Embedder(E5_LARGE, tmp_path / "models", endpoint=endpoint)
    return parallel.worker_pipeline(cfg, pickle.loads(pickle.dumps(parent)), 1), f


def test_a_worker_embedder_never_falls_back(dead_worker, dead_url, stub_text_embedding):
    pipe, f = dead_worker
    with pytest.raises(EndpointError, match=re.escape(dead_url)):
        pipe.prepare_file(f, rules_checked=True)
    assert stub_text_embedding.model_names == []


def test_an_unreachable_endpoint_is_marked_as_an_endpoint_failure(dead_worker, dead_url):
    pipe, f = dead_worker
    got = parallel.prepare_one(pipe, f)
    assert (got.path, got.document, got.endpoint_failed) == (f, None, True)
    assert dead_url in got.error


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db", dim=8)


def recs(source, n, text="t", stamp="2026-08-07T00:00:00+00:00"):
    return [
        ChunkRecord(
            id=f"{source}#{i}",
            source=source,
            source_type="file",
            title="T",
            chunk_index=i,
            text=f"{text} {i}",
            vector=[0.1 * (i + 1)] * 8,
            file_hash="h",
            mtime=1.0,
            ingested_at=stamp,
        )
        for i in range(n)
    ]


def written(store):
    return (
        store._table.to_arrow().sort_by("id").to_pylist(),
        store._db.open_table(SIGNATURE_TABLE).to_arrow().sort_by("source").to_pylist(),
        store._db.open_table(DOCUMENT_TABLE).to_arrow().sort_by("source").to_pylist(),
    )


def _fail_batches(store, monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("batch refused")

    monkeypatch.setattr(store, "replace_sources", refuse)


def test_batch_writer_writes_when_the_batch_is_full_and_on_flush(store):
    w = parallel.BatchWriter(store, indexed=set(), batch_rows=5)
    assert w.add("/a.md", recs("/a.md", 3)) == [] and store.chunk_count() == 0
    assert w.add("/b.md", recs("/b.md", 3)) == [("/a.md", None), ("/b.md", None)]
    assert store.chunk_count() == 6
    assert w.add("/c.md", recs("/c.md", 1)) == []
    assert w.flush() == [("/c.md", None)] and store.chunk_count() == 7
    assert w.flush() == []


def test_batch_rows_defaults_to_the_constant_read_at_construction(store, monkeypatch):
    monkeypatch.setattr(parallel, "BATCH_ROWS", 2)
    w = parallel.BatchWriter(store, indexed=set())
    assert w.add("/a.md", recs("/a.md", 2)) == [("/a.md", None)]


def test_a_failed_batch_is_retried_source_by_source(store, monkeypatch):
    _fail_batches(store, monkeypatch)
    w = parallel.BatchWriter(store, indexed=set(), batch_rows=100)
    w.add("/a.md", recs("/a.md", 2))
    w.add("/b.md", recs("/b.md", 2))
    assert w.flush() == [("/a.md", None), ("/b.md", None)]
    assert sorted(s.source for s in store.list_sources()) == ["/a.md", "/b.md"]


def test_a_source_that_cannot_be_written_fails_alone(store, monkeypatch):
    _fail_batches(store, monkeypatch)
    real = store.replace_source

    def replace_source(source, records):
        if source == "/b.md":
            raise RuntimeError("disk full")
        real(source, records)

    monkeypatch.setattr(store, "replace_source", replace_source)
    w = parallel.BatchWriter(store, indexed=set(), batch_rows=100)
    w.add("/a.md", recs("/a.md", 1))
    w.add("/b.md", recs("/b.md", 1))
    assert w.flush() == [("/a.md", None), ("/b.md", "disk full")]
    assert [s.source for s in store.list_sources()] == ["/a.md"]


def test_a_batch_that_failed_half_way_is_completed_by_the_fallback(tmp_path, monkeypatch):
    """The merge of the indexed source commits, then the append of the new one raises:
    /a.md holds new chunks under its old document row and signature."""
    old = recs("/a.md", 3, "old")
    new_a = recs("/a.md", 2, "new", stamp="2026-09-01T00:00:00+00:00")
    new_b = recs("/b.md", 2, "fresh", stamp="2026-09-01T00:00:00+00:00")
    store, reference = Store(tmp_path / "db", dim=8), Store(tmp_path / "ref", dim=8)
    for s in (store, reference):
        s.replace_source("/a.md", old)
    reference.replace_source("/a.md", new_a)
    reference.replace_source("/b.md", new_b)

    real_add, original = store._table.add, store.replace_sources

    def refuse(*args, **kwargs):
        raise RuntimeError("append refused")

    def half_way(batch, *, indexed):
        monkeypatch.setattr(store._table, "add", refuse)
        try:
            original(batch, indexed=indexed)
        finally:
            monkeypatch.setattr(store._table, "add", real_add)

    monkeypatch.setattr(store, "replace_sources", half_way)
    with pytest.raises(RuntimeError):
        half_way([("/a.md", new_a), ("/b.md", new_b)], indexed={"/a.md"})
    assert written(store) != written(reference)  # the failure left /a.md half written

    w = parallel.BatchWriter(store, indexed={"/a.md"}, batch_rows=100)
    w.add("/a.md", new_a)
    w.add("/b.md", new_b)
    assert w.flush() == [("/a.md", None), ("/b.md", None)]
    assert written(store) == written(reference)
    assert not store.get_source("/a.md").duplicated


def test_a_rebuild_in_progress_is_not_turned_into_file_failures(store, monkeypatch):
    def busy(*args, **kwargs):
        raise IndexBusyError("rebuilding")

    monkeypatch.setattr(store, "replace_sources", busy)
    w = parallel.BatchWriter(store, indexed=set(), batch_rows=100)
    w.add("/a.md", recs("/a.md", 1))
    with pytest.raises(IndexBusyError):
        w.flush()
