"""A sync that reads its files in worker processes and writes their rows in batches."""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import sysconfig
from collections import Counter
from pathlib import Path

import pytest

from conftest import FakeEmbedder
from minirag_mcp import ocr, parallel
from minirag_mcp.config import load_config
from minirag_mcp.endpoints import EndpointError
from minirag_mcp.handle import IndexHandle
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.lock import SyncLock, SyncLockBusy, sync_lock
from minirag_mcp.results import run_query
from minirag_mcp.store import DOCUMENT_TABLE, SIGNATURE_TABLE, IndexBusyError
from parallel_helpers import CRASH_MARK, CrashingEmbedder

TAXONOMY = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.initiative]
default = "general"
values.general = { description = "Work outside any initiative." }
values.alpha = { description = "The Alpha initiative.", paths = ["alpha"] }

[facets.type]
values.minutes = { description = "Minutes. Date, participants, decisions." }
values.report = { description = "Report. Findings and figures." }
"""


# A zip's signature over nothing. Plain bytes would not do: markitdown reads a .docx that
# is not a zip at all as text.
BROKEN_DOCX = b"PK\x03\x04" + bytes(40)


def write_table(path: Path, rows: int) -> Path:
    path.write_text("".join(f"{i},value {i}\n" for i in range(rows)), encoding="utf-8")
    return path


def write_doc(path: Path, parts: int) -> Path:
    """One chunk per part at the default budget."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n\n".join(
        f"## Part {i}\n\n" + f"Body paragraph {i} of {path.stem}. " * 8 for i in range(parts)
    )
    path.write_text(f"# {path.stem}\n\n{body}", encoding="utf-8")
    return path


def all_rows(store) -> list[dict]:
    return store._table.to_arrow().to_pylist()


def signatures(store) -> list[tuple]:
    rows = store._db.open_table(SIGNATURE_TABLE).to_arrow().sort_by("source").to_pylist()
    return [(r["source"], r["signature"], r["shingles"]) for r in rows]


def documents(store) -> list[dict]:
    rows = store._db.open_table(DOCUMENT_TABLE).to_arrow().sort_by("source").to_pylist()
    return [{k: v for k, v in r.items() if k != "ingested_at"} for r in rows]


@pytest.fixture(autouse=True)
def without_ocr(monkeypatch):
    """With the [ocr] extra installed, the parent fetches Tesseract's data before a parallel
    sync. The tests of that fetch make OCR available again."""
    monkeypatch.setattr(ocr, "available", lambda: False)


def test_parallel_and_serial_sync_write_identical_indexes(tmp_path, model_factory, monkeypatch):
    monkeypatch.setattr(parallel, "BATCH_ROWS", 7)
    root = tmp_path / "docs"
    (root / ".minirag").mkdir(parents=True)
    (root / ".minirag" / "classify.toml").write_text(TAXONOMY, encoding="utf-8")
    model_factory.default = {"report": 1.0}
    shrink = write_doc(root / "alpha" / "shrink.md", 6)
    write_doc(root / "keep.md", 1)
    write_table(root / "keep.csv", 5)
    grow = write_table(root / "grow.csv", 5)

    def handle(name: str, workers: int):
        env = {
            "BASE_DIR": str(root),
            "DB_PATH": str(tmp_path / name),
            "CACHE_DIR": str(tmp_path / "cache"),
            "RAG_SYNC_WORKERS": str(workers),
            "RAG_MAX_TABLE_ROWS": "5",
        }
        cfg = load_config(env, cwd=root)
        h = IndexHandle(cfg, embedder_factory=lambda s: FakeEmbedder(), model_factory=model_factory)
        return cfg, h

    def run_sync(h):
        counts, errors, rebuilt = h.run_sync(None)
        assert rebuilt is None
        return counts, errors

    cfg_a, handle_a = handle("a", 1)
    cfg_b, handle_b = handle("b", 2)
    run_sync(handle_a)
    run_sync(handle_b)

    write_doc(shrink, 2)
    new = [
        write_doc(root / ("alpha" if i % 2 else "") / f"new{i}.md", 2 + i % 3) for i in range(10)
    ]
    (root / "broken.docx").write_bytes(BROKEN_DOCX)
    write_table(grow, 6)
    write_table(root / "alpha" / "big.csv", 40)
    round2_sources = {str(shrink), *(str(p) for p in new)}
    counts_a, errors_a = run_sync(handle_a)

    spy_workers: list[int] = []
    real = parallel.prepare_files

    def spy(make_preparer, paths, *, workers):
        spy_workers.append(workers)
        return real(make_preparer, paths, workers=workers)

    monkeypatch.setattr(parallel, "prepare_files", spy)
    counts_b, errors_b = run_sync(handle_b)

    store_a, store_b = handle_a.get().store, handle_b.get().store
    emb = FakeEmbedder()
    assert spy_workers == [2]
    assert counts_a == counts_b and counts_b["failed"] == 1 and counts_b["signed"] == 0
    assert (counts_b["largeTables"], counts_b["deleted"]) == (2, 1)
    assert store_b.get_source(str(grow)) is None and store_b.get_source(str(root / "keep.csv"))
    assert sorted(errors_a, key=str) == sorted(errors_b, key=str)
    rows_a, rows_b = all_rows(store_a), all_rows(store_b)
    assert Counter(r["id"] for r in rows_b) == Counter(r["id"] for r in rows_a)
    by_id = {r["id"]: r for r in rows_a}
    for r in rows_b:
        assert {k: v for k, v in r.items() if k != "ingested_at"} == {
            k: v for k, v in by_id[r["id"]].items() if k != "ingested_at"
        }
    assert not any(r["id"].startswith(str(shrink) + "#") and r["chunk_index"] >= 2 for r in rows_b)
    assert any(r["labels_json"] for r in rows_b)
    written = sum(1 for r in rows_b if r["source"] in round2_sources)
    assert written > 14 and written % 7
    assert signatures(store_a) == signatures(store_b) and store_b.unsigned_sources() == []
    assert documents(store_a) == documents(store_b)
    for q in ("alpha body", "paragraph 3", "unrelated words"):
        found_a = run_query(store_a, emb, cfg_a, q, top_k=20)["sources"]
        assert run_query(store_b, emb, cfg_b, q, top_k=20)["sources"] == found_a
    again, _ = run_sync(handle_b)
    assert (again["ingested"], again["deleted"], again["moved"]) == (0, 0, 0)


def corpus(tmp_path: Path, embedder, files: int, *, workers: int = 2):
    """`files` small documents a.md, b.md, ... and a handle syncing them with `workers`."""
    root = tmp_path / "docs"
    root.mkdir()
    paths = []
    for i in range(files):
        path = root / f"{chr(ord('a') + i)}.md"
        path.write_text(f"# {path.stem}\n\nBody of document {path.stem}.", encoding="utf-8")
        paths.append(path)
    env = {
        "BASE_DIR": str(root),
        "DB_PATH": str(tmp_path / "db"),
        "CACHE_DIR": str(tmp_path / "cache"),
        "RAG_SYNC_WORKERS": str(workers),
    }
    cfg = load_config(env, cwd=root)
    return paths, IndexHandle(cfg, embedder_factory=lambda s: embedder)


def sync(h, on_event=None) -> tuple[dict[str, int], list[dict]]:
    counts, errors, rebuilt = h.run_sync(None, on_event)
    assert rebuilt is None
    return counts, errors


def test_a_worker_that_dies_mid_sync_fails_only_its_own_file(tmp_path):
    paths, h = corpus(tmp_path, CrashingEmbedder(), 8)
    paths[3].write_text(f"# d\n\nBody of document d. {CRASH_MARK}", encoding="utf-8")
    db = h.get().store.db_path
    counts, errors = sync(h)
    assert (counts["ingested"], counts["failed"]) == (7, 1)
    assert errors == [{"source": str(paths[3]), "error": parallel.WORKER_DIED}]
    assert multiprocessing.active_children() == []
    with sync_lock(db):
        pass


class Interrupted(BaseException):
    pass


def test_a_sync_interrupted_while_workers_run_leaves_none_running(
    tmp_path, fake_embedder, monkeypatch
):
    paths, h = corpus(tmp_path, fake_embedder, 8)
    pipeline = h.get().pipeline

    def records(doc, **kwargs):
        raise Interrupted

    monkeypatch.setattr(pipeline, "records", records)
    # Kept, as a caller that logs the exception keeps it: its traceback holds the sync's
    # frames, so nothing they left open is closed when they unwind.
    with pytest.raises(Interrupted) as raised:
        sync(h)
    assert raised.tb is not None and multiprocessing.active_children() == []


def test_a_small_sync_stays_in_this_process(tmp_path, fake_embedder, monkeypatch):
    _, h = corpus(tmp_path, fake_embedder, 3, workers=4)

    def refuse(*args, **kwargs):
        raise AssertionError("a sync of 3 files started workers")

    monkeypatch.setattr(parallel, "prepare_files", refuse)
    counts, errors = sync(h)
    assert counts["ingested"] == 3 and errors == []


def test_failures_in_a_parallel_sync_are_counted_like_serial_ones(
    tmp_path, fake_embedder, inline_workers, monkeypatch
):
    paths, h = corpus(tmp_path, fake_embedder, 8)
    a, c = paths[0], paths[2]
    broken = paths[0].parent / "broken.docx"
    broken.write_bytes(BROKEN_DOCX)
    store = h.get().store

    def refuse(*args, **kwargs):
        raise RuntimeError("batch refused")

    real = store.replace_source

    def replace_source(source, records):
        if source == str(c):
            raise RuntimeError("disk full")
        real(source, records)

    monkeypatch.setattr(store, "replace_sources", refuse)
    monkeypatch.setattr(store, "replace_source", replace_source)
    events: list[str] = []
    counts, errors = sync(h, events.append)

    assert inline_workers == [2]
    assert (counts["ingested"], counts["failed"]) == (7, 2)
    assert {e["source"] for e in errors} == {str(broken), str(c)}
    assert f"failed {c}: disk full" in events and f"ingested {a}" in events
    assert any(e.startswith(f"failed {broken}: ") for e in events)
    assert sorted(s.source for s in store.list_sources()) == sorted(str(p) for p in paths if p != c)


def test_a_file_that_cannot_be_labelled_fails_alone(
    tmp_path, fake_embedder, inline_workers, monkeypatch
):
    paths, h = corpus(tmp_path, fake_embedder, 8)
    pipeline = h.get().pipeline
    real = pipeline.records

    def records(doc, **kwargs):
        if doc.source == str(paths[1]):
            raise RuntimeError("no rows for b")
        return real(doc, **kwargs)

    monkeypatch.setattr(pipeline, "records", records)
    counts, errors = sync(h)
    assert inline_workers == [2]
    assert (counts["ingested"], counts["failed"]) == (7, 1)
    assert errors == [{"source": str(paths[1]), "error": "no rows for b"}]


def test_a_refused_write_fails_its_files_and_the_sync_finishes(
    tmp_path, fake_embedder, inline_workers, monkeypatch
):
    paths, h = corpus(tmp_path, fake_embedder, 8)
    store = h.get().store

    def busy(*args, **kwargs):
        raise IndexBusyError("the index was replaced")

    monkeypatch.setattr(store, "replace_sources", busy)
    counts, errors = sync(h)
    assert inline_workers == [2]
    assert (counts["ingested"], counts["failed"]) == (0, 8)
    assert sorted(e["source"] for e in errors) == sorted(str(p) for p in paths)
    assert {e["error"] for e in errors} == {"the index was replaced"}


def test_a_source_written_by_an_earlier_batch_is_replaced_not_appended(
    tmp_path, fake_embedder, inline_workers, monkeypatch
):
    paths, h = corpus(tmp_path, fake_embedder, 8)
    monkeypatch.setattr(parallel, "BATCH_ROWS", 1)
    real = parallel.prepare_files

    # A sync's diff names a path once, so only the stand-in can hand one back twice.
    def twice(make_preparer, files, *, workers):
        return real(make_preparer, [*files, files[0]], workers=workers)

    monkeypatch.setattr(parallel, "prepare_files", twice)
    sync(h)
    store = h.get().store
    assert inline_workers == [2]
    assert [s.source for s in store.list_sources() if s.duplicated] == []
    assert store.chunk_count() == 8


def test_a_parallel_sync_holds_the_sync_lock_while_it_works(
    tmp_path, fake_embedder, inline_workers, monkeypatch
):
    _, h = corpus(tmp_path, fake_embedder, 8)
    db = h.get().store.db_path
    attempts: list[Exception | None] = []
    real = Pipeline.prepare_file

    def prepare_file(self, path, **kwargs):
        if not attempts:
            rival = SyncLock(db)
            try:
                rival.acquire()
            except SyncLockBusy as e:
                attempts.append(e)
            else:
                rival.release()
                attempts.append(None)
        return real(self, path, **kwargs)

    monkeypatch.setattr(Pipeline, "prepare_file", prepare_file)
    counts, _ = sync(h)
    assert inline_workers == [2] and counts["ingested"] == 8
    assert len(attempts) == 1 and isinstance(attempts[0], SyncLockBusy)
    with sync_lock(db):
        pass


def test_tessdata_is_fetched_once_before_the_workers_start(
    tmp_path, fake_embedder, inline_workers, monkeypatch
):
    _, h = corpus(tmp_path, fake_embedder, 8)
    cfg = h.get().pipeline.config
    order: list[tuple] = []
    real = Pipeline.prepare_file

    def prepare_file(self, path, **kwargs):
        order.append(("prepare", path))
        return real(self, path, **kwargs)

    def ensure_tessdata(cache_dir, languages):
        order.append(("tessdata", cache_dir, languages))
        return cache_dir

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ensure_tessdata", ensure_tessdata)
    monkeypatch.setattr(Pipeline, "prepare_file", prepare_file)
    counts, _ = sync(h)
    assert counts["ingested"] == 8 and inline_workers == [2]
    assert [o for o in order if o[0] == "tessdata"] == [("tessdata", cfg.cache_dir, cfg.ocr_lang)]
    assert order[0][0] == "tessdata"


def test_a_sync_whose_tessdata_cannot_be_fetched_still_reads_its_files(
    tmp_path, fake_embedder, inline_workers, monkeypatch
):
    _, h = corpus(tmp_path, fake_embedder, 8)

    def ensure_tessdata(cache_dir, languages):
        raise ocr.OcrError("offline")

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ensure_tessdata", ensure_tessdata)
    counts, errors = sync(h)
    assert counts["ingested"] == 8 and inline_workers == [2] and errors == []


def test_the_store_hook_runs_once_before_the_workers_start(tmp_path, fake_embedder, inline_workers):
    _, h = corpus(tmp_path, fake_embedder, 8)
    pipeline = h.get().pipeline
    real = pipeline.before_store
    seen: list[list[int]] = []

    def before_store():
        seen.append(list(inline_workers))
        real()

    pipeline.before_store = before_store
    counts, _ = sync(h)
    assert counts["ingested"] == 8
    assert seen == [[]] and inline_workers == [2]


def test_an_unreachable_endpoint_stops_dispatch_and_says_so(
    tmp_path, fake_embedder, inline_workers, monkeypatch
):
    paths, h = corpus(tmp_path, fake_embedder, 8)
    attempted: list[Path] = []
    down = True
    real = Pipeline.prepare_file

    def prepare_file(self, path, **kwargs):
        if down:
            attempted.append(path)
            raise EndpointError("http://router.test/v1/embeddings: connection refused")
        return real(self, path, **kwargs)

    monkeypatch.setattr(Pipeline, "prepare_file", prepare_file)
    counts, errors = sync(h)
    assert len(attempted) <= 3 and counts["failed"] == 8 and counts["ingested"] == 0
    stopped = [e for e in errors if "endpoint unreachable" in e["error"]]
    assert [e["error"] for e in stopped] == ["embedding: endpoint unreachable, 5 files not synced"]
    assert len(errors) == len(attempted) + 1

    down = False
    counts, errors = sync(h)
    assert counts["ingested"] == 8 and errors == []
    assert inline_workers == [2, 2]


def test_endpoint_failures_stop_dispatch_only_when_consecutive(
    tmp_path, fake_embedder, inline_workers, monkeypatch
):
    paths, h = corpus(tmp_path, fake_embedder, 8)
    real = Pipeline.prepare_file
    # Two in a row, a success, then two more: never three in a row.
    down = {paths[i] for i in (0, 1, 3, 4)}

    def prepare_file(self, path, **kwargs):
        if path in down:
            raise EndpointError("http://router.test/v1/embeddings: connection refused")
        return real(self, path, **kwargs)

    monkeypatch.setattr(Pipeline, "prepare_file", prepare_file)
    counts, errors = sync(h)
    assert inline_workers == [2] and (counts["ingested"], counts["failed"]) == (4, 4)
    assert not any("endpoint unreachable" in e["error"] for e in errors)


HARNESS = Path(__file__).resolve().parent / "entry_point_harness.py"
# How a client starts minirag: the Desktop bundle's script, the console script, `python -m`.
ENTRY_POINTS = {
    "bundle": ("path", str(Path(__file__).resolve().parent.parent / "mcpb" / "server" / "main.py")),
    "console script": ("path", str(Path(sysconfig.get_path("scripts")) / "minirag-mcp")),
    "python -m": ("module", "minirag_mcp"),
}


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_parallel_sync_under_each_entry_point_finishes_with_stdout_untouched(tmp_path, entry):
    """Spawned workers run the parent's main script again, as `__mp_main__`, with the
    parent's argv. A script that starts the server unguarded starts one in every worker, on
    the client's stdin and stdout, and the sync waits for those workers forever."""
    kind, target = ENTRY_POINTS[entry]
    if kind == "path" and not Path(target).is_file():
        pytest.skip(f"{target} is not installed")
    root = tmp_path / "docs"
    root.mkdir()
    for i in range(8):
        (root / f"{i}.md").write_text(f"# Doc {i}\n\nBody of document {i}.", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "SYSTEMROOT")}
    env |= {
        "BASE_DIR": str(root),
        "DB_PATH": str(tmp_path / "db"),
        "CACHE_DIR": str(tmp_path / "cache"),
        "RAG_SYNC_WORKERS": "2",
        "ENTRY_KIND": kind,
        "ENTRY": target,
    }
    out, err = tmp_path / "stdout", tmp_path / "stderr"
    with out.open("wb") as stdout, err.open("wb") as stderr:
        # stdin stays open, as a client's pipe does, so a server started in a worker waits.
        proc = subprocess.Popen(
            [sys.executable, str(HARNESS)],
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            env=env,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            pytest.fail(f"the sync under the {entry} did not finish:\n{err.read_text()}")
        finally:
            proc.stdin.close()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    assert proc.returncode == 0, err.read_text()
    report = json.loads(err.read_text().splitlines()[-1])
    assert report["pools"] == [2] and report["errors"] == []
    assert report["counts"]["ingested"] == 8
    assert out.read_bytes() == b""
