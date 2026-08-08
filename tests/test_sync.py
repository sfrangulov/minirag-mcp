import threading
import time
from dataclasses import replace

import pytest

import minirag_mcp.sync as sync_mod
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import load_config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.lock import SyncLock, SyncLockBusy, sync_lock
from minirag_mcp.store import ChunkRecord, Store
from minirag_mcp.sync import SyncBusyError, SyncManager, run_sync


@pytest.fixture
def env(tmp_path, fake_embedder):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    store = Store(tmp_path / ".minirag" / "lancedb", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)
    return cfg, store, pipeline, tmp_path


def seed(root):
    (root / "sub").mkdir(exist_ok=True)
    (root / "a.md").write_text("# A\n\nAlpha body text for the corpus.")
    (root / "sub" / "b.md").write_text("# B\n\nBeta body text for the corpus.")


def test_run_sync_ingests_and_deletes(env):
    cfg, store, pipeline, root = env
    seed(root)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 2 and counts["scanned"] == 2 and errors == []
    (root / "a.md").unlink()
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["deleted"] == 1 and counts["skipped"] == 1
    assert store.get_source(str(root / "a.md")) is None


def test_run_sync_collects_per_file_errors(env, monkeypatch):
    cfg, store, pipeline, root = env
    seed(root)

    real = pipeline.ingest_file

    def flaky(path):
        if path.name == "b.md":
            raise RuntimeError("boom")
        return real(path)

    monkeypatch.setattr(pipeline, "ingest_file", flaky)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 1 and counts["failed"] == 1
    assert errors and "boom" in errors[0]["error"]


def test_manager_lifecycle(env):
    cfg, store, pipeline, root = env
    seed(root)
    mgr = SyncManager(pipeline, store, cfg)
    job_id = mgr.start()
    mgr.wait()
    job = mgr.status(job_id)
    assert job.state == "succeeded"
    assert job.counts["ingested"] == 2
    d = job.to_dict()
    assert d["jobId"] == job_id and d["startedAt"] and d["finishedAt"]


def test_manager_rejects_concurrent_and_forgets_old(env):
    cfg, store, pipeline, root = env
    seed(root)
    gate = threading.Event()
    real = pipeline.ingest_file

    def slow(path):
        gate.wait(5)
        return real(path)

    pipeline.ingest_file = slow
    mgr = SyncManager(pipeline, store, cfg)
    first = mgr.start()
    with pytest.raises(SyncBusyError):
        mgr.start()
    gate.set()
    mgr.wait()
    second = mgr.start()
    mgr.wait()
    with pytest.raises(KeyError):
        mgr.status(first)  # only the latest job is retained
    assert mgr.status(second).state == "succeeded"


def test_unknown_job_id(env):
    cfg, store, pipeline, _ = env
    mgr = SyncManager(pipeline, store, cfg)
    with pytest.raises(KeyError):
        mgr.status("nope")


def test_run_sync_scope_single_file(env):
    cfg, store, pipeline, root = env
    seed(root)
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size, scope=root / "a.md")
    assert counts["ingested"] == 1


def test_run_sync_scope_through_symlink(env, tmp_path):
    """Regression: an unresolved scope silently matched zero files (roots are resolved)."""
    cfg, store, pipeline, root = env
    seed(root)
    link = tmp_path / "link_root"
    link.symlink_to(root, target_is_directory=True)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size, scope=link / "sub")
    assert counts["ingested"] == 1 and errors == []


def test_run_sync_never_indexes_symlink_escaping_the_root(env, tmp_path):
    """Security: a symlink inside a root pointing outside it must not be ingested.

    The extension whitelist sees the link name (.md) while the parser reads the
    target, so without a containment check any readable file — extension-less
    private keys included — lands in the index.
    """
    cfg, store, pipeline, root = env
    outside = tmp_path.parent / f"{tmp_path.name}_outside"  # sibling of the document root
    outside.mkdir()
    secret = outside / "id_rsa"
    secret.write_text("-----BEGIN PRIVATE KEY-----\nSUPERSECRET\n-----END PRIVATE KEY-----\n")
    (root / "readme.md").symlink_to(secret)
    (root / "real.md").write_text("# Real\n\nOrdinary body text that belongs in the corpus.")

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["ingested"] == 1 and errors == []
    assert store.get_source(str(root / "readme.md")) is None
    assert all("SUPERSECRET" not in c.text for c in store.all_chunks(str(root / "real.md")))
    indexed = {s.source for s in store.list_sources()}
    assert indexed == {str(root / "real.md")}


def test_run_sync_purges_content_leaked_by_an_escaping_symlink(env, tmp_path):
    """An index written before the containment check is cleaned up by the next sync."""
    cfg, store, pipeline, root = env
    outside = tmp_path.parent / f"{tmp_path.name}_leak"
    outside.mkdir()
    secret = outside / "id_rsa"
    secret.write_text("-----BEGIN PRIVATE KEY-----\nLEAKED\n-----END PRIVATE KEY-----\n")
    link = root / "readme.md"
    link.symlink_to(secret)
    pipeline.ingest_file(link)  # simulate the pre-fix state: the target got indexed
    assert store.get_source(str(link)) is not None

    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert store.get_source(str(link)) is None
    assert store.chunk_count() == 0


# --- the cross-process sync lock, at the two entry points ------------------------
#
# These hold the lock from the test process rather than spawning a second one. A
# same-process rival contends exactly like a foreign one — flock keys on the open
# file description, and a second `os.open` makes a second one — so this is a
# faithful stand-in for "another process is syncing", without a spawn per test.
# tests/test_lock.py covers what genuinely needs two processes: who gets named,
# and what a dying holder leaves behind.


def test_run_sync_refuses_while_another_process_syncs(env):
    cfg, store, pipeline, root = env
    seed(root)
    with sync_lock(store.db_path):
        with pytest.raises(SyncLockBusy, match="syncing this index"):
            run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    # Once the rival is gone the very same call goes through.
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 2


def test_manager_start_refuses_immediately_and_records_nothing(env):
    """Contention has to surface from `start()`, not from the job it returns.

    Acquiring inside the worker would hand the caller a jobId for a job that is
    already doomed. It also must leave no trace: a half-registered 'pending' job
    would make every later start() raise SyncBusyError forever.
    """
    cfg, store, pipeline, root = env
    seed(root)
    mgr = SyncManager(pipeline, store, cfg)

    with sync_lock(store.db_path):
        with pytest.raises(SyncLockBusy, match="syncing this index"):
            mgr.start()

    job_id = mgr.start()  # not wedged by the refusal
    mgr.wait()
    assert mgr.status(job_id).state == "succeeded"


def test_manager_holds_the_lock_for_the_job_and_releases_it_after(env):
    cfg, store, pipeline, root = env
    seed(root)
    gate = threading.Event()
    real = pipeline.ingest_file

    def slow(path):
        gate.wait(5)
        return real(path)

    pipeline.ingest_file = slow
    mgr = SyncManager(pipeline, store, cfg)
    mgr.start()
    rival = SyncLock(store.db_path)
    with pytest.raises(SyncLockBusy):
        rival.acquire()  # held while the job runs

    gate.set()
    mgr.wait()
    rival.acquire()  # released when the job ended
    rival.release()


def test_manager_releases_the_lock_when_the_job_fails(env, monkeypatch):
    """A catastrophic failure must not strand the lock for the process's lifetime."""
    cfg, store, pipeline, root = env
    seed(root)
    monkeypatch.setattr(
        sync_mod, "scan_roots", lambda roots: (_ for _ in ()).throw(RuntimeError("disk gone"))
    )
    mgr = SyncManager(pipeline, store, cfg)
    job_id = mgr.start()
    mgr.wait()

    assert mgr.status(job_id).state == "failed"
    with sync_lock(store.db_path):
        pass  # no raise = released


def test_in_process_guard_fires_before_the_cross_process_lock(env):
    """The anti-deadlock property: this process never contends with itself.

    `start()` holds the flock for the running job, and a second acquire from the
    same process would conflict like any other rival — so the in-process check
    has to reject a second job first. If the order were reversed, this would
    raise SyncLockBusy naming our own PID, or wedge.
    """
    cfg, store, pipeline, root = env
    seed(root)
    gate = threading.Event()
    real = pipeline.ingest_file

    def slow(path):
        gate.wait(5)
        return real(path)

    pipeline.ingest_file = slow
    mgr = SyncManager(pipeline, store, cfg)
    mgr.start()
    try:
        with pytest.raises(SyncBusyError) as excinfo:
            mgr.start()
        assert not isinstance(excinfo.value, SyncLockBusy)
    finally:
        gate.set()
        mgr.wait()


def test_restarting_the_moment_a_job_reports_terminal_is_never_refused(env, monkeypatch):
    """Regression: the documented start → poll → start pattern hit our own flock.

    The worker used to publish the terminal state and only then release the
    cross-process lock. A client doing exactly what `sync_start` documents —
    poll `sync_status` until the state is terminal, then start the next sync —
    could land in the gap: the in-process guard let it through, because the job
    reads 'succeeded', and the flock this same process was still holding refused
    it, with a message naming the caller's own PID.

    The gap is a few microseconds wide, so the test widens it at the seam it
    belongs to rather than betting on the scheduler: `release()` is made to
    linger before it lets the flock go. That invents nothing — releasing really
    does take three syscalls, and a loaded machine or a slow filesystem stretches
    them — it only makes the existing interval long enough to observe. With the
    old ordering the poller sees 'succeeded' during the lingering and is refused
    every single time; with the release moved ahead of the state, no delay can
    expose a window, because there is none to expose.

    (Shortening `sys.setswitchinterval` was tried first and rejected: measured
    against the old ordering it does not concentrate the race — 400 restarts at
    each of 1e-7, 1e-6, 1e-5, 1e-4 and the 5e-3 default yielded 0 or 1 refusals,
    the same ~5-in-1000 as an untouched interpreter. A test that flaky proves
    nothing when it passes.)
    """
    cfg, store, pipeline, root = env
    seed(root)

    class LingeringLock(SyncLock):
        """Holds the flock well past the moment a poller could act on the job."""

        def release(self) -> None:
            if self.held:
                time.sleep(0.05)
            super().release()

    monkeypatch.setattr(sync_mod, "SyncLock", LingeringLock)
    mgr = SyncManager(pipeline, store, cfg)

    first = mgr.start()
    while mgr.status(first).state not in ("succeeded", "failed"):
        pass  # a client polling sync_status, with nothing in between
    assert mgr.status(first).state == "succeeded"

    try:
        second = mgr.start()
    except SyncLockBusy as e:  # the bug: refused by the lock this process holds
        pytest.fail(f"a sync that had just reported terminal was refused by our own lock: {e}")
    mgr.wait()
    assert mgr.status(second).state == "succeeded"


def test_finished_at_set_when_state_terminal(env):
    cfg, store, pipeline, root = env
    seed(root)
    mgr = SyncManager(pipeline, store, cfg)
    job_id = mgr.start()
    mgr.wait()
    job = mgr.status(job_id)
    assert job.state in ("succeeded", "failed") and job.finished_at is not None


def test_sync_re_ingests_a_source_whose_chunking_scheme_is_stale(tmp_path, fake_embedder):
    """`status` tells the user to re-sync a stale index; the re-sync has to comply.

    A sync skips a file whose bytes are unchanged, and after a scheme change every
    file's bytes are unchanged — so without this the advice would be a dead end."""
    root = tmp_path / "docs"
    root.mkdir()
    doc = root / "spec.md"
    doc.write_text("# 1 Хранение\n\n" + "Правило про хранение запасов. " * 40, encoding="utf-8")
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
    store = Store(tmp_path / "db", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 1
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts == {**counts, "ingested": 0, "skipped": 1}

    # rewrite the source's chunks as an older scheme would have left them
    old = [
        replace(c, scheme_version=SCHEME_VERSION - 1, parent_id="")
        for c in _records_of(store, str(doc))
    ]
    store.replace_source(str(doc), old)
    assert store.stale_chunk_count() == len(old)

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 1 and counts["skipped"] == 0
    assert store.stale_chunk_count() == 0


def _records_of(store, source):
    rows = store._table.search().where(f"source = '{source}'").limit(10_000).to_list()
    rows.sort(key=lambda r: r["chunk_index"])
    return [
        ChunkRecord(
            id=r["id"],
            source=r["source"],
            source_type=r["source_type"],
            title=r["title"],
            chunk_index=r["chunk_index"],
            text=r["text"],
            vector=list(r["vector"]),
            file_hash=r["file_hash"],
            mtime=r["mtime"],
            ingested_at=r["ingested_at"],
            parent_id=r["parent_id"],
            scheme_version=r["scheme_version"],
        )
        for r in rows
    ]
