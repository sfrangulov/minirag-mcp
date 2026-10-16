import json
import os
import shutil
import threading
import time
from dataclasses import asdict, replace

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


def test_sync_ingests_images_via_ocr(env, monkeypatch):
    from minirag_mcp import ocr

    cfg, store, pipeline, root = env
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "scanned text body")
    (root / "note.md").write_text("# plain note\n\nbody")
    (root / "scan.jpg").write_bytes(b"...")
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert errors == []
    assert counts["ingested"] == 2
    counts2, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts2["skipped"] == 2 and counts2["ingested"] == 0


def test_sync_keeps_indexed_images_when_the_ocr_extra_is_gone(env, monkeypatch):
    """A sync from an install without [ocr] used to delete every indexed image.

    The images are untouched on disk; only the scan whitelist shrank. Deleting them
    reports `deleted: N` and a corpus that quietly halved, recoverable only by
    re-syncing from an install that has the extra."""
    from minirag_mcp import ocr

    cfg, store, pipeline, root = env
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "scanned text body")
    (root / "note.md").write_text("# plain note\n\nbody")
    (root / "scan.jpg").write_bytes(b"...")
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 2

    monkeypatch.setattr(ocr, "available", lambda: False)
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["deleted"] == 0
    assert store.get_source(str(root / "scan.jpg")) is not None
    assert store.all_chunks(str(root / "scan.jpg"))


def test_sync_reports_the_sources_it_kept_but_cannot_read(env, monkeypatch):
    """Keeping them silently fails the same way deleting them did: the user is never
    told that part of the corpus is frozen, or which install would thaw it."""
    from minirag_mcp import ocr

    cfg, store, pipeline, root = env
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "scanned text body")
    (root / "note.md").write_text("# plain note\n\nbody")
    (root / "scan.jpg").write_bytes(b"...")
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    monkeypatch.setattr(ocr, "available", lambda: False)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["unreadable"] == 1
    assert counts["failed"] == 0  # a missing extra is not a failed run
    assert [e["source"] for e in errors] == [str(root / "scan.jpg")]
    assert "ocr" in errors[0]["error"]


@pytest.fixture
def two_roots(tmp_path, fake_embedder):
    """Two document roots and an index outside both, so either root can vanish."""
    docs, more = tmp_path / "docs", tmp_path / "more"
    docs.mkdir()
    more.mkdir()
    cfg = load_config({"BASE_DIRS": json.dumps([str(docs), str(more)])}, cwd=tmp_path)
    store = Store(tmp_path / "index", dim=fake_embedder.dim)
    (docs / "sub").mkdir()
    (docs / "a.md").write_text("# A\n\nAlpha body text for the corpus.")
    (docs / "sub" / "b.md").write_text("# B\n\nBeta body text for the corpus.")
    (more / "c.md").write_text("# C\n\nGamma body text for the corpus.")
    pipeline = Pipeline(store, fake_embedder, cfg)
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 3
    return cfg, store, pipeline, cfg.roots[0], cfg.roots[1]


def _indexed(store) -> list[str]:
    return [s.source for s in store.list_sources()]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a chmod 000 folder anyway")
def test_sync_keeps_what_is_indexed_under_a_folder_it_cannot_list(two_roots):
    """macOS refuses a launchd job the listing of ~/Documents the same way.

    os.walk skipped the folder without a word, so every indexed file under it looked
    deleted from disk, and sync deleted it from the index and reported success.
    """
    cfg, store, pipeline, docs, more = two_roots
    before = _indexed(store)
    os.chmod(docs / "sub", 0)
    try:
        counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    finally:
        os.chmod(docs / "sub", 0o755)

    assert _indexed(store) == before
    assert counts["deleted"] == 0
    assert counts["failed"] == 1
    assert [e["source"] for e in errors] == [str(docs / "sub")]
    assert "1 indexed file" in errors[0]["error"]


def test_sync_keeps_what_is_indexed_under_a_root_that_is_gone(two_roots):
    """An unmounted drive or a renamed folder is not the owner deleting every document.

    The other root still syncs.
    """
    cfg, store, pipeline, docs, more = two_roots
    before = _indexed(store)
    docs.rename(docs.with_name("docs-elsewhere"))
    (more / "d.md").write_text("# D\n\nDelta body text for the corpus.")

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert _indexed(store) == sorted([*before, str(more / "d.md")])
    assert counts["deleted"] == 0 and counts["ingested"] == 1
    assert counts["failed"] == 1
    assert [e["source"] for e in errors] == [str(docs)]
    assert "2 indexed files" in errors[0]["error"]


def test_sync_keeps_what_is_indexed_under_a_root_that_lists_empty(two_roots):
    """An empty folder left where a volume was mounted lists without any error."""
    cfg, store, pipeline, docs, more = two_roots
    before = _indexed(store)
    shutil.rmtree(docs)
    docs.mkdir()

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert _indexed(store) == before
    assert counts["deleted"] == 0 and counts["failed"] == 1
    assert [e["source"] for e in errors] == [str(docs)]
    assert "2 indexed files" in errors[0]["error"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a chmod 000 folder anyway")
def test_an_unlistable_folder_with_nothing_indexed_is_reported_but_does_not_fail(two_roots):
    """Nothing in the index depends on it, and failing would fail every sync forever."""
    cfg, store, pipeline, docs, more = two_roots
    (more / "private").mkdir()
    os.chmod(more / "private", 0)
    try:
        counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    finally:
        os.chmod(more / "private", 0o755)

    assert counts["failed"] == 0
    assert [e["source"] for e in errors] == [str(more / "private")]


def test_a_scoped_sync_is_not_failed_by_a_root_outside_its_scope(two_roots):
    cfg, store, pipeline, docs, more = two_roots
    shutil.rmtree(docs)
    (more / "d.md").write_text("# D\n\nDelta body text for the corpus.")

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size, scope=more)

    assert counts["failed"] == 0 and errors == []
    assert counts["ingested"] == 1
    assert str(docs / "a.md") in _indexed(store)


def test_sync_still_deletes_a_file_removed_beside_a_kept_folder(two_roots):
    cfg, store, pipeline, docs, more = two_roots
    shutil.rmtree(docs)
    (more / "d.md").write_text("# D\n\nDelta body text for the corpus.")
    (more / "c.md").unlink()

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["deleted"] == 1
    assert str(more / "c.md") not in _indexed(store)
    assert str(docs / "a.md") in _indexed(store)


def test_a_pending_job_reports_the_counter_shape_a_finished_one_does(env):
    """sync_status is polled from the moment start() returns, so a payload that grows a
    counter only once the job finishes makes every earlier poll quietly incomplete."""
    from minirag_mcp.sync import SyncJob

    cfg, store, pipeline, root = env
    seed(root)
    pending = SyncJob(job_id="not-started")

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert set(pending.to_dict()["counts"]) == set(counts)


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


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a chmod 000 folder anyway")
def test_an_unlistable_folder_inside_two_overlapping_roots_is_reported_once(
    tmp_path, fake_embedder
):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    (inner / "a.md").write_text("# A\n\nAlpha body text for the corpus.")
    cfg = load_config({"BASE_DIRS": json.dumps([str(outer), str(inner)])}, cwd=tmp_path)
    store = Store(tmp_path / "index", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    os.chmod(inner, 0)
    try:
        counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    finally:
        os.chmod(inner, 0o755)

    assert [e["source"] for e in errors] == [str(cfg.roots[1])]
    assert counts["failed"] == 1 and counts["deleted"] == 0


# --- a file moved by hand: same content, same name, new folder ---------------------


@pytest.fixture
def moving(tmp_path, fake_embedder):
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
    store = Store(tmp_path / "index", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)
    parsed: list[str] = []
    real = pipeline.ingest_file

    def spy(path):
        parsed.append(str(path))
        return real(path)

    pipeline.ingest_file = spy
    return cfg, store, pipeline, cfg.roots[0], parsed


def _body(n: int, word: str) -> str:
    return "\n\n".join(f"## Part {i}\n\n" + f"{word} paragraph {i}. " * 60 for i in range(n))


def test_sync_follows_a_file_moved_by_hand_without_reading_it_again(moving):
    """Re-filing a document in Finder used to cost a full parse and embed, which for a
    scan is OCR all over again."""
    cfg, store, pipeline, root, parsed = moving
    (root / "report.md").write_text("# Report\n\n" + _body(3, "alpha"))
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    chunks_before = len(store.all_chunks(str(root / "report.md")))
    parsed.clear()

    (root / "report.md").rename(root / "sub" / "report.md")
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert errors == []
    assert counts["moved"] == 1
    assert counts["ingested"] == 0 and counts["deleted"] == 0
    assert parsed == []
    assert [s.source for s in store.list_sources()] == [str(root / "sub" / "report.md")]
    assert len(store.all_chunks(str(root / "sub" / "report.md"))) == chunks_before
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["skipped"] == 1 and counts["moved"] == 0


def test_a_new_file_at_the_old_path_does_not_collide_with_the_moved_one(moving):
    """Chunk and section ids carry the source path. Left behind by a move, they are
    exactly the ids the next file at the old path gets, and its ingest would overwrite
    the moved document's rows or merge the two into one section."""
    cfg, store, pipeline, root, parsed = moving
    (root / "notes.md").write_text("# Notes\n\n" + _body(2, "alpha"))
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    (root / "notes.md").rename(root / "sub" / "notes.md")
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    (root / "notes.md").write_text("# Notes\n\n" + _body(2, "omega"))
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    moved = store.all_chunks(str(root / "sub" / "notes.md"))
    fresh = store.all_chunks(str(root / "notes.md"))
    assert moved and fresh
    assert all("alpha" in c.text and "omega" not in c.text for c in moved)
    assert all("omega" in c.text and "alpha" not in c.text for c in fresh)
    for chunk in moved:
        assert "omega" not in store.parent_text(chunk.parent_id)
    for chunk in fresh:
        assert "alpha" not in store.parent_text(chunk.parent_id)


def test_a_renamed_file_is_read_again_because_its_title_can_come_from_its_name(moving):
    cfg, store, pipeline, root, parsed = moving
    (root / "draft_v1.md").write_text(_body(1, "alpha"))
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    (root / "draft_v1.md").rename(root / "sub" / "final_contract.md")

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["moved"] == 0
    assert counts["ingested"] == 1 and counts["deleted"] == 1
    assert "final contract" in store.get_source(str(root / "sub" / "final_contract.md")).title


def test_a_moved_file_that_was_also_edited_is_read_again(moving):
    cfg, store, pipeline, root, parsed = moving
    (root / "plan.md").write_text("# Plan\n\n" + _body(1, "alpha"))
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    (root / "plan.md").rename(root / "sub" / "plan.md")
    (root / "sub" / "plan.md").write_text("# Plan\n\n" + _body(1, "omega"))

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["moved"] == 0 and counts["ingested"] == 1 and counts["deleted"] == 1
    assert all("omega" in c.text for c in store.all_chunks(str(root / "sub" / "plan.md")))


def test_one_vanished_file_moves_to_one_of_its_two_new_copies(moving):
    cfg, store, pipeline, root, parsed = moving
    text = "# Memo\n\n" + _body(1, "alpha")
    (root / "memo.md").write_text(text)
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    (root / "memo.md").unlink()
    (root / "a").mkdir()
    (root / "b").mkdir()
    (root / "a" / "memo.md").write_text(text)
    (root / "b" / "memo.md").write_text(text)

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["moved"] == 1 and counts["ingested"] == 1 and counts["deleted"] == 0
    assert [s.source for s in store.list_sources()] == [
        str(root / "a" / "memo.md"),
        str(root / "b" / "memo.md"),
    ]


def test_a_moved_file_cut_by_an_older_scheme_is_read_again(moving):
    """Its chunks need re-cutting anyway; moving them would only delay that a sync."""
    cfg, store, pipeline, root, parsed = moving
    doc = root / "old.md"
    doc.write_text("# Old\n\n" + _body(1, "alpha"))
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    stale = [replace(c, scheme_version=SCHEME_VERSION - 1) for c in _records_of(store, str(doc))]
    store.replace_source(str(doc), stale)
    doc.rename(root / "sub" / "old.md")

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["moved"] == 0 and counts["ingested"] == 1
    assert store.stale_chunk_count() == 0


def test_sync_repairs_a_source_a_racing_first_ingest_stored_twice(env):
    """Its bytes and mtime match the index, so nothing else would ever re-read it."""
    cfg, store, pipeline, root = env
    seed(root)
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    doc = str(root / "a.md")
    store._table.add([asdict(r) for r in _records_of(store, doc)])
    assert len(store.all_chunks(doc)) == 2 * len({c.chunk_index for c in store.all_chunks(doc)})

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["ingested"] == 1
    indexes = [c.chunk_index for c in store.all_chunks(doc)]
    assert indexes == sorted(set(indexes))


def test_files_moved_out_of_a_root_left_empty_are_followed_not_held(tmp_path, fake_embedder):
    """Emptying a drop folder into the archive is a move, and the files prove it.

    An empty root alone is held, in case a volume is unmounted. When every file it held
    shows up elsewhere with the same name and bytes, that doubt is gone.
    """
    archive, inbox = tmp_path / "archive", tmp_path / "inbox"
    archive.mkdir()
    inbox.mkdir()
    cfg = load_config({"BASE_DIRS": json.dumps([str(archive), str(inbox)])}, cwd=tmp_path)
    store = Store(tmp_path / "index", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)
    for name in ("a.md", "b.md"):
        (inbox / name).write_text(f"# {name}\n\nBody of {name} for the corpus.")
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    for name in ("a.md", "b.md"):
        (inbox / name).rename(archive / name)

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert errors == []
    assert counts["moved"] == 2 and counts["ingested"] == 0 and counts["failed"] == 0
    assert _indexed(store) == [str(cfg.roots[0] / "a.md"), str(cfg.roots[0] / "b.md")]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a chmod 000 folder anyway")
def test_syncing_exactly_a_folder_that_cannot_be_listed_fails(two_roots):
    """Nothing indexed is at stake, but nothing the user asked for was done either."""
    cfg, store, pipeline, docs, more = two_roots
    (docs / "new").mkdir()
    (docs / "new" / "n.md").write_text("# N\n\nNew body text for the corpus.")
    os.chmod(docs / "new", 0)
    try:
        counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size, scope=docs / "new")
    finally:
        os.chmod(docs / "new", 0o755)

    assert counts["failed"] == 1 and counts["ingested"] == 0
    assert [e["source"] for e in errors] == [str(docs / "new")]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a chmod 000 folder anyway")
def test_a_scoped_sync_reports_an_unlistable_folder_inside_its_scope(two_roots):
    cfg, store, pipeline, docs, more = two_roots
    os.chmod(docs / "sub", 0)
    try:
        counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size, scope=docs)
    finally:
        os.chmod(docs / "sub", 0o755)

    assert counts["failed"] == 1 and counts["deleted"] == 0
    assert [e["source"] for e in errors] == [str(docs / "sub")]
    assert "b.md" in errors[0]["error"]


def test_a_scoped_sync_inside_a_root_that_is_gone_keeps_it_and_fails(two_roots):
    cfg, store, pipeline, docs, more = two_roots
    before = _indexed(store)
    shutil.rmtree(docs)

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size, scope=docs / "sub")

    assert _indexed(store) == before
    assert counts["failed"] == 1 and counts["deleted"] == 0
    assert [e["source"] for e in errors] == [str(docs)]


def test_a_root_that_is_gone_fails_the_sync_even_with_nothing_indexed(tmp_path, fake_embedder):
    """A configured root that does not exist is a typo or an unplugged drive, not a no-op."""
    docs = tmp_path / "docs"
    cfg = load_config({"BASE_DIR": str(docs)}, cwd=tmp_path)
    store = Store(tmp_path / "index", dim=fake_embedder.dim)

    counts, errors = run_sync(Pipeline(store, fake_embedder, cfg), store, cfg.roots, 10**9)

    assert counts["failed"] == 1
    assert [e["source"] for e in errors] == [str(cfg.roots[0])]


def test_an_empty_root_with_nothing_indexed_is_simply_empty(tmp_path, fake_embedder):
    docs = tmp_path / "docs"
    docs.mkdir()
    cfg = load_config({"BASE_DIR": str(docs)}, cwd=tmp_path)
    store = Store(tmp_path / "index", dim=fake_embedder.dim)

    counts, errors = run_sync(Pipeline(store, fake_embedder, cfg), store, cfg.roots, 10**9)

    assert counts["failed"] == 0 and errors == []


def test_removing_the_last_file_of_a_subfolder_still_deletes_it(two_roots):
    """Only an empty root is suspect; an emptied subfolder is just emptied."""
    cfg, store, pipeline, docs, more = two_roots
    (docs / "sub" / "b.md").unlink()

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["deleted"] == 1 and counts["failed"] == 0 and errors == []
    assert str(docs / "sub" / "b.md") not in _indexed(store)


def test_a_root_that_is_gone_does_not_hold_a_sibling_whose_name_extends_it(tmp_path, fake_embedder):
    docs, archive = tmp_path / "docs", tmp_path / "docs-archive"
    docs.mkdir()
    archive.mkdir()
    cfg = load_config({"BASE_DIRS": json.dumps([str(docs), str(archive)])}, cwd=tmp_path)
    store = Store(tmp_path / "index", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)
    (docs / "a.md").write_text("# A\n\nAlpha body text for the corpus.")
    (archive / "y.md").write_text("# Y\n\nYankee body text for the corpus.")
    (archive / "z.md").write_text("# Z\n\nZulu body text for the corpus.")
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    shutil.rmtree(docs)
    (archive / "z.md").unlink()

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["deleted"] == 1
    assert str(cfg.roots[1] / "z.md") not in _indexed(store)
    assert str(cfg.roots[0] / "a.md") in _indexed(store)


def test_a_file_moved_over_an_indexed_file_of_the_same_name_replaces_it(moving):
    """The destination's own chunks must go: a re-path only removes the origin's."""
    cfg, store, pipeline, root, parsed = moving
    (root / "x.md").write_text("# X\n\n" + _body(1, "alpha"))
    (root / "sub" / "x.md").write_text("# X\n\n" + _body(4, "omega"))
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    (root / "x.md").rename(root / "sub" / "x.md")

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["deleted"] == 1 and counts["moved"] == 0
    chunks = store.all_chunks(str(root / "sub" / "x.md"))
    assert chunks and all("omega" not in c.text for c in chunks)


def _paired_move(moving):
    cfg, store, pipeline, root, parsed = moving
    (root / "r.md").write_text("# R\n\n" + _body(1, "alpha"))
    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    (root / "r.md").rename(root / "sub" / "r.md")
    return cfg, store, pipeline, root


def test_a_move_whose_origin_vanished_meanwhile_is_read_instead(moving, monkeypatch):
    """delete_file takes no lock, so the origin can go between planning and re-pathing."""
    cfg, store, pipeline, root = _paired_move(moving)
    real = store.move_source

    def raced(old, new, *, mtime):
        store.delete_source(old)
        return real(old, new, mtime=mtime)

    monkeypatch.setattr(store, "move_source", raced)
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["moved"] == 0 and counts["ingested"] == 1 and counts["failed"] == 0
    assert _indexed(store) == [str(root / "sub" / "r.md")]


def test_a_move_that_raises_is_read_instead_and_the_origin_goes(moving, monkeypatch):
    cfg, store, pipeline, root = _paired_move(moving)

    def broken(old, new, *, mtime):
        raise RuntimeError("commit conflict")

    monkeypatch.setattr(store, "move_source", broken)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["moved"] == 0 and counts["failed"] == 0
    assert counts["ingested"] == 1 and counts["deleted"] == 1
    assert _indexed(store) == [str(root / "sub" / "r.md")]


def test_a_move_records_the_new_files_mtime(moving):
    cfg, store, pipeline, root = _paired_move(moving)
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    new = root / "sub" / "r.md"
    assert counts["moved"] == 1 and counts["scanned"] == 1
    assert store.get_source(str(new)).mtime == new.stat().st_mtime


def test_a_doubled_source_that_moved_is_read_again_rather_than_carried_over(moving):
    cfg, store, pipeline, root = _paired_move(moving)
    store._table.add([asdict(r) for r in _records_of(store, str(root / "r.md"))])

    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["moved"] == 0 and counts["ingested"] == 1
    indexes = [c.chunk_index for c in store.all_chunks(str(root / "sub" / "r.md"))]
    assert indexes == sorted(set(indexes))


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a chmod 000 folder anyway")
def test_a_copy_beside_a_folder_that_cannot_be_listed_does_not_take_its_entry(two_roots):
    """Behind a folder that cannot be listed the original may well still be there.

    Re-pathing it to the copy dropped its own path from the index, and deleting the
    copy before access came back lost the document entirely.
    """
    cfg, store, pipeline, docs, more = two_roots
    (more / "b.md").write_bytes((docs / "sub" / "b.md").read_bytes())
    os.chmod(docs / "sub", 0)
    try:
        counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    finally:
        os.chmod(docs / "sub", 0o755)

    assert counts["moved"] == 0 and counts["ingested"] == 1
    assert str(docs / "sub" / "b.md") in _indexed(store)
    assert str(more / "b.md") in _indexed(store)
    assert "b.md" in errors[0]["error"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a chmod 000 folder anyway")
def test_a_scoped_sync_reached_through_an_inner_root_is_not_failed_by_its_parent(
    tmp_path, fake_embedder
):
    """docs/a can be passed through but not listed; docs/a/b is a root of its own."""
    docs = tmp_path / "docs"
    inner = docs / "a" / "b"
    inner.mkdir(parents=True)
    cfg = load_config({"BASE_DIRS": json.dumps([str(docs), str(inner)])}, cwd=tmp_path)
    store = Store(tmp_path / "index", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)
    (inner / "y.md").write_text("# Y\n\nYankee body text for the corpus.")
    os.chmod(docs / "a", 0o311)
    try:
        counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size, scope=cfg.roots[1])
    finally:
        os.chmod(docs / "a", 0o755)

    assert counts["ingested"] == 1 and counts["failed"] == 0
