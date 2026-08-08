"""The sync lock, exercised across real processes.

Mocks would prove nothing here: every property under test is a property of the
kernel's flock table, not of Python objects. So a child process really takes the
lock, and the parent really fails to.

Spawn, not fork, is used deliberately. The tests share a process with the rest of
the suite, which has by then started LanceDB's Rust runtime and its threads;
forking that and doing anything non-trivial in the child is asking for a
deadlock. Spawn costs a fresh interpreter per child and buys determinism.
"""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from minirag_mcp.config import load_config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.lock import LOCK_FILENAME, SyncLock, SyncLockBusy, sync_lock
from minirag_mcp.store import Store

MP = multiprocessing.get_context("spawn")
TIMEOUT = 60  # generous: a spawned child pays a fresh interpreter startup


def _hold_until_told(db_path: str, ready, release) -> None:
    """Child entry point: take the lock, announce it, hold until told to stop.

    Must stay importable at module scope — spawn re-imports this module in the
    child and looks the target up by name.
    """
    from minirag_mcp.lock import sync_lock  # re-imported in the fresh interpreter

    with sync_lock(Path(db_path)):
        ready.set()
        release.wait(TIMEOUT)


class Holder:
    """A child process holding the sync lock on `db_path`."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.lock_file = db_path / LOCK_FILENAME
        self._release = MP.Event()
        ready = MP.Event()
        self.proc = MP.Process(target=_hold_until_told, args=(str(db_path), ready, self._release))
        self.proc.start()
        assert ready.wait(TIMEOUT), "child never reported that it took the lock"

    def stop(self) -> None:
        """Let the child finish normally and wait for it to be gone."""
        self._release.set()
        self.proc.join(TIMEOUT)
        assert self.proc.exitcode == 0

    def kill(self) -> None:
        os.kill(self.proc.pid, signal.SIGKILL)
        self.proc.join(TIMEOUT)

    def cleanup(self) -> None:
        if self.proc.is_alive():
            self._release.set()
            self.proc.join(5)
        if self.proc.is_alive():  # pragma: no cover - only if the child wedged
            self.proc.kill()
            self.proc.join(5)


@pytest.fixture
def db_path(tmp_path) -> Path:
    p = tmp_path / ".minirag" / "lancedb"
    p.mkdir(parents=True)
    return p


@pytest.fixture
def holder(db_path):
    """A real second process holding the lock. Costs a spawned interpreter."""
    h = Holder(db_path)
    try:
        yield h
    finally:
        h.cleanup()


@pytest.fixture
def held(db_path):
    """The lock held by *this* process — enough to test the refusal path.

    A same-process rival contends identically to a cross-process one (a second
    `os.open` is a second open file description, which flock treats as someone
    else), so the tests that only care about what the refusal *says* use this
    and skip the spawn. The tests that care about process lifetime — who is
    named, and what a dying holder does — use `holder` instead.
    """
    lock = SyncLock(db_path)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


def test_second_process_is_refused_and_told_who_holds_it(holder):
    with pytest.raises(SyncLockBusy) as excinfo:
        with sync_lock(holder.db_path):
            pytest.fail("acquired a lock another process is holding")

    msg = str(excinfo.value)
    assert str(holder.proc.pid) in msg, msg
    assert "syncing this index" in msg and "different DB_PATH" in msg, msg
    assert excinfo.value.pid == holder.proc.pid
    assert excinfo.value.path == holder.lock_file


def test_holder_writes_its_pid_and_start_time_while_held(holder):
    data = json.loads(holder.lock_file.read_text())
    assert data["pid"] == holder.proc.pid
    # Parseable as an aware timestamp, and recent.
    started = datetime.fromisoformat(data["startedAt"])
    assert started.tzinfo is not None
    assert abs((datetime.now(UTC) - started).total_seconds()) < TIMEOUT


def test_lock_is_released_when_the_holder_exits_normally(holder):
    holder.stop()
    with sync_lock(holder.db_path):
        pass  # no raise = the lock was free


def test_lock_is_released_when_the_holder_is_sigkilled(holder):
    """The property that makes stale locks impossible.

    SIGKILL cannot be caught, so no cleanup code of ours runs — the kernel drops
    the lock when it tears down the process's file descriptors. This is why
    there is no stale-lock recovery path to get wrong.
    """
    with pytest.raises(SyncLockBusy):  # genuinely held before the kill
        with sync_lock(holder.db_path):
            pass

    holder.kill()
    assert holder.proc.exitcode == -signal.SIGKILL

    # No polling, no retry, no reaping of a stale file: it is free the moment the
    # process is gone. The lock file itself survives, still naming the dead holder,
    # which is exactly why its contents are never treated as proof of a live lock.
    with sync_lock(holder.db_path):
        pass
    assert holder.lock_file.exists()


def test_release_leaves_no_holder_info_behind(db_path):
    with sync_lock(db_path) as lock:
        assert lock.held
        assert json.loads(lock.path.read_text())["pid"] == os.getpid()
    assert not lock.held
    assert (db_path / LOCK_FILENAME).read_text() == ""


def test_lock_is_created_under_a_db_path_that_does_not_exist_yet(tmp_path):
    missing = tmp_path / "not" / "made" / "yet"
    with sync_lock(missing):
        assert (missing / LOCK_FILENAME).exists()


def test_reacquiring_the_same_lock_object_is_a_loud_error(db_path):
    """Not reentrant by design: a second flock from this process would conflict."""
    lock = SyncLock(db_path)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already holds"):
            lock.acquire()
    finally:
        lock.release()
    lock.release()  # idempotent


def test_a_second_lock_object_in_this_process_conflicts_too(db_path):
    """Why SyncManager must not nest its acquisition inside run_sync."""
    with sync_lock(db_path):
        with pytest.raises(SyncLockBusy):
            with sync_lock(db_path):
                pass


@pytest.mark.parametrize(
    "contents",
    ["", "   \n", "not json at all", "[1, 2, 3]", "{}", '{"pid": "not-a-number"}', "null"],
    ids=["empty", "whitespace", "garbage", "list", "no-pid", "pid-not-int", "null"],
)
def test_unreadable_holder_info_degrades_to_a_generic_message(held, contents):
    """A holder that died between creating the file and stamping it leaves this.

    The lock file is advisory data, not the lock — writing to it needs no lock —
    so the test can stage each shape while the lock is genuinely held.
    """
    held.path.write_text(contents)

    with pytest.raises(SyncLockBusy) as excinfo:
        with sync_lock(held.path.parent):
            pass

    msg = str(excinfo.value)
    assert "Another minirag-mcp process is syncing this index" in msg, msg
    assert "different DB_PATH" in msg, msg
    assert "PID" not in msg, msg  # no pid invented for one we could not read
    assert excinfo.value.pid is None


def test_message_reports_how_long_the_holder_has_been_running(held):
    started = datetime.now(UTC) - timedelta(minutes=4, seconds=12)
    held.path.write_text(json.dumps({"pid": 12345, "startedAt": started.isoformat()}))

    with pytest.raises(SyncLockBusy) as excinfo:
        with sync_lock(held.path.parent):
            pass

    assert (
        str(excinfo.value) == "Another minirag-mcp process (PID 12345) has been syncing "
        "this index for 4m12s. Wait for it to finish, or use a different DB_PATH."
    )


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (timedelta(seconds=9), "9s"),
        (timedelta(minutes=4, seconds=12), "4m12s"),
        (timedelta(hours=2, minutes=3, seconds=4), "2h03m04s"),
    ],
)
def test_duration_rendering(held, delta, expected):
    started = datetime.now(UTC) - delta
    held.path.write_text(json.dumps({"pid": 999, "startedAt": started.isoformat()}))
    with pytest.raises(SyncLockBusy, match=f"for {expected}\\."):
        with sync_lock(held.path.parent):
            pass


def test_an_unparseable_start_time_still_names_the_holder(held):
    held.path.write_text(json.dumps({"pid": 4242, "startedAt": "yesterday-ish"}))
    with pytest.raises(SyncLockBusy) as excinfo:
        with sync_lock(held.path.parent):
            pass
    assert "(PID 4242) is syncing this index" in str(excinfo.value)
    assert excinfo.value.pid == 4242


def test_a_start_time_in_the_future_does_not_render_a_duration(held):
    """Clock skew between two machines sharing a mount, or a stamp from the future."""
    ahead = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    held.path.write_text(json.dumps({"pid": 7, "startedAt": ahead}))
    with pytest.raises(SyncLockBusy, match=r"\(PID 7\) is syncing this index"):
        with sync_lock(held.path.parent):
            pass


# --- a lock that cannot be established at all ------------------------------------
#
# None of these are contention: nobody holds anything, we never got as far as
# asking. Before this lock existed every one of them synced cleanly, so refusing
# now would make an advisory lock more disruptive than the duplicate work it
# exists to prevent. Each must warn and hand back an unheld lock instead.
#
# These stop at `acquire()` rather than running a whole sync because two of the
# shapes (a read-only index directory, a db_path that is a regular file) leave
# nowhere for LanceDB to write either; `tests/test_cli.py` carries the end-to-end
# proof that a real `sync` completes and exits 0 with an unusable lock file.


def _chmod_000_lock_file(db_path: Path) -> None:
    db_path.mkdir(parents=True)
    lock_file = db_path / LOCK_FILENAME
    lock_file.write_text("")
    lock_file.chmod(0o000)


def _read_only_lock_file(db_path: Path) -> None:
    db_path.mkdir(parents=True)
    lock_file = db_path / LOCK_FILENAME
    lock_file.write_text("")
    lock_file.chmod(0o444)  # O_RDWR is refused; the lock needs to stamp the holder


def _lock_path_is_a_directory(db_path: Path) -> None:
    (db_path / LOCK_FILENAME).mkdir(parents=True)


def _lock_path_is_a_dangling_symlink(db_path: Path) -> None:
    db_path.mkdir(parents=True)
    (db_path / LOCK_FILENAME).symlink_to(db_path / "gone" / "nowhere")


def _read_only_db_dir(db_path: Path) -> None:
    db_path.mkdir(parents=True)
    db_path.chmod(0o555)  # no lock file yet, and no way to create one


def _db_path_is_a_regular_file(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True)
    db_path.write_text("not a directory")  # mkdir(parents=True) cannot make this one


def _restore_permissions(root: Path) -> None:
    """Undo the chmods, so pytest can clear the tmp dir and the next test can read it."""
    for p in [root, *root.rglob("*")]:
        try:
            p.chmod(0o755 if p.is_dir() else 0o644)
        except OSError:  # a dangling symlink, and anything else already unreachable
            pass


@pytest.fixture
def not_root():
    if os.geteuid() == 0:  # pragma: no cover - the suite is not meant to run as root
        pytest.skip("running as root: chmod denies nothing")


@pytest.mark.parametrize(
    ("shape", "expected_errno"),
    [
        (_chmod_000_lock_file, errno.EACCES),
        (_read_only_lock_file, errno.EACCES),
        (_lock_path_is_a_directory, errno.EISDIR),
        (_lock_path_is_a_dangling_symlink, errno.ENOENT),
        (_read_only_db_dir, errno.EACCES),
        (_db_path_is_a_regular_file, errno.EEXIST),
    ],
    ids=["chmod-000", "read-only-file", "directory", "dangling-symlink", "read-only-dir", "file"],
)
def test_a_lock_that_cannot_be_established_warns_instead_of_raising(
    tmp_path, not_root, shape, expected_errno
):
    """Regression: these escaped acquire() as raw OSErrors and killed the sync."""
    db_path = tmp_path / ".minirag" / "lancedb"
    shape(db_path)
    try:
        lock = SyncLock(db_path)
        with pytest.warns(RuntimeWarning) as recorded:
            lock.acquire()  # the regression: PermissionError, IsADirectoryError, ...

        assert not lock.held
        lock.release()  # a no-op, and it must stay quiet about it
        assert len(recorded) == 1
        message = str(recorded[0].message)
        assert str(db_path / LOCK_FILENAME) in message, message
        assert os.strerror(expected_errno) in message, message
        assert "unlocked" in message, message
    finally:
        _restore_permissions(tmp_path)


def test_a_filesystem_that_cannot_flock_at_all_is_not_reported_as_contention(db_path, monkeypatch):
    """ENOTSUP from flock means nobody holds it — naming a rival would be a lie."""

    def unsupported(fd, operation):
        raise OSError(errno.ENOTSUP, os.strerror(errno.ENOTSUP))

    monkeypatch.setattr("fcntl.flock", unsupported)
    lock = SyncLock(db_path)
    with pytest.warns(RuntimeWarning, match="Could not take the sync lock"):
        lock.acquire()
    assert not lock.held


def test_real_contention_is_still_refused_and_never_degrades(held):
    """The one failure that must stay fatal: someone genuinely holds it."""
    with pytest.raises(SyncLockBusy):
        SyncLock(held.path.parent).acquire()


# --- the lock must not touch anything that is not a sync -------------------------


@pytest.fixture
def env(tmp_path, fake_embedder):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    store = Store(cfg.db_path, dim=fake_embedder.dim)
    return cfg, store, Pipeline(store, fake_embedder, cfg), tmp_path


def test_ingest_file_works_while_a_sync_holds_the_lock(env):
    """A single-file ingest is not a sync and must never wait on one.

    Holding the lock in *this* process is a strict enough check: a second flock
    from the same process conflicts exactly as another process's would, so an
    `ingest_file` that wrongly took the lock would fail this test.
    """
    _, store, pipeline, root = env
    doc = root / "a.md"
    doc.write_text("# A\n\nAlpha body text for the corpus.")

    with sync_lock(store.db_path):
        result = pipeline.ingest_file(doc)

    assert result.chunk_count > 0
    assert store.get_source(str(doc)) is not None


def test_query_works_while_a_sync_holds_the_lock(env, fake_embedder):
    _, store, pipeline, root = env
    doc = root / "a.md"
    doc.write_text("# Alpha\n\nAlpha body text for the corpus.")
    pipeline.ingest_file(doc)

    with sync_lock(store.db_path):
        results = store.search("alpha", fake_embedder.embed_query("alpha"), top_k=5)

    assert results and results[0].source == str(doc)


def test_the_lock_file_does_not_disturb_the_index(env, fake_embedder):
    """`.sync.lock` lives inside the LanceDB directory; LanceDB must not mind."""
    cfg, store, pipeline, root = env
    doc = root / "a.md"
    doc.write_text("# A\n\nAlpha body text for the corpus.")
    with sync_lock(store.db_path):
        pipeline.ingest_file(doc)

    reopened = Store(cfg.db_path, dim=fake_embedder.dim)
    assert reopened.chunk_count() > 0
    assert reopened.source_count() == 1
