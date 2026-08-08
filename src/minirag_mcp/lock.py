"""Cross-process advisory lock, held for the duration of a sync.

Scope: sync, and only sync. This is not a mutex over the index. Concurrent
writes to one LanceDB path are safe — LanceDB commits optimistically and
retries, and measurements against this Store showed no loss and no partial
state under parallel writers. What two simultaneous syncs cost is wasted work
(both walk and re-index the same corpus), extra windows where a source is
briefly absent mid-replace, and a user with no way to tell that it is
happening. So single-file ingests and every read path stay unlocked; only the
whole-corpus sync takes this.

The lock is an exclusive, non-blocking `flock` on `<db_path>/.sync.lock`. While
held, the file contains the holder's PID and start time as JSON, so a contender
can say *who* is syncing and for how long.

There is no stale-lock recovery path, because there is nothing to recover from:
`flock` is owned by the open file description and the kernel drops it when the
process dies, however it dies. That is a claim worth distrusting, so it is
tested rather than assumed — `tests/test_lock.py` kills a holder with SIGKILL
and takes the lock immediately afterwards. Note that the lock *file* does
outlive the holder, keeping whatever bytes were last written to it; the holder
info in it is therefore never treated as evidence that a lock is held, only as
a description of a holder we already know exists because our own acquire failed.

Known limitations:

- **Network filesystems.** `flock` over NFS or SMB is unreliable — depending on
  the client, server, and mount options it may be emulated via POSIX locks, be
  silently local to each client, or not work at all. Two machines syncing one
  `DB_PATH` on a share can therefore both win the lock. Local disks only.
- **Advisory.** Nothing stops a process that does not ask for the lock from
  writing to the index. That is deliberate: it is what keeps `ingest_file` and
  queries unblocked.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX-only; the tests all run on POSIX
    # Windows has no fcntl. The project makes no Windows support claim, but an
    # ImportError at import time would break `import minirag_mcp.server` there and
    # everywhere else that transitively imports this, turning "sync locking is
    # unavailable" into "the package does not load". Degrade to an unlocked no-op
    # with a warning instead: the worst case is the pre-existing behaviour, two
    # syncs running at once, which is wasteful and not corrupting.
    fcntl = None

LOCK_FILENAME = ".sync.lock"


class SyncLockBusy(Exception):
    """Another process holds the sync lock for this index.

    `pid` and `started_at` describe the holder when it could be read from the
    lock file, and are None when it could not (see `_read_holder`). The message
    is already user-facing — the CLI prints it, the MCP server returns it as a
    ToolError.
    """

    def __init__(
        self,
        message: str,
        *,
        pid: int | None = None,
        started_at: str | None = None,
        path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.pid = pid
        self.started_at = started_at
        self.path = path


def _read_holder(path: Path) -> tuple[int | None, str | None]:
    """Best-effort (pid, startedAt) of the holder. Never raises.

    Every failure collapses to (None, None) and a vaguer message. The file can
    legitimately be empty or half-written: a holder that died between creating
    the file and stamping it leaves exactly that, as does one we are contending
    with in the microseconds before it writes. Refusing a sync is already the
    bad news; crashing while explaining why would be worse.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None, None
    if not raw:
        return None, None
    try:
        data = json.loads(raw)
        pid = data["pid"]
        started_at = data.get("startedAt")
    except (ValueError, TypeError, KeyError, AttributeError):
        return None, None
    try:
        pid = int(pid)
    except (ValueError, TypeError):
        return None, None
    return pid, started_at if isinstance(started_at, str) else None


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _elapsed(started_at: str | None) -> str | None:
    """How long ago `started_at` was, or None if it is missing or unusable."""
    if not started_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
    except (ValueError, TypeError):
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    seconds = (datetime.now(UTC) - started).total_seconds()
    if seconds < 0:  # clock skew, or a stamp from the future — say nothing
        return None
    return _format_duration(seconds)


def _busy_message(path: Path, pid: int | None, started_at: str | None) -> str:
    tail = "Wait for it to finish, or use a different DB_PATH."
    if pid is None:
        return f"Another minirag-mcp process is syncing this index (lock: {path}). {tail}"
    elapsed = _elapsed(started_at)
    doing = f"has been syncing this index for {elapsed}" if elapsed else "is syncing this index"
    return f"Another minirag-mcp process (PID {pid}) {doing}. {tail}"


class SyncLock:
    """The sync lock as an explicit acquire/release pair.

    Prefer the `sync_lock` context manager. This class exists for the one caller
    that cannot use it: `SyncManager` acquires in `start()`, on the caller's
    thread, and releases in the worker thread when the job ends. That straddles
    a thread boundary, which is safe here — `flock` belongs to the open file
    description, not to the thread that called it.

    One instance holds at most one lock. Acquiring twice from the same process
    would *not* be reentrant: a second `os.open` makes a second open file
    description, and `flock` treats it as a rival, so a nested acquire would
    deadlock the process against itself (verified: errno 35 on macOS). The
    double-acquire guard below turns that into a loud programming error.
    """

    def __init__(self, db_path: Path) -> None:
        self.path = Path(db_path) / LOCK_FILENAME
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self) -> None:
        """Take the lock, or raise SyncLockBusy naming whoever has it."""
        if self._fd is not None:
            raise RuntimeError(f"This SyncLock already holds {self.path}")
        if fcntl is None:
            warnings.warn(
                f"fcntl is unavailable on this platform, so {self.path} is not being taken: "
                "two syncs against this index can run at once.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            pid, started_at = _read_holder(self.path)
            raise SyncLockBusy(
                _busy_message(self.path, pid, started_at),
                pid=pid,
                started_at=started_at,
                path=self.path,
            ) from e
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            payload = {"pid": os.getpid(), "startedAt": datetime.now(UTC).isoformat()}
            os.write(fd, json.dumps(payload).encode("utf-8"))
        except OSError:
            # The lock is held; only the description of who holds it was lost. A
            # contender degrades to the vaguer message, which beats failing a sync
            # that is otherwise perfectly able to run.
            pass
        self._fd = fd

    def release(self) -> None:
        """Release the lock. Idempotent, and safe to call from another thread."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            # Leave no holder info behind: the file outlives us, and stale contents
            # would describe a process that is no longer syncing anything.
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass  # closing the fd releases the lock regardless
        finally:
            os.close(fd)


@contextmanager
def sync_lock(db_path: Path) -> Iterator[SyncLock]:
    """Hold the sync lock for `db_path` for the duration of the block.

    Raises SyncLockBusy immediately — never blocks — if another process is
    already syncing this index.
    """
    lock = SyncLock(db_path)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
