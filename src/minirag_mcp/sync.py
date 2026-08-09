"""Sync engine + single-job background manager."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from minirag_mcp import ocr
from minirag_mcp.config import Config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.ingest.scanner import compute_diff, scan_roots
from minirag_mcp.lock import SyncLock, sync_lock
from minirag_mcp.store import Store


class SyncBusyError(Exception):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class SyncJob:
    job_id: str
    state: str = "pending"  # pending | running | succeeded | failed
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "scanned": 0,
            "ingested": 0,
            "skipped": 0,
            "deleted": 0,
            # Kept in step with the dict `_run_sync_unlocked` builds: a counter that
            # only appears once the job finishes makes every earlier poll report a
            # different shape.
            "unreadable": 0,
            "failed": 0,
        }
    )
    errors: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "jobId": self.job_id,
            "state": self.state,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "counts": dict(self.counts),
            "errors": list(self.errors),
            "error": self.error,
        }


def run_sync(
    pipeline: Pipeline,
    store: Store,
    roots: Sequence[Path],
    max_file_size: int,
    scope: Path | None = None,
    on_event: Callable[[str], None] | None = None,
) -> tuple[dict[str, int], list[dict]]:
    """Scan `roots`, diff against the store, ingest/delete accordingly.

    Holds the cross-process sync lock (see `minirag_mcp.lock`) for the whole
    run and raises `SyncLockBusy` if another process is already syncing this
    index. `SyncManager` does not come through here — it takes the same lock
    itself, earlier, and calls `_run_sync_unlocked`.

    `scope`, if given, is resolved to its canonical form (`Path.resolve()`)
    before use, matching `Config.roots` (which are always resolved). Without
    this, a non-canonical scope (e.g. an unresolved `/tmp` on macOS, where
    the real path is `/private/tmp`) silently matches zero files: the diff
    comes back empty, the job "succeeds" with all-zero counts, and no error
    is ever raised.
    """
    with sync_lock(store.db_path):
        return _run_sync_unlocked(pipeline, store, roots, max_file_size, scope, on_event)


def _run_sync_unlocked(
    pipeline: Pipeline,
    store: Store,
    roots: Sequence[Path],
    max_file_size: int,
    scope: Path | None,
    on_event: Callable[[str], None] | None,
) -> tuple[dict[str, int], list[dict]]:
    """The body of `run_sync`. The caller must already hold the sync lock."""
    scope = scope.resolve() if scope is not None else None

    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    entries = scan_roots(list(roots))
    diff = compute_diff(entries, store.list_sources(), max_file_size=max_file_size, scope=scope)
    counts = {
        "scanned": len(diff.to_ingest) + len(diff.unchanged) + len(diff.oversized),
        "ingested": 0,
        "skipped": len(diff.unchanged),
        "deleted": 0,
        "unreadable": len(diff.unreadable),
        "failed": 0,
    }
    errors: list[dict] = []

    for path in diff.oversized:
        counts["failed"] += 1
        errors.append({"source": str(path), "error": f"exceeds MAX_FILE_SIZE ({max_file_size})"})

    # Not counted as failures: the run did everything it could, and an install that
    # deliberately omits the extra would otherwise exit 1 on every sync forever.
    for source in diff.unreadable:
        errors.append(
            {
                "source": source,
                "error": "kept in the index: this installation cannot read this file type, "
                f"so it was neither re-ingested nor deleted; {ocr.INSTALL_HINT}",
            }
        )

    for path in diff.to_ingest:
        try:
            pipeline.ingest_file(path)
            counts["ingested"] += 1
            emit(f"ingested {path}")
        except Exception as e:  # per-file failures never abort the job
            counts["failed"] += 1
            errors.append({"source": str(path), "error": str(e)})
            emit(f"failed {path}: {e}")

    for source in diff.to_delete:
        store.delete_source(source)
        counts["deleted"] += 1
        emit(f"deleted {source}")

    return counts, errors


class SyncManager:
    def __init__(self, pipeline: Pipeline, store: Store, config: Config):
        self._pipeline = pipeline
        self._store = store
        self._config = config
        self._lock = threading.Lock()
        self._job: SyncJob | None = None
        self._thread: threading.Thread | None = None

    def start(self, scope: Path | None = None) -> str:
        """Start a background sync, or raise if one is already running.

        `SyncBusyError` if this process is already running one, `SyncLockBusy`
        if a different process is.
        """
        lock = SyncLock(self._store.db_path)
        with self._lock:
            # Order matters twice over. The in-process guard runs first, so a second
            # job started while this one is still non-terminal is rejected as
            # SyncBusyError and never reaches the flock — which is what keeps us from
            # deadlocking against ourselves, since a same-process second acquire
            # conflicts like any other rival. That only holds because the worker
            # releases the flock *before* it publishes the terminal state (see
            # `work` below): the guard covers the window in which we hold the lock,
            # and stops covering it only once the lock is gone. And the flock is
            # taken before any state is recorded, so a refusal leaves the manager
            # exactly as it found it rather than stranding a 'pending' job that would
            # wedge every later start().
            if self._job is not None and self._job.state in ("pending", "running"):
                raise SyncBusyError("A sync job is already running")
            # Acquired here, in the caller's thread, rather than inside the worker:
            # acquiring in the worker would hand the caller a jobId for a job that is
            # already doomed, and since only the latest job's record is kept, that
            # failure is easy to miss entirely. Here, contention is an immediate
            # error on sync_start — a ToolError the client can act on.
            lock.acquire()
            job = SyncJob(job_id=uuid.uuid4().hex)
            self._job = job  # newer job replaces any finished record

        def work() -> None:
            job.state = "running"
            # Two orderings, both on the way out, and both on the same principle:
            # the terminal state is published last, because it is the signal the
            # client acts on.
            #
            # finished_at first, so a poller never observes a terminal state with
            # finished_at still None.
            #
            # Then the flock — before the terminal state, not after. A client that
            # follows the documented pattern (start, poll sync_status until
            # terminal, start again) would otherwise clear the in-process guard,
            # because the job reads 'succeeded', and then collide with the flock
            # this very process is still holding: refused with a message naming its
            # own PID. Released first, the same early restart meets only the
            # in-process guard, which still reads 'running' — a SyncBusyError, which
            # is the right answer to "a sync is in flight" and is already handled.
            #
            # The whole exit runs from `finally`, so it also covers what no
            # `except Exception` sees — a KeyboardInterrupt, say. Such a job is
            # reported failed rather than left reading 'running', which would hold
            # the in-process guard shut for the lifetime of the process.
            terminal = "failed"
            try:
                counts, errors = _run_sync_unlocked(
                    self._pipeline,
                    self._store,
                    self._config.roots,
                    self._config.max_file_size,
                    scope,
                    None,
                )
            except Exception as e:  # catastrophic failure (scan error, DB down, ...)
                job.error = str(e)
            else:
                job.counts, job.errors = counts, errors
                terminal = "succeeded"
            finally:
                job.finished_at = _now()
                try:
                    # Released on the worker thread, not the one that acquired it.
                    # flock belongs to the open file description, so the handoff is
                    # legitimate.
                    lock.release()
                finally:
                    # Even if releasing somehow blew up: a job that never reaches a
                    # terminal state wedges every later start() in this process.
                    job.state = terminal

        try:
            self._thread = threading.Thread(target=work, name="minirag-sync", daemon=True)
            self._thread.start()
        except Exception as e:  # the OS refused a new thread
            # Nobody will ever run `work`, so nobody will ever release the lock or
            # move the job off 'pending'. Undo both here, or this process holds the
            # lock until it exits and refuses every later sync.
            lock.release()
            job.error = f"could not start the sync worker thread: {e}"
            job.finished_at = _now()
            job.state = "failed"
            raise
        return job.job_id

    def status(self, job_id: str) -> SyncJob:
        job = self._job
        if job is None or job.job_id != job_id:
            raise KeyError(f"Unknown sync job {job_id!r} (only the latest job is retained)")
        return job

    def wait(self, timeout: float = 30.0) -> None:
        """Block until the current worker thread finishes (test/CLI helper).

        Reads `self._thread` without synchronization, so this is only safe
        when called after `start()` has returned in the same calling thread
        (the ordinary test/CLI usage: `start()` then `wait()`). It is not a
        general-purpose "block until done" API for arbitrary concurrent
        callers — a `wait()` racing a concurrent `start()` may see a stale
        or `None` `self._thread` and return without waiting for the new job.
        """
        if self._thread is not None:
            self._thread.join(timeout)
