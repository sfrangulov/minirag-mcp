"""Sync engine + single-job background manager."""

from __future__ import annotations

import contextlib
import functools
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from minirag_mcp import ocr, parallel
from minirag_mcp.identity import RebuildInfo, acquire_sync_lock
from minirag_mcp.ingest.ignore import IgnoredFileError, scope_excluded_by
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.ingest.scanner import compute_diff, scan_roots
from minirag_mcp.scope import is_under
from minirag_mcp.store import IndexBusyError, Store

# Endpoint failures in a row after which a parallel sync stops handing out files: every
# file needs the endpoint, and each failure can have waited out its whole timeout.
ENDPOINT_FAILURES_TO_STOP = 3

# Told what a rebuild rebuilds and why when it starts, before its counts are known.
OnRebuild = Callable[[RebuildInfo], None]
# One sync of the index, or of `scope` inside it: its counts, its per-file errors, and the
# rebuild it made, if any. Run with the sync lock already held.
SyncRun = Callable[[Path | None, OnRebuild], tuple[dict[str, int], list[dict], RebuildInfo | None]]


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
            "moved": 0,
            "deleted": 0,
            # Kept in step with the dict `_run_sync_unlocked` builds: a counter that
            # only appears once the job finishes makes every earlier poll report a
            # different shape. For the same reason `relabelled` is here whether or not
            # the corpus has a classify.toml.
            "relabelled": 0,
            "signed": 0,
            "ignored": 0,
            "unreadable": 0,
            "largeTables": 0,
            "failed": 0,
        }
    )
    errors: list[dict] = field(default_factory=list)
    error: str | None = None
    # Set when a rebuild starts, its counts None until the job has finished.
    rebuild: RebuildInfo | None = None

    def to_dict(self) -> dict:
        return {
            "jobId": self.job_id,
            "state": self.state,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "counts": dict(self.counts),
            "errors": list(self.errors),
            "error": self.error,
            "rebuild": self.rebuild.to_dict() if self.rebuild else None,
        }


def _ingest_in_workers(
    pipeline: Pipeline,
    store: Store,
    paths: Sequence[Path],
    workers: int,
    indexed: set[str],
    ingested: Callable[[str], None],
    failed: Callable[[str, str], None],
) -> int:
    """Prepare `paths` in `workers` processes and store them from this one, in batches,
    labelled in the caller's labelling job. How many were never prepared because the
    embedding endpoint stopped answering.

    `indexed` gains each source as it is stored, so that a later batch holding it again
    replaces its chunks instead of appending a second copy.
    """
    config = pipeline.config
    # Once, before any worker starts: the batches are written with `replace_sources`, which
    # this hook does not precede. On a fresh index it records the probe and arms the
    # embedder, and each worker's copy of the embedder carries that.
    if pipeline.before_store is not None:
        pipeline.before_store()
    if ocr.available():
        # Fetched here so that the workers do not race each other to download it. Each file
        # that needs it then fails with its own message.
        with contextlib.suppress(ocr.OcrError):
            ocr.ensure_tessdata(config.cache_dir, config.ocr_lang)
    make = functools.partial(
        parallel.worker_pipeline, config, pipeline.embedder, parallel.worker_threads(workers)
    )
    writer = parallel.BatchWriter(store, indexed=indexed)
    # The sources added since the writer last reported: those a refused write drops.
    held: list[str] = []

    def write(call: Callable[..., list[parallel.Written]], *args) -> None:
        try:
            results = call(*args)
        except IndexBusyError as e:
            for source in held:
                failed(source, str(e))
            held.clear()
            return
        if results:
            held.clear()
        for source, error in results:
            if error is None:
                indexed.add(source)
                ingested(source)
            else:
                failed(source, error)

    read = in_a_row = not_read = 0
    with contextlib.closing(parallel.prepare_files(make, paths, workers=workers)) as prepared:
        for result in prepared:
            read += 1
            in_a_row = in_a_row + 1 if result.endpoint_failed else 0
            if result.error is not None:
                failed(str(result.path), result.error)
                if in_a_row >= ENDPOINT_FAILURES_TO_STOP:
                    not_read = len(paths) - read
                    break
                continue
            try:
                records, _ = pipeline.records(result.document)
            except Exception as e:
                failed(str(result.path), str(e))
                continue
            held.append(result.document.source)
            write(writer.add, result.document.source, records)
    write(writer.flush)
    return not_read


def _run_sync_unlocked(
    pipeline: Pipeline,
    store: Store,
    roots: Sequence[Path],
    max_file_size: int,
    scope: Path | None,
    on_event: Callable[[str], None] | None,
    large_tables: list[Path] | None = None,
    *,
    exclude: Sequence[str],
) -> tuple[dict[str, int], list[dict]]:
    """Scan `roots`, diff against the store, ingest and delete accordingly. The caller holds
    the sync lock (`acquire_sync_lock`) and has checked the index is not `rebuild pending`:
    `IndexHandle.run_sync` and the rebuild do both.

    When `large_tables` is given, it collects the tables left out for RAG_MAX_TABLE_ROWS.

    `scope`, if given, is resolved to its canonical form before use, matching
    `Config.roots` (which are always resolved). Without this, a non-canonical scope (an
    unresolved `/tmp` on macOS, whose real path is `/private/tmp`) silently matches zero
    files: the diff comes back empty and the job "succeeds" with all-zero counts.
    """
    scope = scope.resolve() if scope is not None else None
    if scope is not None and (by := scope_excluded_by(scope, roots, exclude)) is not None:
        # A silent no-op otherwise: every count zero, exit 0, nothing synced.
        raise IgnoredFileError(f"{scope} is excluded by {by}; nothing under it can be synced")

    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    blind: list[tuple[Path, str | None]] = []
    ignored: list[Path] = []
    entries = scan_roots(list(roots), blind=blind, ignored=ignored, exclude=exclude)
    listing = store.list_sources()
    diff = compute_diff(
        entries,
        listing,
        max_file_size=max_file_size,
        scope=scope,
        roots=roots,
        blind=blind,
        ignored=ignored,
        record_pages="pages" not in store.missing_columns,
        max_table_rows=pipeline.config.max_table_rows,
    )
    if large_tables is not None:
        large_tables.extend(diff.large_tables)
    counts = {
        "scanned": len(diff.to_ingest)
        + len(diff.moved)
        + len(diff.unchanged)
        + len(diff.oversized)
        + len(diff.kept_scans)
        + len(diff.large_tables),
        "ingested": 0,
        "skipped": len(diff.unchanged),
        "moved": 0,
        "deleted": 0,
        "relabelled": 0,
        "signed": 0,
        "ignored": sum(1 for p in ignored if scope is None or is_under(str(p), str(scope))),
        "unreadable": len(diff.unreadable) + len(diff.kept_scans),
        "largeTables": len(diff.large_tables),
        "failed": 0,
    }
    errors: list[dict] = []

    for held in diff.held:
        if held.fails:
            counts["failed"] += 1
        errors.append({"source": str(held.path), "error": held.message})

    for path in diff.oversized:
        counts["failed"] += 1
        errors.append({"source": str(path), "error": f"exceeds MAX_FILE_SIZE ({max_file_size})"})

    # Not failures: the limit is the owner's choice, made so that these stay out.
    limit = pipeline.config.max_table_rows
    for path in diff.large_tables:
        errors.append(
            {
                "source": str(path),
                "error": f"skipped: more than {limit} data rows exceeds RAG_MAX_TABLE_ROWS "
                f"({limit})",
            }
        )

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
    for path in diff.kept_scans:
        errors.append(
            {
                "source": str(path),
                "error": "kept as indexed, without page numbers: it was read with OCR, which "
                f"this installation lacks, and reading it again needs it; {ocr.INSTALL_HINT}",
            }
        )

    def failed(source: str, error: str) -> None:
        counts["failed"] += 1
        errors.append({"source": source, "error": error})
        emit(f"failed {source}: {error}")

    def ingested(source: str) -> None:
        counts["ingested"] += 1
        emit(f"ingested {source}")

    def ingest(path: Path) -> bool:
        try:
            pipeline.ingest_file(path, rules_checked=True, rows_checked=True)
        except Exception as e:  # per-file failures never abort the job
            failed(str(path), str(e))
            return False
        ingested(str(path))
        return True

    # One labelling job for the whole run, so the model is loaded at most once and the
    # re-label pass compares against the taxonomy the run's own ingests were labelled
    # with.
    with pipeline.job() as report:
        # A re-path that moved nothing (the origin was deleted meanwhile; delete_file
        # takes no lock) or raised falls back to reading the file and then dropping the
        # origin, which ends where the re-path would have. If the read fails too, both
        # stay as they were and the next sync pairs them again.
        for move in diff.moved:
            try:
                rows = store.move_source(move.source, str(move.path), mtime=move.mtime)
            except Exception as e:
                rows = 0
                emit(f"could not re-path {move.source}, reading {move.path} instead: {e}")
            if rows:
                counts["moved"] += 1
                emit(f"moved {move.source} -> {move.path}")
            elif ingest(move.path) and store.delete_source(move.source):
                counts["deleted"] += 1
                emit(f"deleted {move.source}")

        workers = parallel.worker_count(
            pipeline.config.sync_workers,
            len(diff.to_ingest),
            default=parallel.default_workers(
                parallel.physical_cores(),
                parallel.total_memory(),
                parallel.worker_footprint(pipeline.embedder),
            ),
        )
        if workers == 1:
            for path in diff.to_ingest:
                ingest(path)
        else:
            indexed = {info.source for info in listing}
            not_read = _ingest_in_workers(
                pipeline, store, diff.to_ingest, workers, indexed, ingested, failed
            )
            if not_read:
                # Not one error per file: they were never read, and the next sync reads
                # them, their hashes unchanged.
                counts["failed"] += not_read
                stopped = f"embedding: endpoint unreachable, {not_read} files not synced"
                errors.append({"source": str(store.db_path), "error": stopped})
                emit(stopped)

        for source in diff.to_delete:
            store.delete_source(source)
            counts["deleted"] += 1
            emit(f"deleted {source}")

        # Last, so nothing deleted is labelled and every moved document is: a move leaves
        # labels stale, since the new folder's path rules may give other values.
        relabelled, relabel_errors = pipeline.relabel_stale(
            (str(scope),) if scope else (), on_event=on_event
        )
        counts["relabelled"] += relabelled
        errors.extend(relabel_errors)

    # After every other write of the run, which signs what it ingests and moves. What is
    # left unsigned predates signatures or lost a write. Not failures: those documents
    # are indexed and searchable; only their look-alikes go unnamed.
    signed, sign_errors = store.backfill_signatures(
        (str(scope),) if scope else (), on_event=on_event
    )
    counts["signed"] += signed
    errors.extend(sign_errors)
    # Likewise for document rows, and before `update_indices`, which folds the rows it
    # writes into the documents index.
    _, document_errors = store.backfill_documents((str(scope),) if scope else (), on_event=on_event)
    errors.extend(document_errors)

    # Also on a run that changed nothing: an index written before this existed is
    # brought up to date by its next sync.
    if problem := store.update_indices():
        errors.append({"source": str(store.db_path), "error": problem})

    # Read once the job has ended, since releasing the model can add a problem of its
    # own. Not failures: the documents are indexed, and only their labels are missing.
    if report.problems:
        taxonomy = str(pipeline.labels.path) if pipeline.labels else ""
        errors.extend({"source": taxonomy, "error": f"labels: {p}"} for p in report.problems)

    # Not a failure: while the endpoint is unreachable, the in-process model embeds.
    if warning := getattr(pipeline.embedder, "warning", None):
        errors.append({"source": str(store.db_path), "error": f"embedding: {warning}"})

    return counts, errors


class SyncManager:
    def __init__(self, db_path: Path, run: SyncRun):
        """Runs `run` as one background job at a time, holding the sync lock of `db_path`."""
        self._db_path = db_path
        self._run = run
        self._lock = threading.Lock()
        self._job: SyncJob | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Whether this manager's latest job is pending or running."""
        job = self._job
        return job is not None and job.state in ("pending", "running")

    def start(self, scope: Path | None = None) -> str:
        """Start a background sync, or raise if one is already running.

        `SyncBusyError` if this process is already running one, `SyncLockBusy`
        if a different process is.
        """
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
            lock = acquire_sync_lock(self._db_path)
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

            def rebuilding(info: RebuildInfo) -> None:
                job.rebuild = info

            try:
                counts, errors, rebuilt = self._run(scope, rebuilding)
            except Exception as e:  # catastrophic failure (scan error, DB down, ...)
                job.error = str(e)
            else:
                job.counts, job.errors, job.rebuild = counts, errors, rebuilt
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
