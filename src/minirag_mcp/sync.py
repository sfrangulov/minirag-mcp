"""Sync engine + single-job background manager."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from minirag_mcp.config import Config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.ingest.scanner import compute_diff, scan_roots
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
        "failed": 0,
    }
    errors: list[dict] = []

    for path in diff.oversized:
        counts["failed"] += 1
        errors.append({"source": str(path), "error": f"exceeds MAX_FILE_SIZE ({max_file_size})"})

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
        with self._lock:
            if self._job is not None and self._job.state in ("pending", "running"):
                raise SyncBusyError("A sync job is already running")
            job = SyncJob(job_id=uuid.uuid4().hex)
            self._job = job  # newer job replaces any finished record

        def work() -> None:
            job.state = "running"
            try:
                counts, errors = run_sync(
                    self._pipeline, self._store, self._config.roots,
                    self._config.max_file_size, scope=scope,
                )
                job.counts, job.errors = counts, errors
                job.state = "succeeded"
            except Exception as e:  # catastrophic failure (scan error, DB down, ...)
                job.error = str(e)
                job.state = "failed"
            finally:
                job.finished_at = _now()

        self._thread = threading.Thread(target=work, name="minirag-sync", daemon=True)
        self._thread.start()
        return job.job_id

    def status(self, job_id: str) -> SyncJob:
        job = self._job
        if job is None or job.job_id != job_id:
            raise KeyError(f"Unknown sync job {job_id!r} (only the latest job is retained)")
        return job

    def wait(self, timeout: float = 30.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
