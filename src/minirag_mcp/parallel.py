"""Sync workers: how many, how many threads each may use, and the processes they run in."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import re
import subprocess
import sys
from collections import deque
from collections.abc import Callable, Container, Generator, Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

from minirag_mcp.endpoints import EndpointError

# Nothing heavier at the top: every worker process imports this module first, so fastembed
# and the pipeline are imported where they are used.
if TYPE_CHECKING:
    from minirag_mcp.config import Config
    from minirag_mcp.ingest.pipeline import Pipeline, PreparedDocument
    from minirag_mcp.store import ChunkRecord, Store

FILES_PER_WORKER = 4
# Interpreter and libraries per worker process, measured.
WORKER_BASE_BYTES = 3 * 2**29
# ~65 MB of 1024-dim vectors held as Python floats; ~200 commits per 400k rows.
BATCH_ROWS = 2000

# (source, error); error is None when the source was stored.
Written = tuple[str, str | None]


def _logical_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _linux_physical_cores(cpuinfo: str) -> int | None:
    cores: set[tuple[str, str]] = set()
    for block in re.split(r"\n\s*\n", cpuinfo):
        phys = re.search(r"^physical id\s*:\s*(\S+)", block, re.MULTILINE)
        core = re.search(r"^core id\s*:\s*(\S+)", block, re.MULTILINE)
        if phys and core:
            cores.add((phys.group(1), core.group(1)))
    return len(cores) or None


def physical_cores() -> int:
    logical = _logical_cpus()
    found: int | None = None
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout
            found = int(out.strip())
        elif sys.platform.startswith("linux"):
            found = _linux_physical_cores(Path("/proc/cpuinfo").read_text())
    except (OSError, ValueError, subprocess.SubprocessError):
        found = None
    return max(1, min(found or logical, logical))


def total_memory() -> int | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def worker_footprint(embedder) -> int:
    from fastembed import TextEmbedding

    if getattr(embedder, "backend", "") == "endpoint":
        return WORKER_BASE_BYTES
    for entry in TextEmbedding.list_supported_models():
        if entry["model"] == getattr(embedder, "model_name", None):
            return WORKER_BASE_BYTES + int(entry["size_in_GB"] * 10**9)
    return WORKER_BASE_BYTES


def default_workers(cores: int, memory: int | None, footprint: int) -> int:
    workers = max(1, cores - 1)
    if memory is not None:
        workers = min(workers, max(1, memory // 2 // footprint))
    return workers


def worker_count(configured: int | None, files: int, *, default: int) -> int:
    return max(1, min(configured or default, files // FILES_PER_WORKER))


def worker_threads(workers: int) -> int:
    return max(1, physical_cores() // workers)


WORKER_DIED = (
    "the worker process reading this file stopped abruptly (a crash in a native "
    "library, or killed for memory); it was not indexed"
)


class Preparer(Protocol):
    def prepare_file(self, path: Path, *, rules_checked: bool) -> PreparedDocument: ...


class Prepared(NamedTuple):
    path: Path
    document: PreparedDocument | None
    error: str | None
    # Failed by the embedding endpoint (unreachable, timed out, or a bad reply): a failure
    # every file after it is likely to meet too.
    endpoint_failed: bool = False


def worker_pipeline(config: Config, embedder, threads: int | None) -> Pipeline:
    """The worker half of a sync's pipeline: no store, no labels, no store hook.

    `embedder` is the parent pipeline's, unpickled in the worker as its own copy.
    """
    from minirag_mcp.ingest.pipeline import Pipeline

    embedder.threads = threads
    # Every worker would load its own copy of the model in-process, and the worker count
    # is sized for an endpoint: with e5, N workers falling back need N x 3.3 GB.
    embedder.allow_fallback = False
    return Pipeline(None, embedder, config)


def prepare_one(preparer: Preparer, path: Path) -> Prepared:
    """`path` prepared as a sync's worker prepares it; a failure is its error."""
    try:
        return Prepared(path, preparer.prepare_file(path, rules_checked=True), None)
    except EndpointError as e:
        return Prepared(path, None, str(e), endpoint_failed=True)
    except Exception as e:
        return Prepared(path, None, str(e))


# One worker's own: what it prepares with, or why it has nothing to prepare with.
_preparer: Preparer | None = None
_unbuilt: str | None = None


def _init_worker(make_preparer: bytes) -> None:
    global _preparer, _unbuilt
    # A stdio MCP server's stdout is its protocol channel, and this process inherited it.
    os.dup2(2, 1)
    # Tesseract's OpenMP threads would otherwise multiply with the workers.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    # Kept, never raised: an initializer that raises breaks the pool, and every file would
    # read as a crash. Unpickled here, inside the try, for the same reason.
    try:
        _preparer = pickle.loads(make_preparer)()
    except Exception as e:
        _unbuilt = str(e)


def _prepare(path: Path) -> Prepared:
    if _unbuilt is not None:
        return Prepared(path, None, _unbuilt)
    return prepare_one(_preparer, path)


def _run(
    make_preparer: bytes, waiting: deque[Path], workers: int, in_flight: int
) -> Generator[Prepared, None, list[Path]]:
    """Prepare paths from the left of `waiting` in one pool, in order. Returns the paths
    the pool had in flight when it broke, else []."""
    pool = ProcessPoolExecutor(
        workers,
        # Spawn, never fork: the parent may be the MCP server with LanceDB and sync threads
        # running, and a forked child inherits their locks in whatever state they are in.
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(make_preparer,),
    )
    running: deque[tuple[Path, Future[Prepared]]] = deque()
    try:
        while waiting or running:
            while waiting and len(running) < in_flight:
                running.append((waiting[0], pool.submit(_prepare, waiting[0])))
                waiting.popleft()
            result = running[0][1].result()
            running.popleft()
            yield result
    except BrokenProcessPool:
        return [path for path, _ in running]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return []


def prepare_files(
    make_preparer: Callable[[], Preparer], paths: Sequence[Path], *, workers: int
) -> Iterator[Prepared]:
    """Each of `paths` prepared in one of `workers` processes, in input order.

    A worker that dies takes its pool down with the files the pool had in flight. Those are
    prepared again one at a time in a one-worker pool, replaced whenever a file breaks it,
    so only a file that kills its worker fails, with WORKER_DIED; a fresh pool then takes
    the rest.
    """
    factory = pickle.dumps(make_preparer)
    waiting = deque(paths)
    while waiting:
        suspects = deque((yield from _run(factory, waiting, workers, 2 * workers)))
        while suspects:
            for path in (yield from _run(factory, suspects, 1, 1)):
                yield Prepared(path, None, WORKER_DIED)


class BatchWriter:
    """Prepared documents held until `batch_rows` chunks are waiting, then written in one
    `Store.replace_sources` call.

    A batch the store refuses is written one source at a time with `replace_source`, so one
    source's failure is that source's alone. The records keep their stamps, which makes the
    retry safe over a batch that failed half-way: a source whose merge had committed is
    matched row for row and updated, and its document row and signature are written again.
    An IndexBusyError is never a file's failure and propagates.
    """

    def __init__(self, store: Store, *, indexed: Container[str], batch_rows: int | None = None):
        self._store = store
        self._indexed = indexed
        self._batch_rows = BATCH_ROWS if batch_rows is None else batch_rows
        self._batch: list[tuple[str, Sequence[ChunkRecord]]] = []
        self._rows = 0

    def add(self, source: str, records: Sequence[ChunkRecord]) -> list[Written]:
        self._batch.append((source, records))
        self._rows += len(records)
        return self.flush() if self._rows >= self._batch_rows else []

    def flush(self) -> list[Written]:
        batch, self._batch, self._rows = self._batch, [], 0
        if not batch:
            return []
        from minirag_mcp.store import IndexBusyError

        try:
            self._store.replace_sources(batch, indexed=self._indexed)
        except IndexBusyError:
            raise
        except Exception:
            return [self._write_one(source, records) for source, records in batch]
        return [(source, None) for source, _ in batch]

    def _write_one(self, source: str, records: Sequence[ChunkRecord]) -> Written:
        from minirag_mcp.store import IndexBusyError

        try:
            self._store.replace_source(source, records)
        except IndexBusyError:
            raise
        except Exception as e:
            return source, str(e)
        return source, None
