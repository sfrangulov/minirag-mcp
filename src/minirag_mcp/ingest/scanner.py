"""Recursive document-root scanning and sync diff computation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from minirag_mcp.ingest.parser import SUPPORTED_EXTENSIONS
from minirag_mcp.ingest.pipeline import file_sha256
from minirag_mcp.store import SourceInfo

SKIP_DIRS = frozenset({"node_modules", "__pycache__", ".venv", "venv"})


@dataclass(frozen=True)
class ScanEntry:
    path: Path
    size: int
    mtime: float


@dataclass(frozen=True)
class SyncDiff:
    to_ingest: list[Path]
    to_delete: list[str]
    unchanged: list[Path]
    oversized: list[Path]


def scan_roots(roots: Sequence[Path]) -> list[ScanEntry]:
    """Recursively scan roots for files with whitelisted extensions.

    Skips hidden directories (dot-prefixed) and SKIP_DIRS. Prunes directory traversal
    for SKIP_DIRS and hidden dirs, but includes symlinked files with whitelisted extensions.
    Symlinked directories are not traversed (os.walk followlinks=False avoids cycles).
    Overlapping roots are tolerated: each file is reported once (deduplicated by path).

    Returns entries sorted by path.
    """
    entries: list[ScanEntry] = []
    seen: set[Path] = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS
            )
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                p = Path(dirpath) / name
                if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                # Dedupe by resolved path to handle overlapping roots
                resolved = p.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                st = p.stat()
                entries.append(ScanEntry(path=p, size=st.st_size, mtime=st.st_mtime))
    entries.sort(key=lambda e: e.path)
    return entries


def _in_scope(path_str: str, scope: Path | None) -> bool:
    if scope is None:
        return True
    p = Path(path_str)
    return p == scope or scope in p.parents


def compute_diff(
    entries: list[ScanEntry],
    indexed: list[SourceInfo],
    *,
    max_file_size: int,
    scope: Path | None = None,
) -> SyncDiff:
    """Compute sync diff: files to ingest, delete, and unchanged status.

    Uses mtime fast-path: if mtime matches, file is considered unchanged without hashing.
    This is a deliberate trade-off — equal mtime is trusted, so an mtime-preserving restore
    (cp -p, rsync -a, git checkout) may leave a stale index entry. Re-sync will detect and
    update it only if mtime differs (then hash is compared).

    Args:
        entries: Scanned disk files.
        indexed: Files currently in the index (SourceInfo from Store).
        max_file_size: Files larger than this are marked oversized (not ingested).
        scope: If provided, limit processing to files under this path.

    Returns:
        SyncDiff with to_ingest, to_delete, unchanged, and oversized lists.
    """
    indexed_files = {
        s.source: s for s in indexed if s.source_type == "file" and _in_scope(s.source, scope)
    }
    to_ingest: list[Path] = []
    unchanged: list[Path] = []
    oversized: list[Path] = []
    seen: set[str] = set()

    for e in entries:
        if not _in_scope(str(e.path), scope):
            continue
        seen.add(str(e.path))
        if e.size > max_file_size:
            oversized.append(e.path)
            continue
        prior = indexed_files.get(str(e.path))
        if prior is None:
            to_ingest.append(e.path)
        elif prior.mtime == e.mtime or prior.file_hash == file_sha256(e.path):
            unchanged.append(e.path)
        else:
            to_ingest.append(e.path)

    to_delete = [src for src in indexed_files if src not in seen]
    return SyncDiff(
        to_ingest=to_ingest,
        to_delete=sorted(to_delete),
        unchanged=unchanged,
        oversized=oversized,
    )


@dataclass(frozen=True)
class FileState:
    source: str
    source_type: str
    title: str
    state: str  # "ingested" | "not_ingested" | "stale"
    chunk_count: int


def compute_states(entries: list[ScanEntry], indexed: list[SourceInfo]) -> list[FileState]:
    """Compute file states: ingested, stale, or not_ingested for disk files + data/url sources.

    States for disk files (source_type="file"):
    - "ingested": mtime matches indexed record, or content hash matches (file unchanged).
    - "stale": mtime differs AND hash differs (file changed on disk).
    - "not_ingested": file not in index.

    Uses mtime fast-path: if mtime matches, no hash is computed. Equal mtime is trusted,
    so mtime-preserving restores (cp -p, rsync -a, git checkout) may mark changed files
    as ingested. Re-sync will detect and correct if mtime changes.

    Indexed data/url sources are appended as "ingested" and never filtered.
    Results sorted by source path.
    """
    by_source = {s.source: s for s in indexed if s.source_type == "file"}
    states: list[FileState] = []
    for e in entries:
        prior = by_source.get(str(e.path))
        if prior is None:
            states.append(FileState(str(e.path), "file", "", "not_ingested", 0))
        elif prior.mtime == e.mtime or prior.file_hash == file_sha256(e.path):
            states.append(
                FileState(str(e.path), "file", prior.title, "ingested", prior.chunk_count)
            )
        else:
            states.append(FileState(str(e.path), "file", prior.title, "stale", prior.chunk_count))
    for s in indexed:
        if s.source_type in ("data", "url"):
            states.append(FileState(s.source, s.source_type, s.title, "ingested", s.chunk_count))
    states.sort(key=lambda s: s.source)
    return states
