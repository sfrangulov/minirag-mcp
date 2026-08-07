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
    entries: list[ScanEntry] = []
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
        to_ingest=to_ingest, to_delete=sorted(to_delete),
        unchanged=unchanged, oversized=oversized,
    )


@dataclass(frozen=True)
class FileState:
    source: str
    source_type: str
    title: str
    state: str  # "ingested" | "not_ingested" | "stale"
    chunk_count: int


def compute_states(entries: list[ScanEntry], indexed: list[SourceInfo]) -> list[FileState]:
    by_source = {s.source: s for s in indexed}
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
            states.append(
                FileState(str(e.path), "file", prior.title, "stale", prior.chunk_count)
            )
    for s in indexed:
        if s.source_type in ("data", "url"):
            states.append(FileState(s.source, s.source_type, s.title, "ingested", s.chunk_count))
    states.sort(key=lambda s: s.source)
    return states
