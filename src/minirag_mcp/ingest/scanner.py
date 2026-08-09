"""Recursive document-root scanning and sync diff computation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.ingest.parser import supported_extensions
from minirag_mcp.ingest.pipeline import file_sha256
from minirag_mcp.scope import is_under
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


def _contained(real: Path, real_roots: Sequence[Path]) -> bool:
    """True if `real` (already resolved) sits inside any of `real_roots` (already resolved).

    Same containment rule as security.resolve_in_roots: equal to a root, or having
    that root among its parents.
    """
    return any(real == r or r in real.parents for r in real_roots)


def scan_roots(roots: Sequence[Path]) -> list[ScanEntry]:
    """Recursively scan roots for files with whitelisted extensions.

    Skips hidden directories (dot-prefixed) and SKIP_DIRS. Prunes directory traversal
    for SKIP_DIRS and hidden dirs. Symlinked directories are not traversed
    (os.walk followlinks=False avoids cycles).

    Symlinked files are included **only when their target stays inside a configured
    root** — the same containment rule security.resolve_in_roots applies. A symlink
    whose real path escapes every root is skipped silently: it is not an error, it is
    simply not part of the corpus. Without this check the extension whitelist would be
    matched against the link *name* while the parser reads the *target*, so a
    `notes.md -> ~/.ssh/id_rsa` link would pull an arbitrary readable file into the
    index. Non-escaping entries are reported by their on-disk path, not their resolved
    target, so source ids stay stable.

    Overlapping roots are tolerated: each file is reported once (deduplicated by
    resolved path).

    Returns entries sorted by path.
    """
    real_roots = [Path(r).resolve() for r in roots]
    exts = supported_extensions()
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
                if p.suffix.lower() not in exts:
                    continue
                # Dedupe by resolved path to handle overlapping roots
                resolved = p.resolve()
                if resolved in seen:
                    continue
                if not _contained(resolved, real_roots):
                    continue  # symlink escaping every root: not part of the corpus
                seen.add(resolved)
                st = p.stat()
                entries.append(ScanEntry(path=p, size=st.st_size, mtime=st.st_mtime))
    entries.sort(key=lambda e: e.path)
    return entries


def _in_scope(path_str: str, scope: Path | None) -> bool:
    """Whether `path_str` is the scope path itself or sits under it.

    Both sides go through `Path` first, so a non-canonical spelling ("/a//b") is
    compared in its normalised form; the containment rule itself is `scope.is_under`,
    the same one the store's SQL filter and `list_files` apply.
    """
    if scope is None:
        return True
    return is_under(str(Path(path_str)), str(scope))


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

    A source whose chunks predate the current chunking scheme is re-ingested whatever
    its bytes say. Its file has not changed; what it was cut into has. Without this,
    `status` would tell the user to re-sync and the re-sync would skip every file.

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
        if prior is None or prior.scheme_version < SCHEME_VERSION:
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
    state: str  # "ingested" | "not_ingested" | "stale" | "stale_scheme"
    chunk_count: int
    ocr_engine: str = ""


def _indexed_state(prior: SourceInfo) -> str:
    """Whether an indexed source is current, or was cut by an older chunking scheme.

    A scheme-stale source needs the same action as a changed one — re-ingest — and
    `compute_diff` already re-ingests it. Reporting it as plain "ingested" is what made
    the listing contradict `status`, which says the index is stale and to re-sync; an
    agent choosing what to work on from the listing would have seen nothing to do.
    """
    return "stale_scheme" if prior.scheme_version < SCHEME_VERSION else "ingested"


def compute_states(entries: list[ScanEntry], indexed: list[SourceInfo]) -> list[FileState]:
    """Compute file states for disk files plus indexed data/url sources.

    States for disk files (source_type="file"):
    - "ingested": mtime matches indexed record, or content hash matches (file unchanged),
      and its chunks were cut by the current chunking scheme.
    - "stale": mtime differs AND hash differs (file changed on disk).
    - "stale_scheme": unchanged on disk, but indexed under an older chunking scheme —
      its vectors were computed over differently cut text and are not comparable with
      new ones, so it needs re-ingesting all the same.
    - "not_ingested": file not in index.

    A file that is both changed on disk and scheme-stale reports "stale": the bytes are
    the more surprising fact, and the remedy is identical either way.

    Uses mtime fast-path: if mtime matches, no hash is computed. Equal mtime is trusted,
    so mtime-preserving restores (cp -p, rsync -a, git checkout) may mark changed files
    as ingested. Re-sync will detect and correct if mtime changes.

    Indexed data/url sources are appended and never filtered; they have no disk state to
    compare against, but they can be scheme-stale like any other source.
    Results sorted by source path.
    """
    by_source = {s.source: s for s in indexed if s.source_type == "file"}
    states: list[FileState] = []
    for e in entries:
        prior = by_source.get(str(e.path))
        if prior is None:
            states.append(FileState(str(e.path), "file", "", "not_ingested", 0))
        elif prior.mtime == e.mtime or prior.file_hash == file_sha256(e.path):
            state = _indexed_state(prior)
            states.append(
                FileState(
                    str(e.path), "file", prior.title, state, prior.chunk_count, prior.ocr_engine
                )
            )
        else:
            states.append(
                FileState(
                    str(e.path), "file", prior.title, "stale", prior.chunk_count, prior.ocr_engine
                )
            )
    for s in indexed:
        if s.source_type in ("data", "url"):
            states.append(
                FileState(
                    s.source, s.source_type, s.title, _indexed_state(s), s.chunk_count, s.ocr_engine
                )
            )
    states.sort(key=lambda s: s.source)
    return states
