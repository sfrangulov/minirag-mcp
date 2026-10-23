"""Recursive document-root scanning and sync diff computation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from minirag_mcp import ocr
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.ingest import ignore
from minirag_mcp.ingest.pages import predates_pages
from minirag_mcp.ingest.parser import IMAGE_EXTENSIONS, SUPPORTED_EXTENSIONS, supported_extensions
from minirag_mcp.ingest.pipeline import file_sha256
from minirag_mcp.scope import is_under
from minirag_mcp.store import SourceInfo

SKIP_DIRS = frozenset({"node_modules", "__pycache__", ".venv", "venv"})
# Word, Excel and PowerPoint write a lock file beside every open document, named after
# it and so carrying its extension: `~$port.docx` for `report.docx`. It is a short
# binary blob, not the ZIP the extension promises, so ingesting one can only fail —
# and a whole sync exits 1 because somebody had a document open. LibreOffice needs no
# entry here: its lock files start with a dot.
OFFICE_LOCK_PREFIX = "~$"
# Every extension any installation of this package can read. `supported_extensions()`
# is a subset of this that shrinks when an optional extra is absent, so the difference
# is what an install indexed before but cannot read now.
KNOWN_EXTENSIONS = SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS


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
    # Indexed sources missing from the scan because their extension left the whitelist
    # with an optional extra, not because the disk changed — their files are still
    # there. Kept in the index, and separate from `to_delete` so a caller can say why
    # nothing happened to them.
    unreadable: list[str]
    # Folders whose contents the scan could not see. What is indexed under them is
    # unknown rather than absent, so none of it is in `to_delete`.
    held: list[HeldFolder] = field(default_factory=list)
    # Indexed sources whose file reappeared elsewhere with the same name and bytes.
    # Neither side is in `to_delete` or `to_ingest`: the record is re-pathed instead.
    moved: list[Move] = field(default_factory=list)
    # Unchanged PDFs that predate page numbers and were read with OCR, which this
    # install lacks: read again they would lose their scanned pages and pass for
    # current, so they are kept as they are.
    kept_scans: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class Move:
    source: str
    path: Path
    mtime: float


@dataclass(frozen=True)
class HeldFolder:
    path: Path
    # Why the scan could not see inside: the OS error, or None for a root that is an
    # empty folder while the index has something under it.
    error: str | None
    root: bool
    # Indexed files under it that the sync kept, counted within the scope when there
    # is one, and the names of the first few.
    kept: int
    examples: tuple[str, ...] = ()
    scoped: bool = False
    # The sync was asked for this folder or something inside it, so none of what it
    # was asked to do happened.
    wanted: bool = False

    @property
    def fails(self) -> bool:
        """A configured root is always the user's to fix, and so is a folder the sync was
        asked for; any other folder only when it hid something indexed. An unreadable
        folder holding nothing indexed would otherwise fail every sync while it exists."""
        return self.root or self.wanted or self.kept > 0

    @property
    def message(self) -> str:
        one = self.kept == 1
        files = f"{self.kept} indexed file{'' if one else 's'}"
        if self.scoped:
            files += " of the synced path"
        names = ", ".join(self.examples)
        if self.kept > len(self.examples):
            names += f" and {self.kept - len(self.examples)} more"
        it = "it" if one else "them"
        kept = f"{'it was' if one else 'they were'} kept, not deleted"
        if self.error is None:
            return (
                f"this root is an empty folder, but the index holds {files} under it "
                f"({names}), and {kept}: an empty folder is also what an unmounted volume "
                f"leaves behind. If you removed {it} on purpose, delete {it} with "
                "delete_file or `minirag-mcp delete`, or take this root out of the "
                "configuration."
            )
        unlisted = f"could not list this folder ({self.error})"
        if self.kept == 0:
            return f"{unlisted}; nothing under it was scanned"
        return f"{unlisted}; {files} under it ({names}), and {kept}"


def _contained(real: Path, real_roots: Sequence[Path]) -> bool:
    """True if `real` (already resolved) sits inside any of `real_roots` (already resolved).

    Same containment rule as security.resolve_in_roots: equal to a root, or having
    that root among its parents.
    """
    return any(real == r or r in real.parents for r in real_roots)


def scan_roots(
    roots: Sequence[Path],
    blind: list[tuple[Path, str | None]] | None = None,
    ignored: list[Path] | None = None,
    rule_roots: Sequence[Path] | None = None,
) -> list[ScanEntry]:
    """Recursively scan roots for files with whitelisted extensions.

    Skips hidden files and directories (dot-prefixed), Office lock files
    (OFFICE_LOCK_PREFIX) and SKIP_DIRS. Prunes directory traversal for SKIP_DIRS and
    hidden dirs. Symlinked directories are not traversed (os.walk followlinks=False
    avoids cycles).

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

    When `blind` is given, it collects the folders whose contents this result cannot
    vouch for: each directory that could not be listed (a missing root, a folder without
    read permission, a macOS privacy denial) with its OS error, and each root that is an
    empty directory, with None — what an unmounted volume leaves behind on Linux. A
    caller comparing the result with the index must not read the absence of files under
    them as deletion.

    Every path is held to the `.miniragignore` of each configured root above it:
    `rule_roots`, which default to `roots` and differ only when a folder inside a root
    is scanned on its own. A folder a rule excludes is not walked, and a symlink whose
    target a rule excludes is left out too. When `ignored` is given, it collects every
    excluded file and folder, a folder once. A root under an ignore file that cannot be
    read is not walked and goes into `blind` with the reason, like a folder that cannot
    be listed: nothing under it is deleted, and the other roots are scanned as usual.

    Returns entries sorted by path.
    """
    real_roots = [Path(r).resolve() for r in roots]
    exts = supported_extensions()
    entries: list[ScanEntry] = []
    seen: set[Path] = set()

    def note(error: OSError) -> None:
        if blind is not None:
            blind.append((Path(error.filename), error.strerror or str(error)))

    ignorer = ignore.Ignorer(rule_roots or roots)
    broken = ignorer.broken()
    seen_ignored: set[Path] = set()

    def drop(p: Path) -> None:
        if ignored is not None and p not in seen_ignored:
            seen_ignored.add(p)
            ignored.append(p)

    def held(folder: Path) -> bool:
        """A nested root whose rule file cannot be read: reported, never walked."""
        root = ignorer.root_of(folder)
        if root is None or root not in broken:
            return False
        if blind is not None:
            blind.append((folder, broken[root]))
        return True

    def excluded(p: Path, is_dir: bool) -> bool:
        if ignorer.matches(p, is_dir) is None:
            return False
        drop(p)
        return True

    for root in roots:
        top = Path(root)
        reason = next(
            (broken[r] for f in (top, *top.parents) if (r := ignorer.root_of(f)) in broken),
            None,
        )
        if reason is not None:
            if blind is not None:
                blind.append((top, reason))
            continue
        if ignorer.excluded_by(top) is not None:
            drop(top)  # an inner root an outer root's rule takes out whole
            continue
        for dirpath, dirnames, filenames in os.walk(root, onerror=note):
            if blind is not None and dirpath == os.fspath(root) and not dirnames and not filenames:
                blind.append((Path(root), None))
            dirnames[:] = sorted(
                d
                for d in dirnames
                if not d.startswith(".")
                and d not in SKIP_DIRS
                and not held(Path(dirpath) / d)
                and not excluded(Path(dirpath) / d, True)
            )
            for name in sorted(filenames):
                if name.startswith(".") or name.startswith(OFFICE_LOCK_PREFIX):
                    continue
                p = Path(dirpath) / name
                suffix = p.suffix.lower()
                # Checked before the extensions this install reads, so a file indexed
                # earlier by an install with an extra is still known to be excluded.
                if suffix not in KNOWN_EXTENSIONS or excluded(p, False):
                    continue
                if suffix not in exts:
                    continue
                # Dedupe by resolved path to handle overlapping roots
                resolved = p.resolve()
                if resolved in seen:
                    continue
                if not _contained(resolved, real_roots):
                    continue  # symlink escaping every root: not part of the corpus
                if p.is_symlink():
                    try:
                        target_excluded = ignorer.excluded_by(resolved) is not None
                    except ignore.IgnoreFileError:
                        continue  # its target's root is held: so is the link
                    if target_excluded:
                        drop(p)  # its text is the excluded file's
                        continue
                seen.add(resolved)
                st = p.stat()
                entries.append(ScanEntry(path=p, size=st.st_size, mtime=st.st_mtime))
    entries.sort(key=lambda e: e.path)
    return entries


def _kept_unreadable(source: str, needs_extra: set[str]) -> bool:
    """Whether an indexed source absent from the scan is one to keep rather than delete.

    Two facts, and both are needed. The extension says why the scan did not look for it
    — an extra this install lacks reads that type. The file still being on disk says the
    user did not delete it. With only the first, an image really removed from the
    document root is retained forever and reported as a file type this installation
    cannot read: true about the type, silent about the file being gone.
    """
    return Path(source).suffix.lower() in needs_extra and Path(source).exists()


def _in_scope(path_str: str, scope: Path | None) -> bool:
    """Whether `path_str` is the scope path itself or sits under it.

    Both sides go through `Path` first, so a non-canonical spelling ("/a//b") is
    compared in its normalised form; the containment rule itself is `scope.is_under`,
    the same one the store's SQL filter and `list_files` apply.
    """
    if scope is None:
        return True
    return is_under(str(Path(path_str)), str(scope))


def _needs_ocr_for_pages(prior: SourceInfo) -> bool:
    return predates_pages(prior) and bool(prior.ocr_engine) and not ocr.available()


def _reread_for_pages(prior: SourceInfo, record_pages: bool) -> bool:
    return record_pages and predates_pages(prior) and not _needs_ocr_for_pages(prior)


def compute_diff(
    entries: list[ScanEntry],
    indexed: list[SourceInfo],
    *,
    max_file_size: int,
    scope: Path | None = None,
    roots: Sequence[Path] = (),
    blind: Sequence[tuple[Path, str | None]] = (),
    ignored: Sequence[Path] = (),
    record_pages: bool = True,
) -> SyncDiff:
    """Compute sync diff: files to ingest, delete, and unchanged status.

    Uses mtime fast-path: if mtime matches, file is considered unchanged without hashing.
    This is a deliberate trade-off — equal mtime is trusted, so an mtime-preserving restore
    (cp -p, rsync -a, git checkout) may leave a stale index entry. Re-sync will detect and
    update it only if mtime differs (then hash is compared).

    A source whose chunks predate the current chunking scheme is re-ingested whatever
    its bytes say. Its file has not changed; what it was cut into has. Without this,
    `status` would tell the user to re-sync and the re-sync would skip every file. A
    PDF indexed before page numbers were stored is re-ingested for the same reason,
    unless the index has no column to record them in (`record_pages`), or the PDF was
    read with OCR and this install has none: it is then in `kept_scans`.

    Deletion is proposed only for a source this installation could have scanned, or one
    whose file is gone. An extension in `KNOWN_EXTENSIONS` but not in the current
    whitelist is one an optional extra reads (images need [ocr]), so a scan without that
    extra never looked for the file, and if the file is still on disk its absence from
    the scan is evidence about the installation rather than the disk — reported as
    `unreadable`. Without the distinction, one sync from a light install deletes every
    indexed image and reports only `deleted: N`.

    Nor is deletion proposed for a source under a folder in `blind` (see `scan_roots`).
    Missing from the scan says nothing about such a file, and treating it as deleted
    would empty the index whenever a root is unmounted or unreadable. Those folders come
    back in `held`, the ones that bear on `scope` only; an empty root is held only when
    the index has something under it.

    A source that vanished, or is hidden in a held folder, and whose name and bytes
    turn up at a new path is returned in `moved` rather than deleted and re-ingested.
    A source whose chunks two racing first ingests stored twice is re-ingested even
    though its bytes match.

    Args:
        entries: Scanned disk files.
        indexed: Files currently in the index (SourceInfo from Store).
        max_file_size: Files larger than this are marked oversized (not ingested).
        scope: If provided, limit processing to files under this path.
        roots: The roots `entries` was scanned from.
        blind: Folders the scan could not see into, as `scan_roots` reports them.

    Returns:
        SyncDiff with to_ingest, to_delete, unchanged, oversized, unreadable, held
        and moved.
    """
    indexed_files = {
        s.source: s for s in indexed if s.source_type == "file" and _in_scope(s.source, scope)
    }
    to_ingest: list[Path] = []
    unchanged: list[Path] = []
    oversized: list[Path] = []
    kept_scans: list[Path] = []
    seen: set[str] = set()

    for e in entries:
        if not _in_scope(str(e.path), scope):
            continue
        seen.add(str(e.path))
        if e.size > max_file_size:
            oversized.append(e.path)
            continue
        prior = indexed_files.get(str(e.path))
        if prior is None or prior.scheme_version < SCHEME_VERSION or prior.duplicated:
            to_ingest.append(e.path)
        elif prior.mtime == e.mtime or prior.file_hash == file_sha256(e.path):
            if _reread_for_pages(prior, record_pages):
                to_ingest.append(e.path)
            elif record_pages and _needs_ocr_for_pages(prior):
                kept_scans.append(e.path)
            else:
                unchanged.append(e.path)
        else:
            to_ingest.append(e.path)

    # Overlapping roots walk an inner folder twice. Innermost first, so a source is
    # credited to the folder nearest it.
    reason = dict(blind)
    blind = sorted(reason.items(), key=lambda b: len(str(b[0])), reverse=True)

    needs_extra = KNOWN_EXTENSIONS - supported_extensions()
    to_delete: list[str] = []
    unreadable: list[str] = []
    hidden: dict[str, Path] = {}
    for src in indexed_files:
        if src in seen:
            continue
        hidden_by = next((path for path, _ in blind if is_under(src, str(path))), None)
        if hidden_by is not None:
            hidden[src] = hidden_by
        elif any(is_under(src, str(p)) for p in ignored):
            to_delete.append(src)  # even one this install cannot read: a rule took it out
        elif _kept_unreadable(src, needs_extra):
            unreadable.append(src)
        else:
            to_delete.append(src)

    # A source hidden by an empty root is a candidate too: the same name and bytes
    # turning up elsewhere is evidence that it moved, which is all an empty folder left
    # behind cannot give. Behind a folder that cannot be listed the original may still
    # be there, so a copy elsewhere proves nothing.
    emptied = sorted(s for s, folder in hidden.items() if reason[folder] is None)
    moved = _pair_moves(to_delete + emptied, to_ingest, indexed_files, entries, record_pages)
    paired_sources = {m.source for m in moved}
    paired_paths = {m.path for m in moved}
    to_delete = [s for s in to_delete if s not in paired_sources]
    to_ingest = [p for p in to_ingest if p not in paired_paths]
    kept: dict[Path, list[str]] = {path: [] for path, _ in blind}
    for src, folder in sorted(hidden.items()):
        if src not in paired_sources:
            kept[folder].append(src)

    root_set = set(roots)
    held = [
        HeldFolder(
            path=path,
            error=error,
            root=path in root_set,
            kept=len(kept[path]),
            examples=tuple(Path(s).name for s in kept[path][:3]),
            scoped=scope is not None,
            wanted=scope is not None
            and _in_scope(str(scope), path)
            and not _reached_by_inner_root(scope, path, roots, reason),
        )
        for path, error in sorted(blind, key=lambda b: str(b[0]))
        if (error is not None or kept[path])
        and (scope is None or _in_scope(str(path), scope) or _in_scope(str(scope), path))
    ]
    return SyncDiff(
        to_ingest=to_ingest,
        to_delete=sorted(to_delete),
        unchanged=unchanged,
        oversized=oversized,
        unreadable=sorted(unreadable),
        held=held,
        moved=moved,
        kept_scans=kept_scans,
    )


def _reached_by_inner_root(
    scope: Path, folder: Path, roots: Sequence[Path], blind: dict[Path, str | None]
) -> bool:
    """Whether a root nested in `folder`, and listed without trouble, holds `scope`.

    A folder can be passed through without being listable, so a root below it is still
    walked, and a sync scoped inside that root did what it was asked.
    """
    return any(
        root != folder
        and root not in blind
        and is_under(str(root), str(folder))
        and _in_scope(str(scope), root)
        for root in roots
    )


def _pair_moves(
    to_delete: list[str],
    to_ingest: list[Path],
    indexed: dict[str, SourceInfo],
    entries: list[ScanEntry],
    record_pages: bool = True,
) -> list[Move]:
    """Pair each vanished source with a new path holding the same bytes under the same name.

    The name has to match because a title can come from it: a renamed file may need a
    different title, and only a re-ingest computes one. A source cut by an older
    chunking scheme, or a PDF stored without page numbers that a sync would read again,
    is not paired, since it needs reading again anyway. Pairing is one to one, in path
    order, so two identical copies of one vanished file move it to the first and ingest
    the second.
    """
    gone: dict[tuple[str, str], list[str]] = {}
    for source in to_delete:
        prior = indexed[source]
        if (
            prior.file_hash
            and prior.scheme_version >= SCHEME_VERSION
            and not prior.duplicated
            and not _reread_for_pages(prior, record_pages)
        ):
            gone.setdefault((Path(source).name, prior.file_hash), []).append(source)
    if not gone:
        return []
    names = {name for name, _ in gone}
    mtimes = {e.path: e.mtime for e in entries}
    moves: list[Move] = []
    for path in to_ingest:
        if path.name not in names or str(path) in indexed:
            continue
        try:
            digest = file_sha256(path)
        except OSError:
            continue  # gone again, or unreadable: the ingest reports it
        candidates = gone.get((path.name, digest))
        if candidates:
            moves.append(Move(source=candidates.pop(0), path=path, mtime=mtimes[path]))
    return moves


@dataclass(frozen=True)
class FileState:
    source: str
    source_type: str
    title: str
    # "ingested" | "not_ingested" | "stale" | "stale_scheme" | "stale_pages" | "unreadable"
    # | "unlisted"
    state: str
    chunk_count: int
    ocr_engine: str = ""


def _indexed_state(prior: SourceInfo) -> str:
    """Whether an indexed source is current, or needs reading again though unchanged: cut
    by an older chunking scheme, or a PDF stored before page numbers were.

    A scheme-stale source needs the same action as a changed one — re-ingest — and
    `compute_diff` already re-ingests it. Reporting it as plain "ingested" is what made
    the listing contradict `status`, which says the index is stale and to re-sync; an
    agent choosing what to work on from the listing would have seen nothing to do.
    """
    if prior.scheme_version < SCHEME_VERSION:
        return "stale_scheme"
    if _needs_ocr_for_pages(prior):
        return "unreadable"
    return "stale_pages" if predates_pages(prior) else "ingested"


def compute_states(
    entries: list[ScanEntry],
    indexed: list[SourceInfo],
    blind: Sequence[tuple[Path, str | None]] = (),
) -> list[FileState]:
    """Compute file states for disk files plus indexed data/url sources.

    States for disk files (source_type="file"):
    - "ingested": mtime matches indexed record, or content hash matches (file unchanged),
      and its chunks were cut by the current chunking scheme.
    - "stale": mtime differs AND hash differs (file changed on disk).
    - "stale_scheme": unchanged on disk, but indexed under an older chunking scheme —
      its vectors were computed over differently cut text and are not comparable with
      new ones, so it needs re-ingesting all the same.
    - "stale_pages": an unchanged PDF indexed before page numbers were stored; a sync
      reads it again to add them. One that was read with OCR, which this install
      lacks, is "unreadable" instead: no sync here can read it again.
    - "not_ingested": file not in index.
    - "unreadable": indexed, still on disk, but of a type this installation cannot read
      because an optional extra is absent (images need [ocr]). The scan never looked for
      it, so it has no entry to compare against; `compute_diff` keeps it rather than
      deleting it, and it is listed here because `status` still counts its chunks and
      `query_documents` still returns them.
    - "unlisted": indexed, under a folder in `blind` (see `scan_roots`), so whether it
      is still on disk is unknown. Listed for the same reason as "unreadable".

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
    scanned = {str(e.path) for e in entries}
    needs_extra = KNOWN_EXTENSIONS - supported_extensions()
    for source, prior in by_source.items():
        if source in scanned:
            continue
        if any(is_under(source, str(folder)) for folder, _ in blind):
            state = "unlisted"
        elif _kept_unreadable(source, needs_extra):
            state = "unreadable"
        else:
            continue
        states.append(
            FileState(source, "file", prior.title, state, prior.chunk_count, prior.ocr_engine)
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
