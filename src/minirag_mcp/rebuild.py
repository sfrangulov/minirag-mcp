"""Rebuild an index for its configured schemes in `staging_dir(db)`, then swap it in.

An embedding-only change re-embeds the stored chunk text: nothing is parsed, read with
OCR or labelled again. A chunking change re-ingests the roots. Either way the index keeps
answering queries until the swap, and writes to it are refused meanwhile, since the
rebuilt index replaces it.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from minirag_mcp import __version__, ocr
from minirag_mcp.chunker import SCHEME_VERSION, join_document
from minirag_mcp.handle import (
    EmbedderFactory,
    IndexUnavailableError,
    OpenIndex,
    configured_scheme,
    warn_unverifiable,
)
from minirag_mcp.identity import (
    FORMAT,
    LEFT_BEHIND,
    REASON_SEPARATOR,
    ChunkingScheme,
    EmbeddingScheme,
    IndexIdentity,
    RebuildInfo,
    begin_staging,
    differences,
    discard_staging,
    read_identity,
    rebuild_kind,
    seal_staging,
    swap_in,
)
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.ingest.scanner import compute_diff, scan_roots
from minirag_mcp.lock import SyncLock
from minirag_mcp.store import Store
from minirag_mcp.sync import _run_sync_unlocked

KEPT = "kept from its stored text: this installation could not read it again"


class RebuildError(Exception):
    """A rebuild stopped before its swap; the index it was rebuilding still serves."""


def rebuild(
    current: OpenIndex,
    make: EmbedderFactory,
    on_event: Callable[[str], None] | None,
    on_start: Callable[[RebuildInfo], None] | None = None,
) -> tuple[dict[str, int], list[dict], RebuildInfo]:
    """Rebuild `current` with the configured schemes and swap the result in place: the
    sync's counts and errors, and what was rebuilt.

    Before any work, `on_event` hears what is rebuilt and why, and `on_start` gets the
    kind and reasons, its counts still None. The caller holds the sync lock. A failure
    before the swap propagates, and `current` keeps serving as it was.
    """
    config = current.config
    db = config.db_path
    discard_staging(db)
    new = make(configured_scheme(config))
    scheme = EmbeddingScheme(new.model_name, new.dim, new.prefixes)
    chunking = ChunkingScheme(SCHEME_VERSION, config.token_budget)
    reasons = tuple(differences(current.identity, scheme, chunking))
    kind = rebuild_kind(current.identity, scheme, chunking)
    if on_event:
        on_event(
            f"rebuilding {db} ({kind}: {REASON_SEPARATOR.join(reasons)}); it answers from "
            "what it holds until the rebuilt index replaces it"
        )
    if on_start:
        on_start(RebuildInfo(kind, reasons, None, None, None))
    recorded = read_identity(db)
    if recorded is not None and recorded.embedding == scheme and recorded.probe:
        # The model is the one the index was built with, and is checked against its probe.
        probe = recorded.probe
    else:
        probe = tuple(new.probe_vector())
        if why := new.reference_error(probe):
            raise IndexUnavailableError(why)
        warn_unverifiable(new, "the rebuilt index")
    new.expect_probe(probe)

    staging = begin_staging(db)
    # Moves to `db` with the swap, so no sync of the new index starts before this returns.
    lock = SyncLock(staging)
    sealed = False
    try:
        lock.acquire()
        target = Store(staging, dim=new.dim, fts_language=config.fts_language)
        pipeline = Pipeline(target, new, config, labels=current.labels)
        reembedded = 0
        if kind == "reembed":
            reembedded = target.copy_reembedded(
                current.store, new.embed_documents, on_event=on_event
            )
        large: list[Path] = []
        counts, errors = _run_sync_unlocked(
            pipeline,
            target,
            config.roots,
            config.max_file_size,
            None,
            on_event,
            large,
            exclude=config.exclude,
        )
        restored = _restore(current, pipeline, errors, on_event, large)
        if problem := target.update_indices():
            errors.append({"source": str(staging), "error": problem})
        seal_staging(
            db,
            IndexIdentity(
                format=FORMAT,
                minirag=__version__,
                generation=current.identity.generation,
                embedding=scheme,
                probe=probe,
                chunking=chunking,
                fts_language=target.fts_index_language,
                document_table=True,
            ),
        )
        sealed = True
        errors += [{"source": str(left), "error": LEFT_BEHIND} for left in swap_in(db)]
    except BaseException:
        # A sealed staging folder is a complete index, which a swap stopped midway needs.
        if not sealed:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock.release()
    swapped = read_identity(db)
    return counts, errors, RebuildInfo(kind, reasons, swapped.generation, reembedded, restored)


def _restore(
    current: OpenIndex,
    pipeline: Pipeline,
    errors: list[dict],
    on_event: Callable[[str], None] | None,
    large_tables: Sequence[Path],
) -> int:
    """Store again, from its stored text, every source of `current` the rebuilt index
    lacks and a sync of `current` would keep; how many.

    Kept: what was ingested from data or a URL, and a file this installation could not
    read again, or whose folder it could not see into. Not kept: a file gone from disk, or
    excluded by a rule, or moved, which `compute_diff` decides as for any sync, nor one of
    `large_tables`, those the rebuild's own sync left out for RAG_MAX_TABLE_ROWS. A reingest
    reads every table again, so this includes one indexed, unchanged, before the limit
    was set.

    Also stored again: a file read with OCR before, which an installation without OCR read
    again from its text layer alone. A PDF with any text layer reads without error then,
    and the text OCR gave its scanned pages would be gone.
    """
    config = current.config
    blind: list[tuple[Path, str | None]] = []
    ignored: list[Path] = []
    entries = scan_roots(list(config.roots), blind=blind, ignored=ignored, exclude=config.exclude)
    indexed = current.store.list_sources()
    diff = compute_diff(
        entries,
        indexed,
        max_file_size=config.max_file_size,
        roots=config.roots,
        blind=blind,
        ignored=ignored,
    )
    gone = (
        set(diff.to_delete) | {m.source for m in diff.moved} | {str(path) for path in large_tables}
    )
    rebuilt = {info.source: info for info in pipeline.store.list_sources()}
    ocr_lost = not ocr.available()
    restored = 0
    for info in indexed:
        if info.source in gone:
            continue
        again = rebuilt.get(info.source)
        if again is not None and not (ocr_lost and info.ocr_engine and not again.ocr_engine):
            continue
        try:
            pipeline.reingest_stored(info, join_document(current.store.all_chunks(info.source)))
        except Exception as e:
            raise RebuildError(
                f"{info.source} could not be stored again from its stored text ({e}), so the "
                "rebuilt index would have lost it, and it was not swapped in. Delete it from "
                "the index (delete_file), or sync again once the cause is fixed."
            ) from e
        restored += 1
        if info.source_type == "file":
            errors.append({"source": info.source, "error": KEPT})
        if on_event:
            on_event(f"kept {info.source} from its stored text")
    return restored
