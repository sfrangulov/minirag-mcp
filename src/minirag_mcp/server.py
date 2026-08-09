"""FastMCP server: 11 tools over the shared core."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from minirag_mcp import __version__
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import Config, ConfigError, load_config
from minirag_mcp.embedder import Embedder
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.ingest.scanner import compute_states, scan_roots
from minirag_mcp.instructions import build_instructions
from minirag_mcp.results import (
    aggregate_sources,
    join_document,
    parent_map,
    result_dict,
    scheme_status,
)
from minirag_mcp.scope import is_under
from minirag_mcp.security import resolve_in_roots
from minirag_mcp.store import MAX_TOP_K, Store
from minirag_mcp.sync import SyncManager


@dataclass
class _Ctx:
    config: Config
    store: Store
    embedder: object
    pipeline: Pipeline
    sync: SyncManager


def _scopes(scope: str | list[str] | None) -> tuple[str, ...]:
    if scope is None:
        return ()
    if isinstance(scope, str):
        return (scope,)
    return tuple(scope)


def create_app(
    config: Config | None,
    *,
    config_error: str | None = None,
    embedder: object | None = None,
    env: Mapping[str, str] | None = None,
) -> FastMCP:
    # Instructions come from the environment directly rather than from Config, so a
    # server whose configuration failed to load still tells the client how to use it —
    # same reason `status` keeps answering.
    mcp = FastMCP("minirag-mcp", instructions=build_instructions(env))
    holder: dict[str, _Ctx | None] = {"ctx": None}

    def ctx() -> _Ctx:
        if config is None or config_error is not None:
            raise ToolError(f"Configuration error: {config_error or 'no configuration'}")
        if holder["ctx"] is None:
            emb = (
                embedder if embedder is not None else Embedder(config.model_name, config.cache_dir)
            )
            store = Store(config.db_path, dim=emb.dim)
            pipeline = Pipeline(store, emb, config)
            holder["ctx"] = _Ctx(config, store, emb, pipeline, SyncManager(pipeline, store, config))
        return holder["ctx"]

    def guard(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ToolError:
                raise
            except KeyError as e:
                raise ToolError(str(e.args[0]) if e.args else str(e)) from e
            except Exception as e:
                raise ToolError(str(e)) from e

        return inner

    @mcp.tool
    @guard
    def sync_start(path: str | None = None) -> dict:
        """Reconcile the index with the document roots (or one path inside them).

        Returns a jobId immediately; poll sync_status until state is
        'succeeded' or 'failed'. New and changed files are ingested,
        byte-identical files skipped, vanished files removed from the index.
        Only the latest sync job is retained — starting a new one, or a
        server restart, discards the previous job's record.

        Only one sync runs against an index at a time. If one is already in
        progress — in this server or in another process, such as a
        `minirag-mcp sync` in a terminal — this call fails immediately with a
        message identifying it, and no job is started. Single-file ingests and
        queries are never blocked by a running sync.
        """
        c = ctx()
        scope = None
        if path is not None:
            scope = resolve_in_roots(path, c.config.roots)
        return {"jobId": c.sync.start(scope)}

    @mcp.tool
    @guard
    def sync_status(jobId: str) -> dict:
        """Poll a sync job started by sync_start.

        Returns state ('pending' | 'running' | 'succeeded' | 'failed'),
        counts (scanned/ingested/skipped/deleted/failed), and any per-file
        errors. Only the latest job is retained — an old jobId, or any
        jobId from before a server restart, raises an error.
        """
        return ctx().sync.status(jobId).to_dict()

    @mcp.tool
    @guard
    def ingest_file(filePath: str) -> dict:
        """Ingest or re-ingest one file, replacing any content already indexed for it.

        filePath must be an absolute path inside a configured document root.
        Re-ingesting an already-indexed file discards its old chunks and
        replaces them with freshly parsed ones.
        """
        c = ctx()
        path = resolve_in_roots(filePath, c.config.roots)
        r = c.pipeline.ingest_file(path)
        return {"source": r.source, "chunkCount": r.chunk_count, "title": r.title}

    @mcp.tool
    @guard
    def ingest_data(data: str, source: str, format: str = "text", title: str | None = None) -> dict:
        """Ingest text/markdown/html content the client holds, under a source id you choose.

        format is one of "text", "markdown", or "html" (default "text").
        source is a stable identifier you pick, not a filesystem path —
        re-using it replaces the previously ingested content for that id,
        so reuse the same source to update an item.
        """
        r = ctx().pipeline.ingest_data(data, source=source, fmt=format, title=title)
        return {"source": r.source, "chunkCount": r.chunk_count, "title": r.title}

    @mcp.tool
    @guard
    def ingest_url(url: str, source: str | None = None, title: str | None = None) -> dict:
        """Fetch an http(s) URL, convert it to Markdown, and index it.

        Only http and https schemes are accepted, and the host must not be a
        private or local address (loopback, link-local, private, reserved) —
        set ALLOW_PRIVATE_URLS=1 to lift that. This is the one tool that
        reaches the network — every other tool works purely against local
        files and the local index. source defaults to the URL itself; pass
        one to control the index key or to update a previously ingested URL.
        """
        r = ctx().pipeline.ingest_url(url, source=source, title=title)
        return {"source": r.source, "chunkCount": r.chunk_count, "title": r.title}

    @mcp.tool
    @guard
    def query_documents(query: str, topK: int = 8, scope: str | list[str] | None = None) -> dict:
        """Hybrid search: semantic similarity plus a keyword boost for exact terms.

        Returns `results` — ranked chunks with text, source, title,
        chunkIndex, score and parentId — `sources`, the distinct sources
        among those results in rank order each with a hits count, and
        `parents`, a map from parentId to section text. Use `sources` to
        answer "which documents cover this topic" without inspecting
        individual chunks.

        `text` is the passage that matched and that `score` describes.
        `parentId` names the section it sits in — a transcript time
        window, a heading section, a slide, a table. Look it up in
        `parents` to read the whole section when the match alone is too
        small to act on. A chunk indexed before parent sections existed
        has parentId null and no entry; re-sync to fill it in.

        topK must be at least 1 and is capped at 100; a larger value is
        silently clamped to the cap rather than rejected.

        Cite what you take, so the user can verify it, in whatever
        language you answer: [1], [2] inline, then a Sources list —
        one line per document, never per chunk. `source` verbatim: the
        full path is the only thing that locates the file. `title`
        labels it and may be shortened. Plain text, never a markdown
        link or file://.

        Sources:
        [1] /docs/onboarding.md — Onboarding Guide
        """
        if not query.strip():
            raise ToolError("query must not be empty")
        if topK < 1:
            raise ToolError(f"topK must be at least 1, got {topK}")
        c = ctx()
        results = c.store.search(
            query,
            c.embedder.embed_query(query),
            top_k=min(topK, MAX_TOP_K),
            hybrid_weight=c.config.hybrid_weight,
            scopes=_scopes(scope),
            max_distance=c.config.max_distance,
            grouping=c.config.grouping,
            max_files=c.config.max_files,
        )
        return {
            "results": [result_dict(r) for r in results],
            "sources": aggregate_sources(results),
            "parents": parent_map(results),
        }

    @mcp.tool
    @guard
    def read_chunk_neighbors(
        chunkIndex: int,
        filePath: str | None = None,
        source: str | None = None,
        before: int = 1,
        after: int = 1,
    ) -> dict:
        """Read the chunks immediately before and after a search result, for context.

        Provide exactly one of filePath (absolute path inside a document
        root) or source (the id of a data/url item). before/after control
        how many chunks to include on each side of chunkIndex (both
        default to 1).
        """
        c = ctx()
        if filePath is not None:
            key = str(resolve_in_roots(filePath, c.config.roots))
        elif source is not None:
            key = source
        else:
            raise ToolError("Provide filePath or source")
        chunks = c.store.neighbors(key, chunkIndex, before=before, after=after)
        return {"chunks": [result_dict(r) for r in chunks]}

    @mcp.tool
    @guard
    def read_file(filePath: str | None = None, source: str | None = None) -> dict:
        """Read a source's entire indexed content as Markdown.

        The document is reconstructed from its chunks, not concatenated
        from them: the context each chunk repeats so its own vector
        carries it — a heading breadcrumb, a time-window label, a table's
        header row — is emitted once, where the document had it.

        Provide exactly one of filePath (absolute path inside a document
        root) or source (the id of a data/url item). The response holds
        the full document text, so large documents produce large
        responses — prefer read_chunk_neighbors when only the context
        around one chunk is needed.
        """
        c = ctx()
        not_indexed_hint = ""
        if filePath is not None:
            key = str(resolve_in_roots(filePath, c.config.roots))
            p = Path(key)
            if not p.exists():
                raise ToolError(f"File does not exist: {key}")
            if not p.is_file():
                raise ToolError(f"Not a file: {key}")
            not_indexed_hint = " — ingest it first with ingest_file or sync_start."
        elif source is not None:
            key = source
        else:
            raise ToolError("Provide filePath or source")
        chunks = c.store.all_chunks(key)
        if not chunks:
            raise ToolError(f"Source not found in index: {key}{not_indexed_hint}")
        info = c.store.get_source(key)
        return {
            "source": key,
            "sourceType": info.source_type,
            "title": info.title,
            "chunkCount": len(chunks),
            "text": join_document(chunks),
        }

    @mcp.tool
    @guard
    def list_files(scope: str | list[str] | None = None) -> dict:
        """List files found on disk under the document roots, plus indexed data/url sources.

        Each disk file is reported with a state: "ingested" (index matches
        disk), "stale" (changed on disk since it was indexed),
        "stale_scheme" (unchanged on disk, but indexed under an older
        chunking scheme, so its vectors are not comparable with current
        ones), or "not_ingested" (never indexed). Everything but "ingested"
        needs a sync_start. "stale_scheme" is the per-source view of what
        status reports as staleChunkCount. Data and url sources have no disk
        state to compare against, so they are "ingested" or "stale_scheme".
        """
        c = ctx()
        scopes = _scopes(scope)
        entries = scan_roots(c.config.roots)
        if scopes:
            entries = [e for e in entries if any(is_under(str(e.path), p) for p in scopes)]
        indexed = c.store.list_sources(scopes=scopes)
        states = compute_states(entries, indexed)
        return {
            "files": [
                {
                    "source": s.source,
                    "sourceType": s.source_type,
                    "title": s.title,
                    "state": s.state,
                    "chunkCount": s.chunk_count,
                }
                for s in states
            ]
        }

    @mcp.tool
    @guard
    def delete_file(filePath: str | None = None, source: str | None = None) -> dict:
        """Delete an indexed file, data item, or url item from the index.

        Provide exactly one of filePath (absolute path inside a document
        root) or source (the id of a data/url item). This only removes the
        index entry — a file left in place under a document root is
        re-ingested by a later sync_start.
        """
        c = ctx()
        if filePath is not None:
            key = str(resolve_in_roots(filePath, c.config.roots))
        elif source is not None:
            key = source
        else:
            raise ToolError("Provide filePath or source")
        deleted = c.store.delete_source(key)
        if deleted == 0:
            raise ToolError(f"Source not found in index: {key}")
        return {"source": key, "deletedChunks": deleted}

    @mcp.tool
    def status() -> dict:
        """Report configuration and index status. Works even when configuration is invalid.

        Always includes version. When configuration is valid, also includes
        roots, dbPath, model, hybridWeight, and chunkCount/sourceCount (both
        present on every call, 0 before anything is indexed) — or, if opening
        the index itself fails, indexError instead of the counts. When
        configuration is invalid, includes configError instead, and every
        other tool raises an error referencing it until the configuration is
        fixed.

        chunkScheme is the chunking scheme the index is being written with.
        staleChunkCount counts chunks still stored under an older scheme —
        when it is above zero, schemeWarning explains that those chunks need
        a re-sync to be rebuilt.
        """
        if config is None or config_error is not None:
            return {"version": __version__, "configError": config_error or "no configuration"}
        out: dict = {
            "version": __version__,
            "roots": [str(r) for r in config.roots],
            "dbPath": str(config.db_path),
            "model": config.model_name,
            "hybridWeight": config.hybrid_weight,
            "chunkScheme": SCHEME_VERSION,
        }
        try:
            c = ctx()
            out["chunkCount"] = c.store.chunk_count()
            out["sourceCount"] = c.store.source_count()
            out.update(scheme_status(c.store))
        except Exception as e:
            out["indexError"] = str(e)
        return out

    return mcp


def run_server() -> None:
    try:
        config: Config | None = load_config(os.environ)
        error = None
    except ConfigError as e:
        config, error = None, str(e)
    create_app(config, config_error=error, env=os.environ).run()
