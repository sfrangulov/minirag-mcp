"""CLI over the same core as the MCP server. Flags beat env vars."""

from __future__ import annotations

import json as jsonlib
import os
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from minirag_mcp import __version__
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import Config, ConfigError, load_config
from minirag_mcp.embedder import Embedder, EmbedderUnavailableError, UnknownModelError
from minirag_mcp.ingest.pipeline import IngestResult, Pipeline
from minirag_mcp.ingest.scanner import scan_roots
from minirag_mcp.lock import SyncLockBusy
from minirag_mcp.results import (
    QueryError,
    join_document,
    list_payload,
    ocr_status,
    result_dict,
    run_query,
    scheme_status,
)
from minirag_mcp.security import SecurityError, resolve_in_roots
from minirag_mcp.store import DimensionMismatchError, IndexOpenError, Store
from minirag_mcp.sync import run_sync

app = App(name="minirag-mcp", version=__version__)

JsonFlag = Annotated[bool, Parameter(name="--json", help="Machine-readable JSON output")]
BaseDirOpt = Annotated[
    list[str] | None,
    Parameter(name="--base-dir", help="Document root; repeatable. Overrides BASE_DIR/BASE_DIRS."),
]
DbPathOpt = Annotated[
    str | None, Parameter(name="--db-path", help="Index directory. Overrides DB_PATH.")
]
CacheDirOpt = Annotated[
    str | None,
    Parameter(name="--cache-dir", help="Embedding model cache. Overrides CACHE_DIR."),
]
ModelNameOpt = Annotated[
    str | None,
    Parameter(name="--model-name", help="fastembed model id. Overrides MODEL_NAME."),
]


def _make_embedder(cfg: Config) -> Embedder:
    return Embedder(cfg.model_name, cfg.cache_dir)  # patched in tests


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _load_cfg(
    base_dir: list[str] | None,
    db_path: str | None,
    cache_dir: str | None,
    model_name: str | None,
) -> Config:
    return load_config(
        os.environ,
        base_dir_flags=base_dir or (),
        db_path_flag=db_path,
        cache_dir_flag=cache_dir,
        model_name_flag=model_name,
    )


def _load(
    base_dir: list[str] | None,
    db_path: str | None,
    cache_dir: str | None,
    model_name: str | None,
) -> tuple[Config, Store, Pipeline]:
    try:
        cfg = _load_cfg(base_dir, db_path, cache_dir, model_name)
        emb = _make_embedder(cfg)
        store = Store(cfg.db_path, dim=emb.dim)
        return cfg, store, Pipeline(store, emb, cfg)
    except ConfigError as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    except UnknownModelError as e:
        _fail(f"{e} — see fastembed's supported models")
        raise AssertionError from e  # unreachable
    except (DimensionMismatchError, IndexOpenError) as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable


def _emit(payload: dict, as_json: bool, human: str) -> None:
    print(jsonlib.dumps(payload, ensure_ascii=False, indent=2) if as_json else human)


def _result_line(r: dict) -> str:
    return (
        f"[{r['score']:.3f}] {r['source']}#{r['chunkIndex']} — {r['title']}\n    {r['text'][:160]}"
    )


@app.command
def ingest(
    paths: list[str],
    *,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """Ingest files or directories (recursive)."""
    cfg, store, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    files: list[Path] = []
    seen: set[Path] = set()
    failures: list[dict] = []
    try:
        for raw in paths:
            p = resolve_in_roots(raw, cfg.roots, require_absolute=False)
            blind: list[tuple[Path, str | None]] = []
            candidates = [e.path for e in scan_roots([p], blind=blind)] if p.is_dir() else [p]
            failures += [
                {
                    "source": str(folder),
                    "error": f"could not list this folder ({error}); nothing under it was ingested",
                }
                for folder, error in blind
                if error is not None  # an empty folder holds nothing to ingest
            ]
            for c in candidates:
                resolved = c.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(c)
    except Exception as e:  # SecurityError or a scan-time filesystem error
        _fail(str(e))
    done: list[IngestResult] = []
    for f in files:
        try:
            done.append(pipeline.ingest_file(f))
        except Exception as e:  # per-file failures never abort the batch
            failures.append({"source": str(f), "error": str(e)})
    payload = {
        "ingested": [{"source": r.source, "chunkCount": r.chunk_count} for r in done],
        "failed": failures,
    }
    human = "\n".join([f"ingested {len(done)} file(s):"] + [f"  {r.source}" for r in done])
    _emit(payload, json, human)
    for fail in failures:
        print(f"warn: {fail['source']}: {fail['error']}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


@app.command(name="ingest-url")
def ingest_url(
    url: str,
    *,
    source: str | None = None,
    title: str | None = None,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """Fetch an http(s) URL and index it."""
    _, _, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    try:
        r = pipeline.ingest_url(url, source=source, title=title)
    except Exception as e:
        _fail(str(e))
    _emit(
        {"source": r.source, "chunkCount": r.chunk_count, "title": r.title},
        json,
        f"ingested {r.source} ({r.chunk_count} chunks)",
    )


@app.command
def sync(
    path: str | None = None,
    *,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """Reconcile the index with the document roots (synchronous).

    Refuses to start if another process is already syncing this index.
    """
    cfg, store, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    scope = None
    if path is not None:
        try:
            scope = resolve_in_roots(path, cfg.roots, require_absolute=False)
        except SecurityError as e:
            _fail(str(e))
    try:
        counts, errors = run_sync(
            pipeline,
            store,
            cfg.roots,
            cfg.max_file_size,
            scope=scope,
            on_event=lambda m: print(m, file=sys.stderr),
        )
    except SyncLockBusy as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    human = ", ".join(f"{k}: {v}" for k, v in counts.items())
    _emit({"counts": counts, "errors": errors}, json, human)
    for err in errors:
        print(f"warn: {err['source']}: {err['error']}", file=sys.stderr)
    if counts["failed"] > 0:
        raise SystemExit(1)  # same contract as `ingest`: failed files are a failed run


@app.command
def query(
    text: str,
    *,
    scope: list[str] | None = None,
    top_k: int = 8,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """Hybrid search over the index.

    --top-k must be at least 1 and is capped at 100; a larger value is clamped.
    """
    if top_k < 1:
        _fail(f"--top-k must be at least 1, got {top_k}")
    cfg, store, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    try:
        payload = run_query(
            store, pipeline.embedder, cfg, text, top_k=top_k, scopes=tuple(scope or ())
        )
    except (QueryError, EmbedderUnavailableError) as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    if payload["results"]:
        human = "\n".join(_result_line(r) for r in payload["results"])
        human += "\n\nSources:\n" + "\n".join(
            f"  {s['source']} ({s['hits']} hit{'s' if s['hits'] != 1 else ''})"
            for s in payload["sources"]
        )
    else:
        human = "no results" + (f" — {payload['notice']}" if "notice" in payload else "")
    _emit(payload, json, human)


@app.command(name="read-neighbors")
def read_neighbors(
    *,
    file_path: str | None = None,
    source: str | None = None,
    chunk_index: int,
    before: int = 1,
    after: int = 1,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """Read chunks around a given chunk index."""
    cfg, store, _ = _load(base_dir, db_path, cache_dir, model_name)
    if file_path is not None:
        try:
            key = str(resolve_in_roots(file_path, cfg.roots, require_absolute=False))
        except SecurityError as e:
            _fail(str(e))
    elif source is not None:
        key = source
    else:
        _fail("provide --file-path or --source")
    chunks = store.neighbors(key, chunk_index, before=before, after=after)
    payload = {"chunks": [result_dict(r) for r in chunks]}
    human = (
        "\n\n".join(f"[{r.chunk_index}] {r.source} — {r.title}\n{r.text}" for r in chunks)
        or "no chunks"
    )
    _emit(payload, json, human)


@app.command
def read(
    path: str | None = None,
    *,
    source: str | None = None,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """Print a source's full indexed content, as Markdown.

    Reconstructed from its chunks rather than concatenated from them: the context each
    chunk repeats to carry it into its own vector is emitted once, where the document
    had it.
    """
    cfg, store, _ = _load(base_dir, db_path, cache_dir, model_name)
    if path is not None:
        try:
            key = str(resolve_in_roots(path, cfg.roots, require_absolute=False))
        except SecurityError as e:
            _fail(str(e))
    elif source is not None:
        key = source
    else:
        _fail("provide a path or --source")
    chunks = store.all_chunks(key)
    if not chunks:
        _fail(f"source not found in index: {key}")
    info = store.get_source(key)
    text = join_document(chunks)
    payload = {
        "source": key,
        "sourceType": info.source_type,
        "title": info.title,
        "chunkCount": len(chunks),
        "text": text,
    }
    _emit(payload, json, text)


@app.command(name="list")
def list_cmd(
    *,
    scope: list[str] | None = None,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """List files on disk with ingestion state, plus indexed data/url sources.

    States: ingested, stale (changed on disk), stale_scheme (unchanged on disk but
    indexed under an older chunking scheme), not_ingested, unreadable (indexed and
    still on disk, but of a type this install cannot read because an optional extra
    is absent — images need [ocr]). Everything but ingested and unreadable needs a
    sync; an unreadable source is kept as indexed and no sync can refresh it here.
    unlisted: indexed, under a folder that could not be listed — each such folder is
    named in a warning on stderr and in unlistedFolders with --json.
    """
    cfg, store, _ = _load(base_dir, db_path, cache_dir, model_name)
    payload = list_payload(store, cfg.roots, tuple(scope or ()))
    human = "\n".join(
        f"[{f['state']}] {f['source']} ({f['chunkCount']} chunks)"
        + (f" [ocr:{f['ocrEngine']}]" if f["ocrEngine"] else "")
        for f in payload["files"]
    )
    _emit(payload, json, human or "no files found")
    for folder in payload["unlistedFolders"]:
        print(f"warn: {folder['path']}: {folder['message']}", file=sys.stderr)


@app.command
def status(
    *,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """Show index and configuration status."""
    # Degraded mode, matching the server's `status` tool: a broken configuration is
    # exactly what you run `status` to diagnose, so it reports the error and exits 0.
    # Every other command still fails loudly on the same ConfigError.
    try:
        _load_cfg(base_dir, db_path, cache_dir, model_name)
    except ConfigError as e:
        payload = {"version": __version__, "configError": str(e)}
        _emit(payload, json, "\n".join(f"{k}: {v}" for k, v in payload.items()))
        return
    cfg = _load_cfg(base_dir, db_path, cache_dir, model_name)
    payload = {
        "version": __version__,
        "roots": [str(r) for r in cfg.roots],
        "dbPath": str(cfg.db_path),
        "model": cfg.model_name,
        "hybridWeight": cfg.hybrid_weight,
        "chunkScheme": SCHEME_VERSION,
        **ocr_status(),
    }
    # An index that cannot be opened is reported, not raised: the server's `status` does
    # the same, and both are what a broken setup is diagnosed with.
    try:
        store = Store(cfg.db_path, dim=_make_embedder(cfg).dim)
        payload["chunkCount"] = store.chunk_count()
        payload["sourceCount"] = store.source_count()
        payload.update(scheme_status(store))
    except Exception as e:
        payload["indexError"] = str(e)
    human = "\n".join(f"{k}: {v}" for k, v in payload.items())
    _emit(payload, json, human)


@app.command
def delete(
    path: str | None = None,
    *,
    source: str | None = None,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """Delete an indexed file (by path) or data/url item (by --source)."""
    cfg, store, _ = _load(base_dir, db_path, cache_dir, model_name)
    if path is not None:
        try:
            key = str(resolve_in_roots(path, cfg.roots, require_absolute=False))
        except SecurityError as e:
            _fail(str(e))
    elif source is not None:
        key = source
    else:
        _fail("provide a path or --source")
    deleted = store.delete_source(key)
    if deleted == 0:
        _fail(f"source not found in index: {key}")
    _emit({"source": key, "deletedChunks": deleted}, json, f"deleted {deleted} chunk(s) of {key}")
