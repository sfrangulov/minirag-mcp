"""CLI over the same core as the MCP server. Flags beat env vars."""

from __future__ import annotations

import json as jsonlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from minirag_mcp import __version__
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import Config, ConfigError, load_config
from minirag_mcp.embedder import Embedder, EmbedderUnavailableError, UnknownModelError
from minirag_mcp.ingest.pipeline import IngestResult, Pipeline
from minirag_mcp.ingest.scanner import scan_roots
from minirag_mcp.labels import LabelerSource
from minirag_mcp.labels.model import default_model_factory
from minirag_mcp.lock import SyncLockBusy
from minirag_mcp.results import (
    DEFAULT_FIND_LIMIT,
    QueryError,
    join_document,
    labels_status,
    list_payload,
    ocr_status,
    result_dict,
    run_find,
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


def _make_model_factory() -> Callable | None:
    return None  # patched in tests; None selects the real llama.cpp factory


LabelOpt = Annotated[
    list[str] | None,
    Parameter(
        name="--label",
        help="facet=value; repeatable. Repeating a facet matches any of its values.",
    ),
]
DateFromOpt = Annotated[
    str | None, Parameter(name="--date-from", help="Earliest document date, YYYY-MM-DD.")
]
DateToOpt = Annotated[
    str | None, Parameter(name="--date-to", help="Latest document date, YYYY-MM-DD.")
]


def _parse_labels(flags: list[str] | None) -> dict[str, str | list[str]] | None:
    if not flags:
        return None
    out: dict[str, list[str]] = {}
    for flag in flags:
        facet, sep, value = flag.partition("=")
        if not sep or not facet.strip() or not value.strip():
            _fail(f"--label takes facet=value, got {flag!r}")
        out.setdefault(facet.strip(), []).append(value.strip())
    return {k: v[0] if len(v) == 1 else v for k, v in out.items()}


def _label_source(cfg: Config) -> LabelerSource:
    return LabelerSource(cfg.roots, cfg.cache_dir, _make_model_factory() or default_model_factory)


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
        return cfg, store, Pipeline(store, emb, cfg, labels=_label_source(cfg))
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


def _document_suffix(row: dict) -> str:
    """A `sources` or `files` row's date and labels, so `--sort date` order is visible."""
    parts = []
    if "date" in row:
        parts.append(f"{row['date']['value']} ({row['date']['source']})")
    for facet, lab in row.get("labels", {}).items():
        parts.append(f"{facet}={lab['value']}" + (" (uncertain)" if lab["uncertain"] else ""))
    return "".join(f" {p}" for p in parts)


def _documents(n: int) -> str:
    return f"{n} document{'' if n == 1 else 's'} in scope {'was' if n == 1 else 'were'}"


def _warn_filter(payload: dict) -> None:
    """What a filter left out, which a human reader would otherwise only see as silence."""
    counts = payload.get("filterCounts", {})
    if n := counts.get("unlabelled"):
        print(
            f"warn: {_documents(n)} left out for having no label for a filtered facet",
            file=sys.stderr,
        )
    if n := counts.get("undated"):
        print(f"warn: {_documents(n)} left out for having no date", file=sys.stderr)
    if "filterNotice" in payload:
        print(f"warn: {payload['filterNotice']}", file=sys.stderr)


def _warn_labelling(pipeline: Pipeline, problems: list[str]) -> None:
    """A job's labelling problems, in the shape `sync` reports them. Not failures."""
    taxonomy = pipeline.labels.path if pipeline.labels else ""
    for problem in problems:
        print(f"warn: {taxonomy}: labels: {problem}", file=sys.stderr)


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
    # One job, so the model is loaded once for the whole run, not once per file.
    with pipeline.job() as report:
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
    _warn_labelling(pipeline, report.problems)
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
        with pipeline.job() as report:
            r = pipeline.ingest_url(url, source=source, title=title)
    except Exception as e:
        _fail(str(e))
    _emit(
        {"source": r.source, "chunkCount": r.chunk_count, "title": r.title},
        json,
        f"ingested {r.source} ({r.chunk_count} chunks)",
    )
    _warn_labelling(pipeline, report.problems)


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
    label: LabelOpt = None,
    date_from: DateFromOpt = None,
    date_to: DateToOpt = None,
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
    labels = _parse_labels(label)
    cfg, store, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    try:
        payload = run_query(
            store,
            pipeline.embedder,
            cfg,
            text,
            top_k=top_k,
            scopes=tuple(scope or ()),
            labels=labels,
            date_from=date_from,
            date_to=date_to,
            label_source=pipeline.labels,
        )
    except (QueryError, EmbedderUnavailableError) as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    if payload["results"]:
        human = "\n".join(_result_line(r) for r in payload["results"])
        human += "\n\nSources:\n" + "\n".join(
            f"  {s['source']} ({s['hits']} hit{'s' if s['hits'] != 1 else ''})"
            + _document_suffix(s)
            for s in payload["sources"]
        )
    else:
        human = "no results" + (f" — {payload['notice']}" if "notice" in payload else "")
    _emit(payload, json, human)
    _warn_filter(payload)


def _find_human(payload: dict) -> str:
    searched = payload["searchedDocuments"]
    lines = [
        f"{payload['matchedDocuments']} of {searched} document{'' if searched == 1 else 's'} "
        "searched match."
    ]
    for term, forms in payload["forms"].items():
        lines.append(f"{term}: " + ", ".join(f"{form} {n}" for form, n in forms.items()))
    for row in payload["documents"]:
        what = [f"{row['hits']} hit{'' if row['hits'] == 1 else 's'}"]
        if row["inTitle"]:
            what.append("in title")
        terms = f" — {', '.join(row['terms'])}" if "terms" in row else ""
        lines.append(f"{row['displayPath']} ({', '.join(what)}){terms}" + _document_suffix(row))
        if row["firstMatch"]:
            lines.append(f"    {row['firstMatch']['snippet']}")
    if "nextOffset" in payload:
        lines.append(f"more: rerun with --offset {payload['nextOffset']}")
    if "notice" in payload:
        lines.append(payload["notice"])
    return "\n".join(lines)


@app.command
def find(
    terms: list[str],
    *,
    any_: Annotated[
        bool, Parameter(name="--any", help="Match documents containing any term, not all.")
    ] = False,
    scope: list[str] | None = None,
    label: LabelOpt = None,
    date_from: DateFromOpt = None,
    date_to: DateToOpt = None,
    limit: int = DEFAULT_FIND_LIMIT,
    offset: int = 0,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """List every indexed document that mentions a word or phrase.

    A term matches where a word begins and its end is open ("акт" finds "акта", not
    "контакт", but also "актуально"); a quoted term ('"акт"') matches that exact word
    only. Case, ё/е and Unicode composition are ignored. A document must contain every
    term, or any of them with --any. The word forms each term matched are listed first.
    --limit (capped at 200) pages the listing; the counts cover every match. A term that
    starts with a dash goes after `--`, and every option before it.
    """
    labels = _parse_labels(label)
    cfg, store, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    try:
        payload = run_find(
            store,
            cfg.roots,
            terms,
            match="any" if any_ else "all",
            scopes=tuple(scope or ()),
            labels=labels,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
            label_source=pipeline.labels,
        )
    except QueryError as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    _emit(payload, json, _find_human(payload))
    _warn_filter(payload)


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
    label: LabelOpt = None,
    date_from: DateFromOpt = None,
    date_to: DateToOpt = None,
    sort: Annotated[
        str | None, Parameter(name="--sort", help="'date' lists newest first, undated last.")
    ] = None,
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
    labels = _parse_labels(label)
    cfg, store, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    try:
        payload = list_payload(
            store,
            cfg.roots,
            tuple(scope or ()),
            labels=labels,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            label_source=pipeline.labels,
        )
    except QueryError as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    human = "\n".join(
        f"[{f['state']}] {f['source']} ({f['chunkCount']} chunks)"
        + (f" [ocr:{f['ocrEngine']}]" if f["ocrEngine"] else "")
        + _document_suffix(f)
        for f in payload["files"]
    )
    _emit(payload, json, human or "no files found")
    for folder in payload["unlistedFolders"]:
        print(f"warn: {folder['path']}: {folder['message']}", file=sys.stderr)
    _warn_filter(payload)


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
        payload.update(labels_status(_label_source(cfg), store))
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


classify_app = App(name="classify", help="Inspect how documents would be labelled.")
app.command(classify_app)


@classify_app.command(name="try")
def classify_try(
    path: str,
    *,
    base_dir: BaseDirOpt = None,
    db_path: DbPathOpt = None,
    cache_dir: CacheDirOpt = None,
    model_name: ModelNameOpt = None,
    json: JsonFlag = False,
):
    """Label one file as an ingest would and print every facet's full distribution.

    Nothing is embedded or stored.
    """
    cfg, _, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    try:
        target = resolve_in_roots(path, cfg.roots, require_absolute=False)
        preview = pipeline.preview_labels(target)
    except Exception as e:  # labels off, a path outside the roots, a file it cannot read
        _fail(str(e))
        raise AssertionError from e  # unreachable
    doc, trace = preview.labels, preview.distribution
    payload = {
        "source": str(target),
        "labels": {
            facet: {
                "value": lab.value,
                "confidence": round(lab.confidence, 4),
                "source": lab.source,
                "second": lab.second,
                "uncertain": lab.uncertain,
            }
            for facet, lab in doc.labels.items()
        },
        "unset": preview.unset,
        "unanswered": sorted(doc.unanswered),
        "date": {"value": doc.date.value, "source": doc.date.source},
        "distribution": trace,
        "problems": preview.problems,
    }
    lines = [str(target)]
    for facet in preview.facets:
        lab = doc.labels.get(facet)
        if lab is None:
            lines.append(f"{facet}: not set ({preview.unset[facet]})")
            continue
        flag = " (uncertain)" if lab.uncertain else ""
        if facet in doc.unanswered:
            flag += " (the model did not answer)"
        lines.append(f"{facet}: {lab.value} [{lab.source}, {lab.confidence:.2f}]{flag}")
        for value, p in sorted(trace.get(facet, {}).items(), key=lambda kv: -kv[1]):
            lines.append(f"    {p:.3f}  {value}")
    lines.append(
        f"date: {doc.date.value or 'none'}" + (f" [{doc.date.source}]" if doc.date.value else "")
    )
    _emit(payload, json, "\n".join(lines))
    _warn_labelling(pipeline, preview.problems)
    # The preview exists to show the model's answers; one it could not give is a failure.
    if doc.unanswered:
        raise SystemExit(1)
