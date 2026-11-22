"""CLI over the same core as the MCP server. Flags beat env vars.

A command uses the index the environment variables configure, or with `--index NAME` that
index of indexes.toml. `--index NAME` without a command serves that index alone on stdio;
`serve` serves every index of indexes.toml over HTTP.
"""

from __future__ import annotations

import json as jsonlib
import os
import sys
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from minirag_mcp import __version__
from minirag_mcp.chunker import join_document
from minirag_mcp.config import Config, ConfigError, load_config
from minirag_mcp.embedder import EmbedderUnavailableError
from minirag_mcp.handle import OpenIndex, make_embedder
from minirag_mcp.identity import REASON_SEPARATOR, EmbeddingScheme
from minirag_mcp.indexes import default_config_path
from minirag_mcp.ingest.ignore import IGNORE_FILE, excluded_by
from minirag_mcp.ingest.pipeline import IngestResult, Pipeline
from minirag_mcp.ingest.scanner import scan_roots
from minirag_mcp.registry import (
    SINGLE_INDEX_NAME,
    UNRESTRICTED,
    Connection,
    IndexEntry,
    IndexRefused,
    IndexRegistry,
    Use,
)
from minirag_mcp.results import (
    DEFAULT_FIND_LIMIT,
    QueryError,
    ingest_reply,
    list_indexes_payload,
    list_payload,
    registry_status,
    result_dict,
    run_find,
    run_query,
    with_index,
)
from minirag_mcp.security import SecurityError

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


def _make_embedder(cfg: Config) -> object | None:
    return None  # patched in tests; None builds through make_embedder


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


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


@dataclass(frozen=True)
class Selection:
    """The options read before or after the command: an index of indexes.toml, and which
    indexes.toml."""

    index: str | None = None
    config: str | None = None


_NOTHING_SELECTED = Selection()
_SELECTION: ContextVar[Selection] = ContextVar("minirag_selection", default=_NOTHING_SELECTED)

ENV_FLAGS_WITH_INDEX = (
    "--index reads indexes.toml; --base-dir, --db-path, --cache-dir and --model-name "
    "configure the environment-variable index and cannot be combined with it. Set them in "
    "the index's [index.NAME] table."
)
CONFIG_WITHOUT_INDEX = (
    "--config names the indexes.toml that --index reads; without --index this command uses "
    "the index the environment variables configure"
)

# The `--` before a `find` term that starts with a dash is the command's to read, so the
# options around the command are given a delimiter no argument equals: NUL, which argv
# cannot hold. Not "": cyclopts compares every token to it and would drop an empty
# argument. --index and --config are therefore read anywhere, even after `--`:
# `find foo -- --index beta` searches beta for foo.
app.meta.end_of_options_delimiter = "\0"


@app.meta.default
def main(
    *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
    index: Annotated[
        str | None,
        Parameter(
            name="--index",
            help="An index of indexes.toml, which the command uses in place of the index the "
            "environment variables configure. Without a command, serve that index alone over "
            "MCP on stdio.",
        ),
    ] = None,
    config: Annotated[
        str | None,
        Parameter(
            name="--config",
            help="The indexes.toml to read. Overrides MINIRAG_CONFIG and the default "
            "~/.config/minirag/indexes.toml.",
        ),
    ] = None,
) -> object:
    selected = _SELECTION.set(Selection(index, config))
    try:
        if tokens:
            return app(tokens)
        if index is not None:
            return _serve(index)
        _fail("nothing to do: give a command, or --index NAME to serve that index on stdio")
    finally:
        _SELECTION.reset(selected)


def _config_path() -> Path:
    """--config, else MINIRAG_CONFIG, else where this platform keeps indexes.toml."""
    config = _SELECTION.get().config
    if config is not None:
        return Path(config).expanduser().absolute()
    try:
        return default_config_path(os.environ)
    except ConfigError as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable


def _file_registry() -> IndexRegistry:
    """The indexes of indexes.toml; exit 1 naming the file when it does not load."""
    path, models = _config_path(), _make_model_factory()
    try:
        registry = IndexRegistry.from_file(path, os.environ, model_factory=models)
        # As in `_single`, only a test's stand-in is handed in: the handles build and own
        # every real embedder, so both shells refuse an index alike.
        if any(_make_embedder(entry.config) is not None for entry in registry.entries):
            registry = IndexRegistry.from_file(
                path, os.environ, embedder_factory=_cli_embedder, model_factory=models
            )
    except ConfigError as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    return registry


def _serve(name: str) -> None:
    """The MCP server on stdio for index `name` of indexes.toml alone: what one entry of a
    client's configuration starts. A file that does not load, or names no such index,
    fails before it serves."""
    from minirag_mcp.server import create_app  # fastmcp loads in a serving process only

    registry = _file_registry()
    try:
        registry.resolve(UNRESTRICTED, name)
    except IndexRefused as e:
        _fail(str(e))
    create_app(registry, connection=Connection.only(name), env=os.environ).run()


def _load_cfg(
    base_dir: list[str] | None,
    db_path: str | None,
    cache_dir: str | None,
    model_name: str | None,
) -> Config:
    if _SELECTION.get().config is not None:
        _fail(CONFIG_WITHOUT_INDEX)
    return load_config(
        os.environ,
        base_dir_flags=base_dir or (),
        db_path_flag=db_path,
        cache_dir_flag=cache_dir,
        model_name_flag=model_name,
    )


def _cli_embedder(cfg: Config, scheme: EmbeddingScheme) -> object:
    # Looked up on every call, so that a test's patch of _make_embedder applies.
    emb = _make_embedder(cfg)
    return emb if emb is not None else make_embedder(cfg, scheme)


def _single(cfg: Config) -> IndexRegistry:
    # Without a stand-in the handle builds its own embedders, as the server's does, so both
    # shells refuse an index alike and the handle closes what it built.
    factory = _cli_embedder if _make_embedder(cfg) is not None else None
    return IndexRegistry.single(cfg, embedder_factory=factory, model_factory=_make_model_factory())


# Reaches every index. Without --index its default is the environment's, named default;
# with it, --index names the index of indexes.toml a command uses.
_CONNECTION = Connection(SINGLE_INDEX_NAME, None)


def _registry(
    base_dir: list[str] | None,
    db_path: str | None,
    cache_dir: str | None,
    model_name: str | None,
) -> IndexRegistry:
    if _SELECTION.get().index is not None:
        if any(flag is not None for flag in (base_dir, db_path, cache_dir, model_name)):
            _fail(ENV_FLAGS_WITH_INDEX)
        return _file_registry()
    try:
        return _single(_load_cfg(base_dir, db_path, cache_dir, model_name))
    except ConfigError as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable


def _open_in(registry: IndexRegistry, use: Use) -> tuple[IndexEntry, OpenIndex]:
    try:
        return registry.open(_CONNECTION, _SELECTION.get().index, use)
    except IndexRefused as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable


def _open(
    use: Use,
    base_dir: list[str] | None,
    db_path: str | None,
    cache_dir: str | None,
    model_name: str | None,
) -> tuple[IndexRegistry, IndexEntry, OpenIndex]:
    registry = _registry(base_dir, db_path, cache_dir, model_name)
    return (registry, *_open_in(registry, use))


def _inside(registry: IndexRegistry, entry: IndexEntry, path: str) -> Path:
    try:
        return registry.in_roots(_CONNECTION, entry, path, require_absolute=False)
    except (SecurityError, IndexRefused) as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable


def _emit(payload: dict, as_json: bool, human: str) -> None:
    print(jsonlib.dumps(payload, ensure_ascii=False, indent=2) if as_json else human)


def _on_pages(pages: str) -> str:
    """`p. 12` or `pp. 12-13` for a PDF chunk's pages, after a space; "" without them."""
    if not pages:
        return ""
    return f" {'pp' if '-' in pages else 'p'}. {pages}"


def _result_line(r: dict) -> str:
    where = f"{r['source']}#{r['chunkIndex']}{_on_pages(r.get('pages', ''))}"
    return f"[{r['score']:.3f}] {where} — {r['title']}\n    {r['text'][:160]}"


def _document_suffix(row: dict) -> str:
    """A `sources` or `files` row's date and labels, so `--sort date` order is visible."""
    parts = []
    if "date" in row:
        parts.append(f"{row['date']['value']} ({row['date']['source']})")
    for facet, lab in row.get("labels", {}).items():
        parts.append(f"{facet}={lab['value']}" + (" (uncertain)" if lab["uncertain"] else ""))
    return "".join(f" {p}" for p in parts)


def _look_alike_lines(row: dict, indent: str) -> list[str]:
    return [
        f"{indent}looks like {a['displayPath']} ({a['overlap']:.0%} shared text)"
        + _document_suffix(a)
        for a in row.get("lookAlikes", ())
    ]


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


def _warn(notes: list[str]) -> None:
    """A reply's `warnings`, which the JSON output carries in the reply itself."""
    for note in notes:
        print(f"warn: {note}", file=sys.stderr)


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
    registry, entry, opened = _open(Use.INGEST, base_dir, db_path, cache_dir, model_name)
    cfg, pipeline = entry.config, opened.pipeline
    files: list[Path] = []
    # Already held to the rules by their scan; only files named directly are checked again.
    scanned: set[Path] = set()
    seen: set[Path] = set()
    failures: list[dict] = []
    try:
        for raw in paths:
            p = registry.in_roots(_CONNECTION, entry, raw, require_absolute=False)
            blind: list[tuple[Path, str | None]] = []
            if (
                p.is_dir()
                and (
                    by := excluded_by(p, cfg.roots, cfg.exclude, exclude_source=cfg.exclude_source)
                )
                is not None
            ):
                failures.append({"source": str(p), "error": f"{p} is excluded by {by}"})
                continue
            if p.is_dir():
                candidates = [
                    e.path
                    for e in scan_roots([p], blind=blind, rule_roots=cfg.roots, exclude=cfg.exclude)
                ]
                scanned.update(candidates)
            else:
                candidates = [p]
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
    except Exception as e:  # a path outside the roots, or a scan-time filesystem error
        _fail(registry.failure(entry, e))
    done: list[IngestResult] = []
    # One job, so the model is loaded once for the whole run, not once per file.
    with pipeline.job() as report:
        for f in files:
            try:
                with registry.refusals(_CONNECTION, entry):
                    done.append(pipeline.ingest_file(f, rules_checked=f in scanned))
            except Exception as e:  # per-file failures never abort the batch
                failures.append({"source": str(f), "error": registry.failure(entry, e)})
    notes = entry.handle.notes()
    payload = {"ingested": [ingest_reply(r, notes) for r in done], "failed": failures}
    human = "\n".join([f"ingested {len(done)} file(s):"] + [f"  {r.source}" for r in done])
    _emit(with_index(payload, entry), json, human)
    if done:
        _warn(notes)
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
    registry, entry, opened = _open(Use.INGEST, base_dir, db_path, cache_dir, model_name)
    pipeline = opened.pipeline
    try:
        with pipeline.job() as report, registry.refusals(_CONNECTION, entry):
            r = pipeline.ingest_url(url, source=source, title=title)
    except Exception as e:
        _fail(registry.failure(entry, e))
    reply = ingest_reply(r, entry.handle.notes())
    _emit(with_index(reply, entry), json, f"ingested {r.source} ({r.chunk_count} chunks)")
    _warn(reply.get("warnings", []))
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

    Refuses to start if another process is already syncing this index. An index whose
    configuration changed since it was built ("rebuild pending" in status) is rebuilt by a
    sync without a path, and a sync with one is refused.
    """
    registry, entry, _ = _open(Use.SYNC, base_dir, db_path, cache_dir, model_name)
    scope = None if path is None else _inside(registry, entry, path)
    try:
        with registry.refusals(_CONNECTION, entry):
            lock = entry.handle.admit_sync(scope)
        counts, errors, rebuilt = entry.handle.run_admitted(
            scope, lock, on_event=lambda m: print(m, file=sys.stderr)
        )
    except Exception as e:  # a refusal, a busy lock, a rebuild that stopped before its swap
        _fail(str(e))
        raise AssertionError from e  # unreachable
    human = ", ".join(f"{k}: {v}" for k, v in counts.items())
    if rebuilt is not None:
        human += f"; rebuilt ({rebuilt.kind}): {REASON_SEPARATOR.join(rebuilt.reasons)}"
    payload = {
        "counts": counts,
        "errors": errors,
        "rebuild": rebuilt.to_dict() if rebuilt is not None else None,
    }
    _emit(with_index(payload, entry), json, human)
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
    registry, entry, opened = _open(Use.READ, base_dir, db_path, cache_dir, model_name)
    cfg = entry.config
    try:
        payload = run_query(
            opened.store,
            opened.embedder,
            cfg,
            text,
            top_k=top_k,
            scopes=tuple(scope or ()),
            labels=labels,
            date_from=date_from,
            date_to=date_to,
            label_source=opened.labels,
            reranker=opened.reranker,
        )
    except (QueryError, EmbedderUnavailableError) as e:
        _fail(registry.failure(entry, e))
        raise AssertionError from e  # unreachable
    if payload["results"]:
        human = "\n".join(_result_line(r) for r in payload["results"])
        lines = []
        for s in payload["sources"]:
            lines.append(
                f"  {s['source']} ({s['hits']} hit{'s' if s['hits'] != 1 else ''})"
                + _document_suffix(s)
            )
            lines += _look_alike_lines(s, "      ")
        human += "\n\nSources:\n" + "\n".join(lines)
    else:
        human = "no results" + (f" — {payload['notice']}" if "notice" in payload else "")
    _emit(with_index(payload, entry), json, human)
    _warn(payload.get("warnings", []))
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
        if first := row["firstMatch"]:
            on = _on_pages(first.get("pages", ""))
            lines.append(f"   {on}: {first['snippet']}" if on else f"    {first['snippet']}")
        lines += _look_alike_lines(row, "    ")
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
    registry, entry, opened = _open(Use.READ, base_dir, db_path, cache_dir, model_name)
    cfg = entry.config
    try:
        payload = run_find(
            opened.store,
            cfg.roots,
            terms,
            match="any" if any_ else "all",
            scopes=tuple(scope or ()),
            labels=labels,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
            label_source=opened.labels,
        )
    except QueryError as e:
        _fail(registry.failure(entry, e))
        raise AssertionError from e  # unreachable
    _emit(with_index(payload, entry), json, _find_human(payload))
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
    registry, entry, opened = _open(Use.READ, base_dir, db_path, cache_dir, model_name)
    if file_path is not None:
        key = str(_inside(registry, entry, file_path))
    elif source is not None:
        key = source
    else:
        _fail("provide --file-path or --source")
    chunks = opened.store.neighbors(key, chunk_index, before=before, after=after)
    payload = {"chunks": [result_dict(r) for r in chunks]}
    human = (
        "\n\n".join(
            f"[{r.chunk_index}] {r.source}{_on_pages(r.pages)} — {r.title}\n{r.text}"
            for r in chunks
        )
        or "no chunks"
    )
    _emit(with_index(payload, entry), json, human)


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
    registry, entry, opened = _open(Use.READ, base_dir, db_path, cache_dir, model_name)
    if path is not None:
        key = str(_inside(registry, entry, path))
    elif source is not None:
        key = source
    else:
        _fail("provide a path or --source")
    chunks = opened.store.all_chunks(key)
    if not chunks:
        _fail(f"source not found in index: {key}")
    info = opened.store.get_source(key)
    text = join_document(chunks)
    payload = {
        "source": key,
        "sourceType": info.source_type,
        "title": info.title,
        "chunkCount": len(chunks),
        "text": text,
    }
    _emit(with_index(payload, entry), json, text)


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
    indexed under an older chunking scheme), stale_pages (an unchanged PDF indexed
    before page numbers were stored), not_ingested, unreadable (indexed and
    still on disk, but of a type this install cannot read because an optional extra
    is absent — images need [ocr]). Everything but ingested and unreadable needs a
    sync; an unreadable source is kept as indexed and no sync can refresh it here.
    unlisted: indexed, under a folder that could not be listed — each such folder is
    named in a warning on stderr and in unlistedFolders with --json.
    """
    labels = _parse_labels(label)
    registry, entry, opened = _open(Use.READ, base_dir, db_path, cache_dir, model_name)
    cfg = entry.config
    try:
        payload = list_payload(
            opened.store,
            cfg.roots,
            tuple(scope or ()),
            exclude=cfg.exclude,
            exclude_source=cfg.exclude_source,
            labels=labels,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            label_source=opened.labels,
            db_path=cfg.db_path,
        )
    except QueryError as e:
        _fail(registry.failure(entry, e))
        raise AssertionError from e  # unreachable
    human = "\n".join(
        f"[{f['state']}] {f['source']} ({f['chunkCount']} chunks)"
        + (f" [ocr:{f['ocrEngine']}]" if f["ocrEngine"] else "")
        + _document_suffix(f)
        + (f"\n  ! {f['lastSyncError']}" if "lastSyncError" in f else "")
        for f in payload["files"]
    )
    _emit(with_index(payload, entry), json, human or "no files found")
    for folder in payload["unlistedFolders"]:
        print(f"warn: {folder['path']}: {folder['message']}", file=sys.stderr)
    if report := payload.get("lastSyncReport"):
        print(f"warn: last-sync report unreadable: {report['unreadable']}", file=sys.stderr)
    if n := payload.get("ignored"):
        print(
            f"note: {n} file(s) and folder(s) excluded by {IGNORE_FILE} or {cfg.exclude_source}",
            file=sys.stderr,
        )
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
    selection = _SELECTION.get()
    # Degraded mode, matching the server's `status` tool: a broken configuration is
    # exactly what you run `status` to diagnose, so it reports the error and exits 0.
    # Every other command still fails loudly on the same ConfigError. An indexes.toml that
    # does not load fails it too: no configuration loaded before it to report beside it.
    if selection.index is not None:
        registry = _registry(base_dir, db_path, cache_dir, model_name)
    else:
        try:
            registry = _single(_load_cfg(base_dir, db_path, cache_dir, model_name))
        except ConfigError as e:
            registry = IndexRegistry.single(None, error=str(e))
    try:
        payload = registry_status(registry, _CONNECTION, selection.index)
    except IndexRefused as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    human = "\n".join(
        _last_sync_line(v) if k == "lastSync" else f"{k}: {v}" for k, v in payload.items()
    )
    if "dbPath" in payload and "lastSync" not in payload:
        human += "\nlast sync: no report yet"
    _emit(payload, json, human)


def _last_sync_line(last: dict) -> str:
    if "unreadable" in last:
        return f"last sync: report unreadable: {last['unreadable']}"
    by = last["ranBy"] + (f", scope {last['scope']}" if last["scope"] else "")
    return (
        f"last sync: {last['finishedAt']} ({by}) {last['outcome']}, "
        f"{last['failed']} failed, {last['largeTables']} large tables"
    )


def _index_line(row: dict) -> str:
    """One `indexes` row; an unavailable index says why in place of its counts."""
    if "documentCount" in row:
        found = f"{row['documentCount']} documents, {row['chunkCount']} chunks"
    else:
        found = row["stateReason"]
    line = f"{row['name']} [{row['state']}] {found}, {row['model']}, {row['language']}"
    return line + (f" — {row['description']}" if row["description"] else "")


@app.command
def indexes(*, json: JsonFlag = False) -> None:
    """List the indexes of indexes.toml, as the MCP list_indexes tool does.

    Each is opened to be reported, which may load its embedding model. A row is
    "name [state] counts, model, language — description"; an index that cannot be opened
    gives why in place of its counts, and --json also the reason a syncing or "rebuild
    pending" index is not ready.
    """
    if _SELECTION.get().index is not None:
        _fail(
            "`indexes` lists every index of indexes.toml; --index selects the one another "
            "command uses and cannot be combined with it"
        )
    payload = list_indexes_payload(_file_registry(), UNRESTRICTED)
    _emit(payload, json, "\n".join(_index_line(row) for row in payload["indexes"]))


@app.command
def serve() -> None:
    """Serve every index of indexes.toml over streamable HTTP on this machine.

    A client connects to `http://127.0.0.1:8765/mcp/<index>`, which makes `<index>` its
    default; `?allow=a,b` lets it name a and b too, and `/mcp` has no default. Every request
    sends the token that the first start writes next to indexes.toml (or to [server]
    token_file) as `Authorization: Bearer <token>`. SIGHUP reads indexes.toml again;
    [server] changes take a restart.
    """
    if _SELECTION.get().index is not None:
        _fail(
            "`serve` serves every index of indexes.toml; --index selects the one another "
            "command uses and cannot be combined with it"
        )
    from minirag_mcp.http_server import serve as serve_http  # fastmcp loads in a server only

    path = _config_path()
    if not path.exists():
        _fail(
            f"no indexes.toml at {path}; create one "
            "(docs/launchd/local.minirag-mcp.plist shows how to run it)"
        )
    try:
        serve_http(path, os.environ)
    except ConfigError as e:
        _fail(str(e))


DASHBOARD_CODE_TIMEOUT_SECONDS = 5


@app.command
def dashboard(*, no_open: Annotated[bool, Parameter(name="--no-open")] = False) -> None:
    """Open the dashboard of the running `serve` in the browser, already signed in.

    Asks the server for a one-time code with the token of its token file, which is never
    shown, and opens the page that spends the code. With --no-open, print that URL alone.
    """
    if _SELECTION.get().index is not None:
        _fail("`dashboard` shows every index of indexes.toml; --index cannot be combined with it")
    import urllib.error
    import urllib.request

    from minirag_mcp.http_server import read_token
    from minirag_mcp.indexes import load_server_config

    path = _config_path()
    try:
        settings = load_server_config(path, os.environ)
    except ConfigError as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    host = f"[{settings.host}]" if ":" in settings.host else settings.host
    base = f"http://{host}:{settings.port}"
    unanswered = f"No minirag server answers at {base}. Start it with: minirag-mcp serve"
    if not settings.token_file.exists():
        _fail(unanswered)
    try:
        token = read_token(settings.token_file)
    except ConfigError as e:
        _fail(str(e))
        raise AssertionError from e  # unreachable
    request = urllib.request.Request(
        f"{base}/dashboard/code", method="POST", headers={"Authorization": f"Bearer {token}"}
    )
    # Never through a proxy: urllib sends a loopback address to the one http_proxy or the
    # system settings name too, and this request carries the token.
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with direct.open(request, timeout=DASHBOARD_CODE_TIMEOUT_SECONDS) as reply:
            code = jsonlib.load(reply)["code"]
    except urllib.error.HTTPError as e:
        _fail(f"the server at {base} refused the request for a login code (HTTP {e.code})")
    except (OSError, ValueError, KeyError, TypeError):
        _fail(unanswered)
    url = f"{base}/dashboard/login?code={code}"
    if no_open:
        print(url)
        return
    import webbrowser

    webbrowser.open(url)
    print("Opened the minirag dashboard in your browser.")


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
    registry, entry, opened = _open(Use.READ, base_dir, db_path, cache_dir, model_name)
    if path is not None:
        key = str(_inside(registry, entry, path))
    elif source is not None:
        key = source
    else:
        _fail("provide a path or --source")
    try:
        with registry.refusals(_CONNECTION, entry):
            deleted = opened.store.delete_source(key)
    except IndexRefused as e:  # a rebuild runs, or swapped another index in meanwhile
        _fail(str(e))
        raise AssertionError from e  # unreachable
    if deleted == 0:
        _fail(f"source not found in index: {key}")
    _emit(
        with_index({"source": key, "deletedChunks": deleted}, entry),
        json,
        f"deleted {deleted} chunk(s) of {key}",
    )


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
    registry, entry, opened = _open(Use.READ, base_dir, db_path, cache_dir, model_name)
    pipeline = opened.pipeline
    try:
        target = registry.in_roots(_CONNECTION, entry, path, require_absolute=False)
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
    _emit(with_index(payload, entry), json, "\n".join(lines))
    _warn_labelling(pipeline, preview.problems)
    # The preview exists to show the model's answers; one it could not give is a failure.
    if doc.unanswered:
        raise SystemExit(1)
