"""FastMCP server: 12 tools over the shared core."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from minirag_mcp import __version__
from minirag_mcp.chunker import join_document
from minirag_mcp.config import Config, ConfigError, load_config
from minirag_mcp.handle import IndexHandle, OpenIndex
from minirag_mcp.ingest.ignore import scope_excluded_by
from minirag_mcp.instructions import build_instructions
from minirag_mcp.results import (
    DEFAULT_FIND_LIMIT,
    ingest_reply,
    list_payload,
    result_dict,
    run_find,
    run_query,
    status_payload,
)
from minirag_mcp.security import resolve_in_roots


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
    model_factory: Callable | None = None,
) -> FastMCP:
    # Instructions come from the environment directly rather than from Config, so a
    # server whose configuration failed to load still tells the client how to use it —
    # same reason `status` keeps answering.
    mcp = FastMCP("minirag-mcp", instructions=build_instructions(env))
    handle = (
        None
        if config is None or config_error is not None
        else IndexHandle(
            config,
            embedder_factory=None if embedder is None else (lambda scheme: embedder),
            model_factory=model_factory,
        )
    )

    def live() -> IndexHandle:
        if handle is None:
            raise ToolError(f"Configuration error: {config_error or 'no configuration'}")
        return handle

    def opened() -> OpenIndex:
        return live().get()

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
        byte-identical files skipped, vanished files removed from the index. A
        file moved to another folder under the same name and with the same bytes
        keeps its chunks under the new path instead of being read again.
        Only the latest sync job is retained — starting a new one, or a
        server restart, discards the previous job's record.

        Only one sync runs against an index at a time. If one is already in
        progress — in this server or in another process, such as a
        `minirag-mcp sync` in a terminal — this call fails immediately with a
        message identifying it, and no job is started. Queries are never
        blocked by a running sync; single-file ingests and deletes are refused
        only while it rebuilds the index.

        On an index whose configuration changed since it was built (status:
        indexState "rebuild pending"), a sync without a path rebuilds it: the
        stored text is embedded again when only the model or its prefixes
        changed, the roots are ingested again when the chunking changed. The
        index answers from its existing vectors until the rebuilt one replaces
        it, and refuses writes meanwhile. A sync with a path is refused there.
        """
        h = live()
        c = h.get()
        scope = None
        if path is not None:
            scope = resolve_in_roots(path, c.config.roots)
            if (by := scope_excluded_by(scope, c.config.roots, c.config.exclude)) is not None:
                raise ToolError(f"{scope} is excluded by {by}; nothing under it can be synced")
        return {"jobId": h.start_sync(scope)}

    @mcp.tool
    @guard
    def sync_status(jobId: str) -> dict:
        """Poll a sync job started by sync_start.

        Returns state ('pending' | 'running' | 'succeeded' | 'failed'),
        counts (scanned/ingested/skipped/moved/deleted/relabelled/signed/
        ignored/unreadable/largeTables/failed), and any per-file errors. Only
        the latest job is retained — an old jobId, or any jobId from before a
        server restart, raises an error.

        relabelled counts documents whose labels were rewritten from their
        stored text, with nothing re-embedded: classify.toml changed, a move
        put a file under other path rules, or the model answered after a model
        problem had left a facet unanswered. It stays 0 without a classify.toml.
        A labelling problem appears in errors with its message prefixed
        "labels:"; it is not a failure.

        signed counts documents given the signature that finds their
        look-alikes, computed from their stored text with nothing read or
        embedded again: documents indexed before signatures existed, or whose
        signature could not be written when they were ingested. An ingest signs
        its own documents, so it is usually 0. A problem appears in errors
        prefixed "lookAlikes:"; it is not a failure.

        Each sync also writes, from the same stored text, the row that ranks
        a document as a whole in search, for every document left without
        one; a problem appears in errors prefixed "documents:" and is not a
        failure.

        ignored counts the files and folders a .miniragignore rule or a
        RAG_EXCLUDE pattern excluded from the
        scan, a folder once; a file indexed before its rule was added is
        deleted from the index and counted under deleted.

        unreadable counts indexed sources this installation cannot read
        because an optional extra is absent (images need [ocr], and so does
        reading again a PDF that was read with OCR before page numbers were
        stored); each one appears in errors saying it was kept as it is, but
        it is not a failure and does not make the job fail.

        largeTables counts the .xlsx and .csv files the sync would have read
        but left out for holding more data rows than RAG_MAX_TABLE_ROWS; each
        appears in errors as "skipped: ...", and is not a failure. One already
        indexed is deleted from the index and counted under deleted. The rows
        of every table the sync would read again are counted, and of no other,
        so one it leaves as indexed stays indexed whatever its size.

        A folder the scan could not see into (a missing root, one without read
        permission, an empty folder where a root was mounted) also appears in
        errors, and nothing indexed under it is deleted. It counts as failed
        when it is a root, when indexed files are under it, or when the sync
        was started for it or for a path inside it.

        rebuild is null unless the sync rebuilds the index (only a sync without
        a path rebuilds, and only a "rebuild pending" index). From the moment
        the rebuild starts it names kind ("reembed": the stored text embedded
        again; "reingest": the roots read again, which can take hours) and
        reasons (what changed, as status's rebuildReasons named it); once the
        job has finished also generation (the rebuilt index's), reembedded
        (chunks embedded again) and restored (documents the roots could not
        give back — ingested data and URLs, files this installation could not
        read again — stored again from their indexed text, each file among
        them also in errors), which are null until then.
        """
        return live().sync.status(jobId).to_dict()

    @mcp.tool
    @guard
    def ingest_file(filePath: str) -> dict:
        """Ingest or re-ingest one file, replacing any content already indexed for it.

        filePath must be an absolute path inside a configured document root.
        Re-ingesting an already-indexed file discards its old chunks and
        replaces them with freshly parsed ones. With a classify.toml,
        `labelProblem` says which labels the document was stored without,
        and why; a full sync asks again. `warnings`, present only when there
        is something to say, notes that the embedding endpoint is unreachable
        and the in-process model embedded instead: same vectors, slower.

        Refused while the index needs a rebuild (status: indexState "rebuild
        pending"), until an unscoped sync_start rebuilds it, and for a file
        above MAX_FILE_SIZE or a table above RAG_MAX_TABLE_ROWS data rows.
        """
        h = live()
        pipeline = h.ingest_pipeline()
        path = resolve_in_roots(filePath, h.config.roots)
        return ingest_reply(pipeline.ingest_file(path), h.notes())

    @mcp.tool
    @guard
    def ingest_data(data: str, source: str, format: str = "text", title: str | None = None) -> dict:
        """Ingest text/markdown/html content the client holds, under a source id you choose.

        format is one of "text", "markdown", or "html" (default "text").
        source is a stable identifier you pick, not a filesystem path —
        re-using it replaces the previously ingested content for that id,
        so reuse the same source to update an item. `labelProblem`,
        `warnings` and the refusal while a rebuild is pending are as in
        ingest_file.
        """
        h = live()
        return ingest_reply(
            h.ingest_pipeline().ingest_data(data, source=source, fmt=format, title=title),
            h.notes(),
        )

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
        `labelProblem`, `warnings` and the refusal while a rebuild is pending
        are as in ingest_file.
        """
        h = live()
        return ingest_reply(
            h.ingest_pipeline().ingest_url(url, source=source, title=title), h.notes()
        )

    @mcp.tool
    @guard
    def query_documents(
        query: str,
        topK: int = 8,
        scope: str | list[str] | None = None,
        labels: dict[str, str | list[str]] | None = None,
        dateFrom: str | None = None,
        dateTo: str | None = None,
    ) -> dict:
        """Hybrid search: semantic similarity plus a keyword boost for exact terms
        and one for passages of documents that match the query as a whole.

        Returns `results` — ranked chunks with text, source, title,
        chunkIndex, score and parentId — `sources`, the distinct
        sources in rank order each with a hits count and a
        `displayPath` to show the user (`source` stays absolute and is
        what the other tools take), and `parents`, a map from parentId
        to section text. Use `sources` to answer "which documents
        cover this topic" without inspecting individual chunks.

        A `sources` row carries `lookAlikes` when other indexed documents
        overlap it by 0.5 or more: up to 3, most first, each with `source`,
        `displayPath`, `overlap` (0..1, the estimated Jaccard similarity of
        the two texts' five-word runs: runs in common over runs in either,
        so a short text wholly inside a much longer one scores low) and
        `date` when known. They
        come from the whole index, outside scope and filters too. Overlap is
        measured text, not a verdict on which one is current: an older
        version or a sibling filled in from one template can hold other
        figures, so check both before quoting one.

        Results are ranked and stop at topK without saying what they
        left out. For every document that mentions a term, how many do,
        or whether something is mentioned at all, use find_mentions.

        `text` is the passage that matched and that `score` describes.
        `parentId` names the section it sits in — a transcript time
        window, a heading section, a slide, a table. Look it up in
        `parents` to read the whole section when the match alone is too
        small to act on. A chunk indexed before parent sections existed
        has parentId null and no entry; re-sync to fill it in.

        A hit from a PDF file also carries `pages`, the page or pages of
        the file its text came from ("12" or "12-13"); a hit from any other
        source, or whose page is not certain, has none.

        topK must be at least 1 and is capped at 100; a larger value is
        silently clamped to the cap rather than rejected.

        When nothing matches because nothing could — an empty index, or a
        scope no indexed document is under — `notice` says which.

        `embeddingBackend` says what embedded the query: "endpoint",
        "in-process", "in-process (endpoint unreachable)", or "in-process
        (endpoint timed out)" when this query alone was. `warnings`,
        present only when there is something to say, explains the last two:
        the results are the same, only slower to come.

        Results are in reranked order when a reranker is configured:
        `rerankScore` is the cross-encoder's score, `score` the first-stage
        score, and `reranked` says which happened. If the reranker could not
        be reached, `warnings` says so and the order is the first stage's.

        Cite what you take, so the user can verify it, in whatever
        language you answer: end with a Sources list, one line per
        document you actually used, each line just that document's
        `displayPath` copied verbatim — no [n] markers, nothing else
        on the line. Plain text, never a markdown link or file://.

        `labels` ({facet: value or [values]}) and `dateFrom`/`dateTo`
        (YYYY-MM-DD, inclusive) narrow the search to labelled documents when
        the corpus has a classify.toml; `status` lists its facets and values
        under `labels` — map an approximate name to a value from there.
        `filterCounts` then says how many documents in scope were left out
        only for having no label or no date; each `sources` row carries its
        `labels` and `date`.
        """
        c = opened()
        return run_query(
            c.store,
            c.embedder,
            c.config,
            query,
            top_k=topK,
            scopes=_scopes(scope),
            labels=labels,
            date_from=dateFrom,
            date_to=dateTo,
            label_source=c.labels,
            reranker=c.reranker,
        )

    @mcp.tool
    @guard
    def find_mentions(
        terms: str | list[str],
        match: Literal["all", "any"] = "all",
        scope: str | list[str] | None = None,
        labels: dict[str, str | list[str]] | None = None,
        dateFrom: str | None = None,
        dateTo: str | None = None,
        limit: int = DEFAULT_FIND_LIMIT,
        offset: int = 0,
    ) -> dict:
        """List every indexed document that mentions a word or phrase, with counts.

        For "which documents mention X", "list all", "how many", and "is X
        mentioned anywhere": this scans the whole index, where
        query_documents ranks a few passages and stops.

        A term matches where a word begins and its end is open: акт finds
        акта and актов, not контакт — but also актуально. `forms` lists the
        word forms each term matched, with how many documents have each;
        check it before trusting a count. A term wrapped in double quotes
        matches that exact word only, so to narrow, pass the forms you
        want, each quoted, with match="any". Case, ё/е and Unicode
        composition are ignored. The match is literal, so a synonym or
        another spelling is another term. `terms` is one string or a list
        of up to 10; match="all" keeps documents containing every term
        somewhere, match="any" those with at least one, and each row then
        lists its `terms`. Titles are searched too.

        Returns searchedDocuments (indexed documents in scope that pass the
        filters), matchedDocuments, matchedChunks, `forms`, and `documents`:
        title matches (`inTitle`) first, then by `hits`, the chunks that
        mention a term. Each row has `displayPath` and `firstMatch`
        {chunkIndex, parentId, snippet, and `pages` in a PDF as in
        query_documents}, or null when only the title matched. `limit`
        (default 50, capped at 200) pages the list and
        `nextOffset` is present while more remain; the counts always cover
        every match. With no match, `notice` states the absence and what it
        covered. A row carries `lookAlikes` as in query_documents, from the
        whole index: check them before quoting a figure.

        Only indexed text is searched: files not yet synced (list_files
        shows them) and images without OCR are not, and a phrase that a
        chunk boundary splits is not found. scope, labels, dateFrom
        and dateTo filter as in query_documents, and citations follow its
        rule: each document's `displayPath`, verbatim.
        """
        c = opened()
        return run_find(
            c.store,
            c.config.roots,
            terms,
            match=match,
            scopes=_scopes(scope),
            labels=labels,
            date_from=dateFrom,
            date_to=dateTo,
            limit=limit,
            offset=offset,
            label_source=c.labels,
        )

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
        default to 1). A chunk of a PDF carries `pages` as in
        query_documents.
        """
        c = opened()
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
        c = opened()
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
    def list_files(
        scope: str | list[str] | None = None,
        labels: dict[str, str | list[str]] | None = None,
        dateFrom: str | None = None,
        dateTo: str | None = None,
        sort: str | None = None,
    ) -> dict:
        """List files found on disk under the document roots, plus indexed data/url sources.

        Each disk file is reported with a state: "ingested" (index matches
        disk), "stale" (changed on disk since it was indexed),
        "stale_scheme" (unchanged on disk, but indexed under an older
        chunking scheme, so its vectors are not comparable with current
        ones), "stale_pages" (an unchanged PDF indexed before page numbers
        were stored, so its results carry none), "not_ingested" (never
        indexed), or "unreadable" (indexed and still on disk, but of a type
        this installation cannot read because an optional extra is absent —
        images need [ocr]). Everything but
        "ingested" and "unreadable" needs a sync_start; an "unreadable"
        source is kept as indexed and no sync can refresh it here.
        "stale_scheme" is the per-source view of what status reports as
        staleChunkCount. Data and url sources have no disk state to compare
        against, so they are "ingested" or "stale_scheme".

        unlistedFolders names each folder the scan could not see into (no read
        permission, a missing or unmounted drive), with what to do about it.
        Indexed files under one are listed as "unlisted": still indexed and
        searchable, not checked against the disk.

        `ignored`, present when a .miniragignore rule or a RAG_EXCLUDE
        pattern excluded anything, counts the files and folders it left out, a folder once.

        `labels`, `dateFrom` and `dateTo` filter as in query_documents;
        sort="date" lists newest first, documents without a date last. Rows
        carry `labels` and `date` when present.
        """
        c = opened()
        return list_payload(
            c.store,
            c.config.roots,
            _scopes(scope),
            exclude=c.config.exclude,
            labels=labels,
            date_from=dateFrom,
            date_to=dateTo,
            sort=sort,
            label_source=c.labels,
        )

    @mcp.tool
    @guard
    def delete_file(filePath: str | None = None, source: str | None = None) -> dict:
        """Delete an indexed file, data item, or url item from the index.

        Provide exactly one of filePath (absolute path inside a document
        root) or source (the id of a data/url item). This only removes the
        index entry — a file left in place under a document root is
        re-ingested by a later sync_start.
        """
        c = opened()
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

        indexState is "ready", "rebuild pending", "syncing" or "unavailable".
        indexGeneration, embedding (model, prefixes, dim, backend) and
        chunkBudget say what the index was built with, which every query
        follows; `model` is what the configuration asks for. When the two
        differ the index is "rebuild pending": rebuildReasons lists what an
        unscoped sync_start would change, and until then the index answers
        from its existing vectors and refuses ingests. An index that cannot
        be opened (a minirag 1.x index, an unreadable index.json, a model
        that cannot be loaded here) is "unavailable", and indexError says why.

        chunkScheme is the chunking scheme the index is being written with.
        staleChunkCount counts chunks still stored under an older scheme —
        when it is above zero, schemeWarning explains that those chunks need
        a re-sync to be rebuilt.

        embeddingBackend is what embeds: "endpoint", "in-process", or
        "in-process (endpoint unreachable)", when embeddingWarning says why
        and when the endpoint is tried again. embeddingFallback, present only
        then, says the model has no local copy in the cache folder: with an
        endpoint configured it is never downloaded, so every search and
        ingest fails until the endpoint is back.

        ftsLanguage is the stemmer keyword search uses (RAG_FTS_LANGUAGE).
        ftsWarning, present only when the keyword index could not be rebuilt
        after that setting changed, names the stemmer it still uses.
        documentFtsWarning, present only when an index of the whole-document
        ranking could not be created, names the columns it lacks.

        unsignedSourceCount, present when above zero, counts indexed documents
        without the signature that finds their look-alikes, so no result names
        theirs; lookAlikesWarning says a sync computes them.

        undocumentedSourceCount, present when above zero, counts indexed
        documents without the row that lets search rank them as whole
        documents; documentsWarning says a sync builds them.

        unpagedSourceCount, present when above zero, counts PDFs indexed
        before page numbers were stored, whose results carry no `pages`;
        pagesWarning says the next sync reads them again.

        ignoreFiles, present when a root has a .miniragignore: each file with
        its number of rules, or the error that stops scanning under its root.

        excludePatterns, present when RAG_EXCLUDE is set, is the number of
        patterns it holds.

        maxTableRows, present only when RAG_MAX_TABLE_ROWS is set, is the
        number of data rows above which an .xlsx or .csv is not indexed.

        ocr names the OCR engine this install can use, or is "unavailable"
        when the optional extra is missing — scanned PDFs and images cannot
        be indexed then, and ocrHint says what to install. ocrSecondReader,
        present on macOS beside an engine, is "vision" when Apple Vision
        re-reads the pages Tesseract is unsure of (ocrEngine
        "tesseract+vision"), "unavailable" when its bindings are missing, or
        "off" when RAG_OCR_SECOND_READER_BELOW is 0.

        labels, present only when the corpus has a classify.toml: its facets,
        value descriptions and document counts, unlabelled and uncertain
        counts, and the model state, with modelHint saying why when it is
        "unavailable" — or the file's error.
        """
        if handle is None:
            return {"version": __version__, "configError": config_error or "no configuration"}
        return status_payload(config, handle)

    return mcp


def run_server() -> None:
    try:
        config: Config | None = load_config(os.environ)
        error = None
    except ConfigError as e:
        config, error = None, str(e)
    create_app(config, config_error=error, env=os.environ).run()
