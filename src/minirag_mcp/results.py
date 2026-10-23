"""Wire-shape for search results, shared by the MCP server and the CLI.

Both interfaces must describe a hit the same way — divergence here is how the CLI
ended up returning fewer fields than the server for the same query. Anything that
shapes a SearchResult for output belongs in this module, imported by both sides.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from minirag_mcp import literal, ocr
from minirag_mcp.chunker import join_parent, shared_lines
from minirag_mcp.config import Config
from minirag_mcp.ingest.scanner import FileState, compute_states, scan_roots
from minirag_mcp.labels.filters import (
    LabelFilter,
    LabelFilterError,
    facet_applies,
    filter_counts,
    filter_notice,
    parse_filter,
    source_matches,
    to_sql,
)
from minirag_mcp.labels.labeler import Labeler, LabelerSource
from minirag_mcp.labels.model import INSTALL_HINT
from minirag_mcp.labels.resolve import decode_labels, document_fields, root_relative
from minirag_mcp.scope import is_under
from minirag_mcp.store import (
    LABEL_COLUMNS,
    MAX_TOP_K,
    RESYNC_HINT,
    SearchResult,
    SourceInfo,
    Store,
)

EMPTY_INDEX_NOTICE = (
    "Nothing is indexed yet, so no query can match. Index the document roots first: "
    "sync_start, or `minirag-mcp sync`."
)
FILTERED_OUT_NOTICE = (
    "No indexed document in scope passes the label and date filters, so nothing was "
    "searched; filterCounts says how many were left out and why."
)
MAX_TERMS = 10
# Documents per find_mentions page. Every match is still counted; the rest is paged.
DEFAULT_FIND_LIMIT = 50
MAX_FIND_LIMIT = 200
SNIPPET_CHARS = 200
# Word forms reported per term: enough to show what a stem caught besides the word meant.
MAX_FORMS = 20
_SNIPPET_BEFORE = 60
_SPACE = re.compile(r"\s")


class QueryError(ValueError):
    """A query that cannot run as given."""


def result_dict(r: SearchResult) -> dict:
    """One search hit as its JSON object. Keys are camelCase, matching the MCP tools.

    `text` stays the retrieval unit — the thing that was embedded and the thing `score`
    describes. The enclosing section is named by `parentId` and its text lives once in
    the response's `parents` map, not once per hit: a good chunking scheme puts several
    hits of one query in the same section, and attaching the section to every hit made
    about a third of a response the same words resent.
    """
    return {
        "text": r.text,
        "source": r.source,
        "title": r.title,
        "chunkIndex": r.chunk_index,
        "score": r.score,
        "distance": r.distance,
        "parentId": r.parent_id or None,
    }


def parent_map(results: Iterable[SearchResult]) -> dict[str, str]:
    """The enclosing sections of `results`, one entry each, keyed by `parentId`.

    A map rather than a per-hit field, and rather than "on the first hit only": a
    client cannot tell "this hit has no section" from "look it up on an earlier hit"
    when both are a missing value, and a reader of the tool description should not have
    to. A hit whose section could not be rebuilt — a chunk from an index predating
    parent sections — has no entry here and a null `parentId`.
    """
    out: dict[str, str] = {}
    for r in results:
        if r.parent_id and r.parent_text is not None:
            out.setdefault(r.parent_id, r.parent_text)
    return out


def join_document(chunks: Sequence[SearchResult]) -> str:
    """A source's indexed text, rebuilt for reading rather than concatenated.

    Every retrieval unit repeats whatever context its own vector needed — a heading
    breadcrumb, a time-window label, a table's header row — so joining the units
    verbatim prints that context once per unit. Measured on the real corpus, that
    inflated the document by 22% at the median and 2.64x at the tail, and for a table it
    plants a header row in the middle of the rows. `join_parent` already knows how to
    drop the leading lines a unit shares with the one before it, so this applies it per
    section and joins the sections.

    A chunk with no `parent_id` — an index predating parent sections — is its own group,
    so nothing is ever deduplicated across two chunks that were never one section.
    """
    groups: list[list[str]] = []
    current: str | None = None
    for chunk in chunks:
        parent = chunk.parent_id or None
        if parent is None or parent != current:
            groups.append([])
            current = parent
        groups[-1].append(chunk.text)
    return "\n\n".join(text for text in (join_parent(g) for g in groups) if text)


def scheme_status(store: Store) -> dict:
    """The chunking-scheme half of `status`, identical for the server and the CLI.

    The count is measured against the index, not inferred from a version number the
    tool happens to be running: an index part-way through a re-sync holds chunks from
    both schemes, and only counting says so.
    """
    stale = store.stale_chunk_count()
    out: dict = {"staleChunkCount": stale}
    if stale:
        out["schemeWarning"] = (
            f"{stale} chunk(s) predate the current chunking scheme. {RESYNC_HINT}"
        )
    return out


def ocr_status() -> dict:
    """The OCR half of `status`, identical for the server and the CLI.

    Measured, not configured: the extra is either importable in this interpreter or it
    is not, and the whole of the feature turns on that one answer. Nothing here reads
    the index, so the server reports it alongside the fields that do not need one.
    """
    if ocr.available():
        return {"ocr": ocr.ENGINE_RAPIDOCR}
    return {
        "ocr": "unavailable",
        "ocrHint": f"scanned PDFs and images cannot be indexed; {ocr.INSTALL_HINT}",
    }


def display_path(source: str, roots: Sequence[Path] = ()) -> str:
    """`source` as the user would name the file: its path relative to the root holding it.

    A display string, and only that. `source` remains the identity key — `read_file`,
    `delete_file` and re-ingest all address a document by it — so the two live side by
    side rather than one replacing the other.

    The relative path rather than the `title`, because the title is derived: underscores
    become spaces and the extension is dropped, so `И-112_ЗПС_Хранение ТМЗ.docx` reaches
    the user as `И-112 ЗПС Хранение ТМЗ`, which names nothing on disk. What is wanted is
    the filename exactly as it is, in enough of its directory to be unambiguous.

    Computed by slicing the string, not by re-rendering a `Path`: the result must be a
    byte-wise suffix of `source`, since a normaliser could change a separator or a
    Unicode composition and still look right. The trailing separator on the prefix is
    what keeps a root of `/data` from claiming `/data_archive/x.md`, and the longest
    matching root wins so that nested roots do not make the answer depend on the order
    they were configured in.

    Not normalised, deliberately, even though it would raise a number. APFS stores these
    filenames decomposed — the `й` in `…показателей…` is `и` plus a combining breve — and
    a model reproduces them composed, so about one delivered entry in ten differs from
    what was served by exactly that and nothing else. The two are canonically
    equivalent: the same filename, opening the same file, identical on screen. Composing
    here would make the strings compare equal at the cost of handing the user a name
    that is not the one on disk, which is the trade this field exists to refuse.

    Falls back to `source` itself — never to an empty string — when nothing matches: a
    `data` or `url` item has no filesystem path at all (its `source` is the id it was
    ingested under, which for a URL is the URL), and a file can outlive the root it was
    indexed under. An absolute path is worse to read than a relative one and far better
    than leaving the caller with nothing to print.
    """
    prefix = ""
    for root in roots:
        candidate = f"{str(root).rstrip('/')}/"
        if source.startswith(candidate) and len(candidate) > len(prefix):
            prefix = candidate
    return source[len(prefix) :] if prefix else source


def aggregate_sources(results: Iterable[SearchResult], roots: Sequence[Path] = ()) -> list[dict]:
    """Distinct sources among `results`, in rank order, each with a hit count.

    The first hit of a source fixes that source's position, so the list answers
    "which documents cover this topic" without inspecting individual chunks.

    Each row also carries `displayPath`, the string an answer's Sources list is written
    from. It hangs here rather than on a hit because a citation names a document, and
    this deduplicated list is the document list already.

    `roots` are the configured document roots; a caller with none gets `source` echoed
    back, which is what an unrooted source would resolve to anyway.
    """
    sources: dict[str, dict] = {}
    for r in results:
        agg = sources.setdefault(
            r.source,
            {
                "source": r.source,
                "title": r.title,
                "hits": 0,
                "displayPath": display_path(r.source, roots),
                **document_fields(r.labels_json, r.doc_date, r.doc_date_source),
            },
        )
        agg["hits"] += 1
    return list(sources.values())


def empty_result_notice(store: Store, scopes: Sequence[str]) -> str | None:
    """Why a search came back empty, when the reason is not the query's wording.

    Without it an empty index and a scope that excludes everything both answer
    `results: []`, which reads exactly like a corpus that does not cover the question —
    and the caller's next move for those (reword, or conclude it is not there) is the
    wrong one for both.
    """
    if store.chunk_count() == 0:
        return EMPTY_INDEX_NOTICE
    if scopes and not store.list_sources(scopes=tuple(scopes)):
        return (
            f"No indexed document is under scope {', '.join(scopes)}. A scope is matched "
            "against `source`, which for a file is its absolute path, not against "
            "`displayPath`; pass an absolute folder path, as in a result's `source`."
        )
    return None


def run_query(
    store: Store,
    embedder,
    config: Config,
    query: str,
    *,
    top_k: int = 8,
    scopes: Sequence[str] = (),
    labels: Mapping[str, str | Sequence[str]] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    label_source: LabelerSource | None = None,
) -> dict:
    """`query_documents` and `minirag-mcp query`, as one implementation.

    Two copies of this had drifted: the CLI searched for an empty string the server
    refused. With one, the two cannot disagree on what they accept or on what an empty
    result is said to mean.
    """
    if not query.strip():
        raise QueryError("query must not be empty")
    if top_k < 1:
        raise QueryError(f"topK must be at least 1, got {top_k}")
    scopes = tuple(scopes)
    label_filter, labeler = _parse_filter(store, labels, date_from, date_to, label_source)
    results = store.search(
        query,
        embedder.embed_query(query),
        top_k=min(top_k, MAX_TOP_K),
        hybrid_weight=config.hybrid_weight,
        scopes=scopes,
        max_distance=config.max_distance,
        grouping=config.grouping,
        max_files=config.max_files,
        filter_sql=to_sql(label_filter) if label_filter else None,
    )
    payload = {
        "results": [result_dict(r) for r in results],
        "sources": aggregate_sources(results, config.roots),
        "parents": parent_map(results),
    }
    if label_filter is not None and labeler is not None:
        payload["filterCounts"] = filter_counts(
            label_filter, store.list_sources(scopes=scopes), labeler.taxonomy, config.roots
        )
        if notice := filter_notice(label_filter, labeler):
            payload["filterNotice"] = notice
    if not results and (notice := empty_result_notice(store, scopes)):
        payload["notice"] = notice
    return payload


@dataclass
class _Mentions:
    """What one document's matching chunks add up to."""

    first: SearchResult
    in_title: bool
    terms: set[int] = field(default_factory=set)
    forms: dict[int, set[str]] = field(default_factory=dict)
    hits: int = 0
    first_match: dict | None = None

    def add(self, found: Sequence[literal.Match | None]) -> list[literal.Match]:
        for i, m in enumerate(found):
            if m is not None:
                self.terms.add(i)
                self.forms.setdefault(i, set()).update(m.forms)
        return [m for m in found if m is not None]


def _snippet(text: str, span: tuple[int, int]) -> str:
    """A window of `text` around `span`, cut at whitespace, whitespace collapsed."""
    start, end = span
    lo = max(0, start - _SNIPPET_BEFORE)
    hi = min(len(text), lo + SNIPPET_CHARS)
    if lo > 0 and (m := _SPACE.search(text, lo, start)):
        lo = m.end()
    if hi < len(text) and (tail := list(_SPACE.finditer(text, min(end, hi), hi))):
        hi = tail[-1].start()
    body = " ".join(text[lo:hi].split())
    return ("…" if lo > 0 else "") + body + ("…" if hi < len(text) else "")


def _parse_terms(terms: str | Sequence[str]) -> list[literal.Term]:
    raw = [terms] if isinstance(terms, str) else list(terms)
    if not raw:
        raise QueryError("terms must not be empty")
    if len(raw) > MAX_TERMS:
        raise QueryError(f"at most {MAX_TERMS} terms, got {len(raw)}")
    try:
        return [literal.term(t) for t in raw]
    except ValueError as e:
        raise QueryError(str(e)) from e


def _absence_notice(searched: int, terms: Sequence[literal.Term], match: str) -> str:
    shown = ", ".join(t.text if t.exact else f'"{t.text}"' for t in terms)
    what = shown if len(terms) == 1 else f"{match} of {shown}"
    head = (
        f"None of the {searched} documents searched contains {what}"
        if searched != 1
        else f"The one document searched does not contain {what}"
    )
    if all(t.exact for t in terms):
        how = "as a whole word" if len(terms) == 1 else "as whole words"
    elif any(t.exact for t in terms):
        how = "(quoted terms as whole words, the others from the start of a word)"
    else:
        how = "from the start of a word"
    return (
        f"{head} {how}, ignoring case, ё/е and Unicode composition. "
        "The match is literal: another word or spelling for the same thing is not found, "
        "so try a shorter stem or a synonym, or query_documents to search by meaning."
    )


def run_find(
    store: Store,
    roots: Sequence[Path],
    terms: str | Sequence[str],
    *,
    match: str = "all",
    scopes: Sequence[str] = (),
    labels: Mapping[str, str | Sequence[str]] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = DEFAULT_FIND_LIMIT,
    offset: int = 0,
    label_source: LabelerSource | None = None,
) -> dict:
    """`find_mentions` and `minirag-mcp find`, as one implementation.

    Every document is counted, however many match; only the listing is paged. The
    counts are the point: "how many", "all of them" and "none" are the questions ranked
    search cannot answer, because it stops at top-k without saying so.
    """
    parsed = _parse_terms(terms)
    if match not in ("all", "any"):
        raise QueryError(f"match must be 'all' or 'any', got {match!r}")
    if limit < 1:
        raise QueryError(f"limit must be at least 1, got {limit}")
    if offset < 0:
        raise QueryError(f"offset must not be negative, got {offset}")
    limit = min(limit, MAX_FIND_LIMIT)
    scopes = tuple(scopes)
    label_filter, labeler = _parse_filter(store, labels, date_from, date_to, label_source)
    filter_sql = to_sql(label_filter) if label_filter else None

    # Read before the scan, so a sync landing in between cannot make matched exceed it.
    searched = len(store.list_sources(scopes=scopes, filter_sql=filter_sql))
    docs: dict[str, _Mentions] = {}
    previous: tuple[str, int, list[str]] | None = None
    for chunk in store.literal_candidates(parsed, scopes=scopes, filter_sql=filter_sql):
        doc = docs.get(chunk.source)
        if doc is None:
            doc = docs[chunk.source] = _Mentions(chunk, in_title=False)
            doc.in_title = bool(doc.add(literal.matches(parsed, chunk.title)))
        # A header repeated from the chunk before is that chunk's mention, not a new
        # one. When that chunk is not a candidate, its header holds no term anyway.
        lines = chunk.text.split("\n")
        own = lines
        if previous is not None and previous[:2] == (chunk.source, chunk.chunk_index - 1):
            own = lines[shared_lines(previous[2], lines) :]
        previous = (chunk.source, chunk.chunk_index, lines)
        text = "\n".join(own)
        found = doc.add(literal.matches(parsed, text))
        if not found:
            continue
        doc.hits += 1
        if doc.first_match is None:
            span = min(m.span for m in found)
            doc.first_match = {
                "chunkIndex": chunk.chunk_index,
                "parentId": chunk.parent_id or None,
                "snippet": _snippet(unicodedata.normalize("NFC", text), span),
            }

    needed = len(parsed) if match == "all" else 1
    matched = [d for d in docs.values() if len(d.terms) >= needed]
    rows = []
    for d in matched:
        row = {
            "source": d.first.source,
            "displayPath": display_path(d.first.source, roots),
            "title": d.first.title,
            "hits": d.hits,
            "inTitle": d.in_title,
        }
        if len(parsed) > 1:
            row["terms"] = [parsed[i].text for i in sorted(d.terms)]
        row["firstMatch"] = d.first_match
        row.update(document_fields(d.first.labels_json, d.first.doc_date, d.first.doc_date_source))
        rows.append(row)
    rows.sort(key=lambda r: (not r["inTitle"], -r["hits"], r["displayPath"]))
    forms: dict[str, dict[str, int]] = {}
    for i, t in enumerate(parsed):
        counts: dict[str, int] = {}
        for d in matched:
            for form in d.forms.get(i, ()):
                counts[form] = counts.get(form, 0) + 1
        if counts:
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            forms[t.text] = dict(ranked[:MAX_FORMS])

    payload: dict = {
        "searchedDocuments": searched,
        "matchedDocuments": len(rows),
        "matchedChunks": sum(d.hits for d in matched),
        "forms": forms,
        "documents": rows[offset : offset + limit],
    }
    if offset + limit < len(rows):
        payload["nextOffset"] = offset + limit
    if label_filter is not None and labeler is not None:
        payload["filterCounts"] = filter_counts(
            label_filter, store.list_sources(scopes=scopes), labeler.taxonomy, roots
        )
        if notice := filter_notice(label_filter, labeler):
            payload["filterNotice"] = notice
    if not rows:
        if searched:
            payload["notice"] = _absence_notice(searched, parsed, match)
        else:
            payload["notice"] = empty_result_notice(store, scopes) or FILTERED_OUT_NOTICE
    return payload


def _parse_filter(
    store: Store,
    labels: Mapping[str, str | Sequence[str]] | None,
    date_from: str | None,
    date_to: str | None,
    label_source: LabelerSource | None,
) -> tuple[LabelFilter | None, Labeler | None]:
    """The filter a request asks for, as a QueryError when it cannot run.

    Refused before the search because an older index opened without write access has no
    label columns, and a filter over a column that is not there fails inside the SQL
    engine with a message that names neither the cause nor the remedy.
    """
    labeler, error = label_source.get() if label_source else (None, None)
    try:
        label_filter = parse_filter(
            labels,
            date_from,
            date_to,
            labeler=labeler,
            error=error,
            taxonomy_file=label_source.path if label_source else None,
        )
    except LabelFilterError as e:
        raise QueryError(str(e)) from e
    if label_filter is not None and (set(store.missing_columns) & set(LABEL_COLUMNS)):
        raise QueryError(
            "this index has no label columns and they could not be added: it was opened "
            "without write access. Reopen it with write access (run `minirag-mcp sync` "
            "once) before filtering by label or date."
        )
    return label_filter, labeler


def labels_status(source: LabelerSource | None, store: Store | None) -> dict:
    """The labels half of `status`, identical for the server and the CLI.

    Empty when the feature is off, so a corpus without a taxonomy file sees no new key.
    Whatever goes wrong in here is a labels error: both callers report an exception from
    `status` as `indexError`, which would name an index that is fine.
    """
    if source is None:
        return {}
    try:
        return _labels_status(source, store)
    except Exception as e:
        return {"labels": {"file": str(source.path), "error": f"{type(e).__name__}: {e}"}}


def _labels_status(source: LabelerSource, store: Store | None) -> dict:
    labeler, error = source.get()
    if error is not None:
        return {"labels": {"file": str(source.path), "error": error}}
    if labeler is None:
        return {}
    taxonomy = labeler.taxonomy
    infos = store.list_sources() if store is not None else []
    state = labeler.model_state()
    out: dict = {"file": str(source.path), "model": state}
    if state == "unavailable":
        out["modelHint"] = INSTALL_HINT
    facets: dict = {}
    for facet in taxonomy.facets:
        counts = dict.fromkeys(facet.value_ids, 0)
        unlabelled = uncertain = 0
        for info in infos:
            lab = decode_labels(info.labels_json).get(facet.name)
            if lab is None:
                if facet_applies(facet, root_relative(info.source, labeler.roots)):
                    unlabelled += 1
                continue
            if lab.value in counts:
                counts[lab.value] += 1
            uncertain += lab.uncertain
        facets[facet.name] = {
            "question": facet.question,
            "values": {
                v.id: {"description": v.description, "count": counts[v.id]} for v in facet.values
            },
            "unlabelled": unlabelled,
            "uncertain": uncertain,
        }
    out["facets"] = facets
    out["undated"] = sum(1 for info in infos if not info.doc_date)
    return {"labels": out}


def _unlisted_message(error: str | None, indexed: int) -> str:
    files = f"{indexed} indexed file{'' if indexed == 1 else 's'}"
    reach = "Make it readable, or reconnect the drive it is on, to list it."
    if error is None:
        return (
            f"this root is an empty folder, but the index holds {files} under it, listed "
            'as "unlisted"; an unmounted volume looks like this. Mount it to list them, '
            "or delete their entries if they are gone."
        )
    if indexed == 0:
        return f"could not list this folder ({error}), so this listing is missing it. {reach}"
    return (
        f"could not list this folder ({error}); {files} under it listed as "
        f'"unlisted": still indexed and searchable, not checked against the disk. {reach}'
    )


def list_payload(
    store: Store,
    roots: Sequence[Path],
    scopes: Sequence[str] = (),
    *,
    labels: Mapping[str, str | Sequence[str]] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort: str | None = None,
    label_source: LabelerSource | None = None,
) -> dict:
    """`list_files` and `minirag-mcp list`, as one implementation.

    `unlistedFolders` names every folder the scan could not see into, because a listing
    that silently leaves out what is under it contradicts `status` and search, which
    still count and return what is indexed there.

    A filter keeps only documents whose stored labels match. A file not yet ingested has
    none, so it drops out, and is counted as unlabelled for the facets that reach it
    rather than vanishing silently.
    """
    if sort not in (None, "date"):
        raise QueryError(f"sort must be 'date', got {sort!r}")
    label_filter, labeler = _parse_filter(store, labels, date_from, date_to, label_source)
    blind: list[tuple[Path, str | None]] = []
    entries = scan_roots(roots, blind=blind)
    if scopes:
        entries = [e for e in entries if any(is_under(str(e.path), p) for p in scopes)]
        blind = [
            (folder, error)
            for folder, error in blind
            if any(is_under(str(folder), p) or is_under(p, str(folder)) for p in scopes)
        ]
    infos = store.list_sources(scopes=tuple(scopes))
    by_source = {i.source: i for i in infos}
    states: list[FileState] = compute_states(entries, infos, blind)
    folders = []
    for folder, error in blind:
        indexed = sum(
            1 for st in states if st.state == "unlisted" and is_under(st.source, str(folder))
        )
        if error is None and indexed == 0:
            continue  # an empty root with nothing indexed under it is just empty
        folders.append(
            {
                "path": str(folder),
                "error": error,
                "indexedCount": indexed,
                "message": _unlisted_message(error, indexed),
            }
        )
    rows = [
        {
            "source": st.source,
            "sourceType": st.source_type,
            "title": st.title,
            "state": st.state,
            "chunkCount": st.chunk_count,
            "ocrEngine": st.ocr_engine,
            **_fields_of(by_source.get(st.source)),
        }
        for st in states
    ]
    payload: dict = {"files": rows, "unlistedFolders": folders}
    if label_filter is not None and labeler is not None:
        counts = filter_counts(label_filter, infos, labeler.taxonomy, roots)
        pending = [
            SourceInfo(st.source, st.source_type, "", 0, "", 0.0)
            for st in states
            if st.state == "not_ingested"
        ]
        # Dates are not part of this count: a file never read has no date to be missing.
        counts["unlabelled"] += filter_counts(
            LabelFilter(label_filter.labels), pending, labeler.taxonomy, roots
        )["unlabelled"]
        payload["files"] = [
            row
            for row in rows
            if (info := by_source.get(row["source"])) is not None
            and source_matches(label_filter, info)
        ]
        payload["filterCounts"] = counts
        if notice := filter_notice(label_filter, labeler):
            payload["filterNotice"] = notice
    if sort == "date":
        payload["files"] = sorted(
            payload["files"],
            key=lambda row: row.get("date", {}).get("value", ""),
            reverse=True,
        )
        # Stable under reverse=True, and "" sorts lowest, so undated rows come last.
    return payload


def _fields_of(info: SourceInfo | None) -> dict:
    if info is None:
        return {}
    return document_fields(info.labels_json, info.doc_date, info.doc_date_source)
