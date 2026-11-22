"""LanceDB persistence: a `chunks` table with BM25 FTS indexes, a `documents` table that
ranks whole documents by BM25, and look-alike signatures."""

from __future__ import annotations

import statistics
import warnings
from collections.abc import Callable, Container, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import NamedTuple

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS

from minirag_mcp import literal
from minirag_mcp.chunker import SCHEME_VERSION, join_document, join_parent
from minirag_mcp.config import DEFAULT_FTS_LANGUAGE
from minirag_mcp.identity import rebuild_running, staging_dir, writing
from minirag_mcp.lookalike import NUM_PERM, LookAlike, Signature, SignatureIndex, signature
from minirag_mcp.scope import is_under, sql_clause, sql_str

TABLE = "chunks"
SIGNATURE_TABLE = "signatures"
# One row per source: its whole text, joined from its chunks as `read_file` joins them.
DOCUMENT_TABLE = "documents"
# Documents a query ranks as a whole; a chunk of any document past them gets no bonus.
DOCUMENT_FETCH = 100
# Sources a backfill writes per commit.
_SIGN_BATCH = 64
# Columns carrying BM25 indexes. `title` is indexed because for many document sets the
# filename is where the document code and subject live, and nowhere else.
FTS_COLUMNS = ("text", "title")
# The stemmer language the FTS indexes were last built with. LanceDB does not expose it
# back, and without a record a changed RAG_FTS_LANGUAGE could not be told from an old one.
FTS_LANGUAGE_FILENAME = ".fts-language"
_LIST_LIMIT = 2**31 - 1  # LanceDB scalar queries default to limit 10 — always set explicitly
RRF_K = 60  # standard RRF damping constant
GAP_FACTOR = 2.0  # relevance_cutoff boundary must exceed mean gap by this factor
# Largest top_k the query interfaces will pass down. `search` fetches a multiple of
# top_k from both the vector and the keyword side, so an unbounded top_k is an
# unbounded scan. Callers clamp rather than reject: an oversized request is a bad
# guess at how much context is useful, not an error.
MAX_TOP_K = 100
# Label and date columns. Labels are one string rather than a list column: a list<string>
# cannot be added to an existing table through the SQL-expression migration. `label_keys`
# is the `|facet=value|` form that `contains()` filters match; `labels_json` is what is
# shown; an empty `label_hash` marks a document the labeling pass has yet to visit.
LABEL_COLUMNS = ("labels_json", "label_keys", "label_hash", "doc_date", "doc_date_source")


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    source: str
    source_type: str  # "file" | "data" | "url"
    title: str
    chunk_index: int
    text: str
    vector: list[float]
    file_hash: str
    mtime: float
    ingested_at: str
    # Chunks cut from the same parent section share this; the section's text is
    # rebuilt from them rather than stored a second time.
    parent_id: str = ""
    # Which OCR backend produced this chunk's text, "" when none was used. Only the
    # ingest pipeline sets this; the store's job is just to carry it through storage
    # and listings unchanged.
    ocr_engine: str = ""
    scheme_version: int = SCHEME_VERSION
    labels_json: str = ""
    label_keys: str = ""
    label_hash: str = ""
    doc_date: str = ""
    doc_date_source: str = ""
    # The PDF pages the chunk came from, "12" or "12-13"; "" for other formats and where
    # the page is not known. A row written before the column existed reads as None.
    pages: str = ""


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    title: str
    chunk_index: int
    score: float
    distance: float | None
    parent_id: str = ""
    # The enclosing section, when one could be rebuilt. Additive on purpose: `text`
    # keeps meaning "the chunk that matched", so a client already reading it is
    # unaffected by the change.
    parent_text: str | None = None
    labels_json: str = ""
    doc_date: str = ""
    doc_date_source: str = ""
    pages: str = ""
    # The cross-encoder's raw score; None for a hit that was not reranked. `score` stays
    # the first-stage fused score.
    rerank_score: float | None = None


# The served reranker has 2048-token slots; this many characters of Russian or English
# stay under that with the query. One passage over the limit fails the whole call.
RERANK_PASSAGE_CHARS = 4000


def _passage(text: str) -> str:
    """`text` cut to RERANK_PASSAGE_CHARS at whitespace; only what the cross-encoder sees."""
    if len(text) <= RERANK_PASSAGE_CHARS:
        return text
    cut = text[:RERANK_PASSAGE_CHARS]
    head, sep, _ = cut.rpartition(" ")
    for ws in ("\n", "\t"):
        h, s2, _ = cut.rpartition(ws)
        if len(h) > len(head):
            head, sep = h, s2
    return head if sep and head else cut


def _rerank(
    results: list[SearchResult],
    query_text: str,
    reranker: Callable[[str, list[str]], list[float]],
    rerank_top: int,
) -> list[SearchResult]:
    head, tail = results[:rerank_top], results[rerank_top:]
    try:
        scores = [float(x) for x in reranker(query_text, [_passage(r.text) for r in head])]
    except Exception:
        return results
    if len(scores) != len(head):
        return results
    scored = [replace(r, rerank_score=x) for r, x in zip(head, scores, strict=True)]
    scored.sort(key=lambda r: -r.rerank_score)  # stable: ties keep fused order
    return scored + tail


@dataclass(frozen=True)
class SourceInfo:
    source: str
    source_type: str
    title: str
    chunk_count: int
    file_hash: str
    mtime: float
    # Oldest chunking scheme among this source's chunks. Below SCHEME_VERSION means
    # the source needs re-ingesting even though its bytes on disk are unchanged.
    scheme_version: int = SCHEME_VERSION
    # First non-empty OCR engine among this source's chunks, "" when none used OCR.
    ocr_engine: str = ""
    # More rows than distinct chunk indexes: two first ingests of the file raced and
    # both inserted. Its bytes still match, so only this flag gets it re-read.
    duplicated: bool = False
    labels_json: str = ""
    label_hash: str = ""
    doc_date: str = ""
    doc_date_source: str = ""
    # False when a chunk of this source was stored before page numbers were.
    pages_recorded: bool = True
    label_keys: str = ""
    ingested_at: str = ""


_META_COLS = [
    "source",
    "source_type",
    "title",
    "chunk_index",
    "file_hash",
    "mtime",
    "scheme_version",
    "ocr_engine",
    "labels_json",
    "label_keys",
    "label_hash",
    "doc_date",
    "doc_date_source",
    "pages",
    "ingested_at",
]


def relevance_cutoff(distances: Sequence[float], mode: str) -> int:
    """Identify natural gaps in distance ordering for clustering into relevance groups.

    A boundary occurs where gap > mean(gaps) * GAP_FACTOR. This avoids spurious
    cuts from jitter and detects only materially significant distance jumps.

    Args:
        distances: distances sorted ascending (callers must sort; the gap math is
            meaningless on an arbitrary order such as RRF rank)
        mode: "similar" → cut at first boundary; "related" → cut at second

    Returns:
        Index to cut the list at (len(distances) if no boundary found)
    """
    n = len(distances)
    if n < 3:
        return n
    gaps = [distances[i + 1] - distances[i] for i in range(n - 1)]
    mean = statistics.fmean(gaps)
    threshold = mean * GAP_FACTOR
    if threshold <= 0:
        return n  # degenerate (all-equal or unsorted input): no meaningful boundary
    boundaries = [i + 1 for i, g in enumerate(gaps) if g > threshold]
    if not boundaries:
        return n
    if mode == "similar":
        return boundaries[0]
    return boundaries[1] if len(boundaries) >= 2 else n


def _weighted_rrf(
    vector_ids: Sequence[str], fts_ids: Sequence[str], keyword_weight: float, k: int = RRF_K
) -> dict[str, float]:
    """Fuse two ranked id lists by reciprocal rank.

    Rank-based fusion sidesteps the incomparable scales of L2 distance and BM25:
    only positions matter. keyword_weight is the FTS side's share (0..1).

    Args:
        vector_ids: id list from vector search (ranked by distance)
        fts_ids: id list from FTS search (ranked by BM25)
        keyword_weight: how much to weight FTS results (0 = pure vector, 1 = pure FTS)
        k: RRF damping constant (default 60)

    Returns:
        dict mapping id → fused score, higher is better
    """
    scores: dict[str, float] = {}
    for rank, rid in enumerate(vector_ids):
        scores[rid] = scores.get(rid, 0.0) + (1.0 - keyword_weight) / (k + rank + 1)
    for rank, rid in enumerate(fts_ids):
        scores[rid] = scores.get(rid, 0.0) + keyword_weight / (k + rank + 1)
    return scores


def _is_missing_table(exc: Exception) -> bool:
    """True only for lancedb's "no such table" signal, not for any other open failure.

    lancedb 0.36 raises ValueError("Table 'chunks' was not found"), and
    FileNotFoundError on the code path that checks the dataset directory. A corrupt
    manifest or an unreadable directory arrives as RuntimeError instead.
    """
    return isinstance(exc, FileNotFoundError) or "was not found" in str(exc)


def _and_clauses(*clauses: str | None) -> str | None:
    """AND the non-empty clauses, each parenthesised: the scope clause is an OR chain."""
    present = [c for c in clauses if c]
    if not present:
        return None
    return " AND ".join(f"({c})" for c in present)


def _label_fields(row: dict) -> dict[str, str]:
    """SourceInfo's label fields; an index whose columns could not be added has none."""
    return {
        name: row.get(name) or ""
        for name in ("labels_json", "label_keys", "label_hash", "doc_date", "doc_date_source")
    }


def _signature_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source", pa.string()),
            # The `ingested_at` of the chunks the signature was computed from. Every ingest
            # stamps its chunks afresh, so a signature is current exactly when it matches.
            pa.field("ingested_at", pa.string()),
            pa.field("signature", pa.list_(pa.uint32(), NUM_PERM)),
            pa.field("shingles", pa.int64()),
        ]
    )


def _document_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("source", pa.string()),
            pa.field("title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("file_hash", pa.string()),
            # The `ingested_at` of the chunks the row was joined from, as for signatures.
            pa.field("ingested_at", pa.string()),
        ]
    )


def _named(sources: Sequence[str]) -> str:
    return sources[0] if len(sources) == 1 else f"{len(sources)} documents"


def _in_clause(sources: Iterable[str]) -> str:
    values = ", ".join(f"'{sql_str(s)}'" for s in sources)
    return f"source IN ({values})"


class _Unit(NamedTuple):
    text: str
    parent_id: str


class _DocumentGaps(NamedTuple):
    # Indexed sources without a row of the stamp their chunks carry, each with that
    # stamp, or None when they carry more than one: two first ingests raced.
    undocumented: dict[str, str | None]
    # (source, stamp) of each row whose source has no chunks left.
    orphans: list[tuple[str, str]]


class _StoredChunk(NamedTuple):
    """The fields of a stored chunk that a documents row is built from."""

    chunk_index: int
    text: str
    parent_id: str
    title: str
    file_hash: str
    ingested_at: str


@dataclass(frozen=True)
class _Signatures:
    """What the signature table says about the chunks table, at one pair of versions."""

    index: SignatureIndex
    dates: dict[str, tuple[str, str]]
    # Indexed sources without a current signature, each with the stamp its chunks carry,
    # or None when they carry more than one: two first ingests raced and both inserted.
    unsigned: dict[str, str | None]


class DimensionMismatchError(Exception):
    """An existing index was built with a different embedding dimension than requested."""


class IndexBusyError(Exception):
    """A write refused because the index is being rebuilt, or was replaced meanwhile."""


def _refused_while_rebuilt(write):
    """`write` to the store, held off a swap and the start of a rebuild while it runs
    (`writing`), and refused with IndexBusyError while a rebuild of the store runs: the
    rebuilt index replaces this one, and the write would be lost with it. Then the
    store's `write_check` runs, under the same lock, so no swap can land between it and
    the write."""

    @wraps(write)
    def guarded(self: Store, *args, **kwargs):
        with writing(self.db_path):
            if rebuild_running(self.db_path):
                raise IndexBusyError(
                    f"{self.db_path} is being rebuilt into {staging_dir(self.db_path)}; writes "
                    "to it are refused until the rebuild finishes, since the rebuilt index "
                    "replaces it. Retry once sync_status reports the sync finished."
                )
            if self.write_check is not None:
                self.write_check()
            return write(self, *args, **kwargs)

    return guarded


# Columns added after the first release. An index created before them is migrated in
# place on open; the SQL expression is the value every pre-existing row gets, and
# scheme_version 0 is precisely the marker that says "cut by an older scheme".
#
# The split is by what their absence costs. Without the scheme columns nothing can tell
# an old chunk from a new one or name the section a chunk belongs to, so every reader
# has to assume the worst; an additive column is carried through storage and read by
# nobody who judges a chunk's age, so its absence costs only itself.
_SCHEME_COLUMNS = {
    "parent_id": "''",
    "scheme_version": "CAST(0 AS INT)",
}
_ADDITIVE_COLUMNS = {
    "ocr_engine": "''",
    **dict.fromkeys(LABEL_COLUMNS, "''"),
    # NULL, not '': a row that predates page numbers is told apart from one whose page
    # is not known.
    "pages": "CAST(NULL AS STRING)",
}
_ADDED_COLUMNS = {**_SCHEME_COLUMNS, **_ADDITIVE_COLUMNS}


def _pages_recorded(rows: Iterable[dict]) -> bool:
    return all(row.get("pages") is not None for row in rows)


def _scheme_of(row: dict) -> int:
    """A row's chunking scheme. A row with no such column predates every scheme."""
    value = row.get("scheme_version")
    return 0 if value is None else int(value)


# Reads as the second half of "N chunk(s) predate the current chunking scheme."
RESYNC_HINT = (
    "Their boundaries follow the old rules and their vectors were computed over text "
    "the embedding model truncated. Re-sync to rebuild them (sync_start, or "
    "`minirag-mcp sync`)."
)


class IndexOpenError(Exception):
    """The index folder cannot be created or opened."""

    @classmethod
    def at(cls, db_path: Path, error: OSError) -> IndexOpenError:
        return cls(
            f"Cannot open the index at {db_path}: {error.strerror or error}. DB_PATH (or "
            "--db-path) has to name a folder this user can create and write; set it to one."
        )


def _indexed_columns(table) -> set[str]:
    try:
        return {col for idx in table.list_indices() for col in idx.columns}
    except Exception:  # unreadable index manifest: treat every column as missing
        return set()


def _uncreated_index_warning(column: str, where: str, consequence: str, e: Exception) -> str:
    return (
        f"Could not create the BM25 index on the {column!r} column of {where}: {e}. Search "
        f"still works, but {consequence} until the index exists — reopen the database with "
        f"write access (and no concurrent writer) to retry."
    )


def _read_fts_language(db_path: Path) -> str | None:
    try:
        value = (db_path / FTS_LANGUAGE_FILENAME).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _write_fts_language(db_path: Path, language: str) -> None:
    try:
        (db_path / FTS_LANGUAGE_FILENAME).write_text(language + "\n", encoding="utf-8")
    except OSError as e:
        warnings.warn(
            f"The BM25 indexes at {db_path} were built with the {language} stemmer, but "
            f"recording that in {FTS_LANGUAGE_FILENAME} failed: {e}. Search is unaffected; "
            f"the next open cannot tell which stemmer they use and may rebuild them again.",
            RuntimeWarning,
            stacklevel=4,
        )


class Store:
    def __init__(self, db_path: Path, dim: int, fts_language: str = DEFAULT_FTS_LANGUAGE):
        try:
            db_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise IndexOpenError.at(db_path, e) from e
        # Zero means every read checks for a newer table version first. LanceDB's default
        # pins the version a handle opened with, and the MCP server holds one Store for
        # its lifetime while a terminal `sync` writes the same table. The check grows
        # with the versions the table has kept: about 0.1 ms a read at ten, 1 ms at
        # 1,500, 5 ms at 10,000.
        self._db = lancedb.connect(str(db_path), read_consistency_interval=timedelta(0))
        # The directory this index lives in. Exposed because the sync lock is a file
        # inside it, and both sync entry points must derive that path the same way —
        # deriving one from the Store and the other from Config would silently stop
        # locking the moment those two disagreed.
        self.db_path = db_path
        self.dim = dim
        # FTS columns this instance could not index (see _ensure_fts_indices), in the
        # chunks table and in the documents table
        self.missing_fts_indices: tuple[str, ...] = ()
        self.missing_document_fts_indices: tuple[str, ...] = ()
        # Whether a failed whole-document query was warned about; once is enough.
        self._document_query_warned = False
        # The stemmer configured, and the one the FTS indexes really use. They differ only
        # while a language change could not be written (see _ensure_fts_indices).
        self.fts_language = fts_language
        self.fts_index_language = fts_language
        # Columns this instance could not add to an older index (see _ensure_scheme_columns)
        self.missing_columns: tuple[str, ...] = ()
        # Run before every write, under the lock a swap of the index needs exclusively; it
        # raises to refuse the write. The path stays when a swap puts another index there,
        # and LanceDB follows it, so a store opened before a swap writes into the new one.
        self.write_check: Callable[[], None] | None = None
        # False only when an older index is missing the chunking-scheme columns and they
        # could not be added (see _ensure_scheme_columns)
        self.schema_migrated = True
        try:
            self._table = self._db.open_table(TABLE)
        except (FileNotFoundError, ValueError) as e:
            # Only a missing table means "create it". Letting every failure fall through
            # to create_table turns a corrupt table or a permissions problem into a
            # create-time error that names neither cause.
            if not _is_missing_table(e):
                raise
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("source_type", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("chunk_index", pa.int32()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    pa.field("file_hash", pa.string()),
                    pa.field("mtime", pa.float64()),
                    pa.field("ingested_at", pa.string()),
                    pa.field("parent_id", pa.string()),
                    pa.field("ocr_engine", pa.string()),
                    pa.field("scheme_version", pa.int32()),
                    *(pa.field(name, pa.string()) for name in LABEL_COLUMNS),
                    pa.field("pages", pa.string()),
                ]
            )
            self._table = self._db.create_table(TABLE, schema=schema)
        else:
            self._check_dim(db_path, dim)
            self._ensure_scheme_columns(db_path)
        self._documents = self._open_documents(db_path)
        self._ensure_fts_indices(db_path)
        self._signatures = self._open_signatures(db_path)
        self._signature_cache: tuple[tuple, _Signatures] | None = None

    def _open_signatures(self, db_path: Path):
        """The look-alike signature table, created when missing — best effort, never fatal.

        Creating it is a write, and opening an index must not require write access, for
        the reasons `_ensure_fts_indices` gives. Without it no result names a look-alike
        and `status` counts every document as unsigned, so the gap is reported.
        """
        try:
            return self._db.open_table(SIGNATURE_TABLE)
        except Exception as e:
            problem: Exception = e
            if _is_missing_table(e):
                try:
                    return self._db.create_table(
                        SIGNATURE_TABLE, schema=_signature_schema(), exist_ok=True
                    )
                except Exception as created:
                    problem = created
        warnings.warn(
            f"Could not open or create the look-alike signature table of the index at "
            f"{db_path}: {problem}. Search still works, but no result names its "
            f"look-alikes until the table exists — reopen the index with write access "
            f"and sync to build it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    def _open_documents(self, db_path: Path):
        """The `documents` table, created when missing — best effort, never fatal, for the
        reasons `_open_signatures` gives. Without it no document is ranked as a whole, and
        `status` counts every document as undocumented."""
        try:
            return self._db.open_table(DOCUMENT_TABLE)
        except Exception as e:
            problem: Exception = e
            if _is_missing_table(e):
                try:
                    return self._db.create_table(
                        DOCUMENT_TABLE, schema=_document_schema(), exist_ok=True
                    )
                except Exception as created:
                    problem = created
        warnings.warn(
            f"Could not open or create the documents table of the index at {db_path}: "
            f"{problem}. Search still works, but ranks no document as a whole until the "
            f"table exists — reopen the index with write access and sync to build it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    def _ensure_scheme_columns(self, db_path: Path) -> None:
        """Add the columns this version writes, if an older index lacks them.

        Adding a column is a write, and opening a database must not require write
        access — the same reasoning as `_ensure_fts_indices`. A failure is reported and
        recorded rather than raised: the columns are left in `missing_columns`, so what
        failed is inspectable and not only a warning in a log. Only a missing scheme
        column clears `schema_migrated`, which readers take as "every chunk is stale and
        no section can be rebuilt" — a verdict an additive column has no standing to
        pronounce.
        """
        present = set(self._table.schema.names)
        missing = {n: expr for n, expr in _ADDED_COLUMNS.items() if n not in present}
        if not missing:
            self.schema_migrated = True
            return
        scheme_missing = bool(missing.keys() & _SCHEME_COLUMNS.keys())
        try:
            self._table.add_columns(missing)
            self.schema_migrated = True
        except Exception as e:
            self.schema_migrated = not scheme_missing
            self.missing_columns = tuple(sorted(missing))
            consequence = (
                "This index predates the current chunking scheme, and it cannot be "
                "re-ingested until those columns exist"
                if scheme_missing
                else "Those columns read as empty for every row; the chunking scheme is unaffected"
            )
            warnings.warn(
                f"Could not add the {', '.join(sorted(missing))} column(s) to the index at "
                f"{db_path}: {e}. {consequence} — reopen it with write access and no "
                f"concurrent writer. Searching it still works.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _ensure_fts_indices(self, db_path: Path) -> None:
        """Create any FTS index the chunks or documents table is missing, and rebuild the
        ones built with another stemmer language — best effort, never fatal.

        Creating an FTS index on an empty table is supported (verified against
        lancedb 0.36), so a fresh table gets both here. Running this on open as well
        means an index built by an older version — `text` only — gains `title` search
        without a re-ingest of the corpus.

        The stemmer is fixed when an index is built, and LanceDB applies it to rows
        added later too, so a changed RAG_FTS_LANGUAGE reaches an existing index only
        through a rebuild. The language is recorded in FTS_LANGUAGE_FILENAME because
        LanceDB does not report it; an index with no record was built by `FTS()`, whose
        language is English. A rebuild re-reads every row and needs no embedding: under
        a second for 165,000 chunks.

        Both are writes, and opening a database must not require write access: a
        read-only copy would raise "Permission denied", and two processes opening the
        same un-upgraded database race on the same commit ("Retryable commit conflict").
        Both are stepped over with a warning. Searching `fts_columns=["text", "title"]`
        against a table that only has the `text` index works — LanceDB searches the
        indexes it has — so the only cost is that titles stay out of keyword results.
        An index that could not be rebuilt keeps answering in its old language, which
        `fts_index_language` reports, and the record is left as it was so the next
        open retries.

        Warnings go through `warnings.warn`, never `print`: stdout is the MCP stdio
        channel and anything written there corrupts the protocol.
        """
        tables = [(self._table, f"the index at {db_path}")]
        if self._documents is not None:
            tables.append((self._documents, f"the documents table of the index at {db_path}"))
        indexed_by = [_indexed_columns(table) for table, _ in tables]
        # One record serves both tables, so a language change rebuilds both, and one of
        # them left in the old language keeps the record unchanged for the next open.
        recorded = _read_fts_language(db_path)
        built = recorded or (DEFAULT_FTS_LANGUAGE if indexed_by[0] else self.fts_language)
        stale = built != self.fts_language
        config = FTS(language=self.fts_language)
        missing: list[str] = []
        document_missing: list[str] = []
        created = rebuilt = not_rebuilt = 0
        for (table, where), indexed in zip(tables, indexed_by, strict=True):
            chunks = table is self._table
            for column in FTS_COLUMNS:
                if column in indexed and not stale:
                    continue
                try:
                    table.create_index(column, config=config, replace=True)
                except Exception as e:
                    if column in indexed:
                        not_rebuilt += 1
                        warnings.warn(
                            f"Could not rebuild the BM25 index on the {column!r} column of "
                            f"{where} for RAG_FTS_LANGUAGE={self.fts_language}: {e}. Keyword "
                            f"search still works, but with the {built} stemmer until the index "
                            f"is rebuilt — reopen the database with write access (and no "
                            f"concurrent writer) to retry.",
                            RuntimeWarning,
                            stacklevel=3,
                        )
                    else:
                        if chunks:
                            missing.append(column)
                            consequence = f"{column} content is not keyword-searchable"
                        else:
                            document_missing.append(column)
                            consequence = f"documents are not ranked as a whole by their {column}"
                        warnings.warn(
                            _uncreated_index_warning(column, where, consequence, e),
                            RuntimeWarning,
                            stacklevel=3,
                        )
                else:
                    if column in indexed:
                        rebuilt += 1
                    else:
                        created += 1
        self.missing_fts_indices = tuple(missing)
        self.missing_document_fts_indices = tuple(document_missing)
        if not_rebuilt:
            self.fts_index_language = built
        elif (created or rebuilt) and recorded != self.fts_language:
            _write_fts_language(db_path, self.fts_language)

    def update_indices(self) -> str | None:
        """Fold rows written since the search indexes were built into them; the problem,
        if any, as text.

        LanceDB indexes only the rows present when an index is built. Rows added later
        are searched by brute force, scored with statistics that depend on how the rows
        happen to be split into data fragments, and returned out of score order — and
        every ingest writes a fragment of its own, while the FTS indexes are built on the
        empty table. `optimize` also compacts those fragments.
        """
        problems = []
        for table, what in (
            (self._table, "search indexes"),
            (self._document_table(create=False), "document search indexes"),
        ):
            if table is None:
                continue
            try:
                stale = any(
                    table.index_stats(i.name).num_unindexed_rows for i in table.list_indices()
                )
                if stale:
                    table.optimize()
            except Exception as e:
                problems.append(f"could not update the {what} of {self.db_path}: {e}")
        return "; ".join(problems) or None

    def _check_dim(self, db_path: Path, dim: int) -> None:
        """Fail early and clearly when an existing index has a different vector width.

        Without this, a MODEL_NAME change against an existing DB_PATH surfaces much
        later as LanceDB's "There is no vector column in the data" — which is both
        false and unactionable.
        """
        field = self._table.schema.field("vector")
        existing = getattr(field.type, "list_size", None)
        if existing is None or existing == dim:
            return
        raise DimensionMismatchError(
            f"Index at {db_path} was built with {existing}-dimensional vectors, "
            f"but the configured embedding model produces {dim} dimensions. "
            f"Vectors from different models are not comparable: point DB_PATH at a new "
            f"directory, or delete the index and re-ingest."
        )

    @_refused_while_rebuilt
    def replace_source(self, source: str, records: Sequence[ChunkRecord]) -> None:
        """Swap a source's chunks for `records` in one commit.

        A delete followed by an add is two versions, and the one between them has no
        such source: a concurrent reader finds it missing, and a second writer whose
        delete and add interleave with these leaves both copies. Single-file ingest
        takes no lock, so that interleaving is reachable.

        The merge key includes `ingested_at`, which every ingest stamps afresh, so no
        prior row matches and all of them are deleted. A caller that reuses a stored
        stamp gets the stored rows updated in place instead, duplicates included.
        Keyed on `id` alone, a source holding an id twice kept both copies forever.
        Two first ingests of one file racing each other still store it twice, since
        both see no rows to match; `list_sources` flags such a source as duplicated.
        """
        if not records:
            clause = f"source = '{sql_str(source)}'"
            self._table.delete(clause)
            self._delete_signature(source)
            self._delete_documents([source])
            return
        self._replace_many([(source, records)], indexed={source})

    @_refused_while_rebuilt
    def replace_sources(
        self, batch: Sequence[tuple[str, Sequence[ChunkRecord]]], *, indexed: Container[str]
    ) -> None:
        """Swap the chunks of every source in `batch` for its records, as `replace_source`
        does for one: the sources in `indexed` in one merge, the rest in one append, so a
        batch holding both kinds is two commits.

        The caller vouches that a source outside `indexed` has no chunks. One that a
        racing single-file ingest stored meanwhile ends up stored twice, which
        `list_sources` flags as duplicated and the next sync reads again. Each source's
        documents row and signature are written after the chunks, warning and never
        raising, as `replace_source` writes them.

        An entry without records, a source given twice, or a record filed under another
        source is a ValueError, raised before anything is written.
        """
        self._replace_many(batch, indexed)

    def _replace_many(
        self, batch: Sequence[tuple[str, Sequence[ChunkRecord]]], indexed: Container[str]
    ) -> None:
        if not batch:
            return
        seen: set[str] = set()
        for source, records in batch:
            if not records:
                raise ValueError(
                    f"no chunks to write for {source}; replace_source(source, []) deletes one"
                )
            if source in seen:
                raise ValueError(f"{source} is in one batch twice")
            seen.add(source)
            for r in records:
                if r.source != source:
                    raise ValueError(f"a chunk of {r.source} is batched as one of {source}")
        # Built up front: both writes after the chunk write are gated on stamps read right
        # after it, and a racing ingest landing between that read and them is overwritten.
        documents = [self._document_row(source, records) for source, records in batch]
        entries = [self._signature_entry(source, records) for source, records in batch]
        replaced = [(source, records) for source, records in batch if source in indexed]
        added = [(source, records) for source, records in batch if source not in indexed]
        if replaced:
            (
                self._table.merge_insert(["id", "ingested_at"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(_in_clause(s for s, _ in replaced))
                .execute(self._chunk_rows(r for _, records in replaced for r in records))
            )
        if added:
            self._table.add(self._chunk_rows(r for _, records in added for r in records))
        sources = [source for source, _ in batch]
        try:
            stamps = self._chunk_stamps(sources)
        except Exception as e:
            self._warn_documents(f"write the document row of {_named(sources)}", e)
            self._warn_signature(f"write the look-alike signature of {_named(sources)}", e)
            return
        self._write_documents(documents, stamps=stamps)
        try:
            self._put_signatures(entries, stamps=stamps)
        except Exception as e:
            self._warn_signature(f"write the look-alike signature of {_named(sources)}", e)

    def _chunk_rows(self, records: Iterable[ChunkRecord]) -> list[dict]:
        """`records` as rows of the chunks table.

        An additive column this index could not gain reads as empty, and is written as
        nothing: the write must not fail on a column the table does not have.
        """
        absent = [c for c in self.missing_columns if c in _ADDITIVE_COLUMNS]
        rows = [asdict(r) for r in records]
        for row in rows:
            for column in absent:
                del row[column]
        return rows

    @_refused_while_rebuilt
    def add_sources(self, batch: Sequence[tuple[str, Sequence[ChunkRecord]]]) -> None:
        """Append the chunks and signatures of sources that have none yet, in one commit.

        For a bulk load into an index that holds none of these sources: unlike
        `replace_source` it deletes nothing, and it writes no documents rows, which
        `backfill_documents` writes from the stored chunks afterwards.
        """
        self._table.add(self._chunk_rows(r for _, records in batch for r in records))
        self._put_signatures([self._signature_entry(source, records) for source, records in batch])

    @_refused_while_rebuilt
    def move_source(self, old: str, new: str, *, mtime: float) -> int:
        """Re-path a source's chunks in one commit, keeping their text and vectors.

        Chunk and section ids are prefixed with the source. Left as they were, they are
        the ids the next file ingested at the old path gets, and that ingest would
        overwrite these rows or fuse the two documents into one section. The ids are
        rewritten here rather than in SQL: Lance's update accepts `substr` only in some
        positions, and a regex replacement would have to escape arbitrary file names.
        """
        clause = f"source = '{sql_str(old)}'"
        columns = self._table.schema.names
        rows = self._table.search().where(clause).limit(_LIST_LIMIT).to_list()
        if not rows:
            return 0
        cut = len(old)
        moved = []
        for row in rows:
            row = {c: row[c] for c in columns}
            row["source"] = new
            row["id"] = new + row["id"][cut:]
            if row.get("parent_id"):
                row["parent_id"] = new + row["parent_id"][cut:]
            row["mtime"] = mtime
            if "label_hash" in row:
                row["label_hash"] = ""
            moved.append(row)
        (
            self._table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .when_not_matched_by_source_delete(clause)
            .execute(moved)
        )
        self._move_signature(old, new)
        self._move_document(old, new)
        return len(moved)

    @_refused_while_rebuilt
    def delete_source(self, source: str) -> int:
        clause = f"source = '{sql_str(source)}'"
        before = self._table.count_rows(filter=clause)
        if before:
            self._table.delete(clause)
        self._delete_signature(source)
        self._delete_documents([source])
        return before

    def copy_reembedded(
        self,
        source: Store,
        embed: Callable[[list[str]], list[list[float]]],
        *,
        batch_rows: int = 1024,
        on_event: Callable[[str], None] | None = None,
    ) -> int:
        """Copy every chunk of `source` into this index with its stored text embedded again
        by `embed`; the number of chunks copied.

        Every other column is copied as it is, labels included, and so are the document
        rows and the look-alike signatures, which stay current because the chunks keep
        their stamps. Nothing is parsed. For an empty index, which may be of another width.
        """
        total = source.chunk_count()
        schema = self._table.schema
        columns = [name for name in schema.names if name != "vector"]
        batches = source._table.search().select(columns).limit(_LIST_LIMIT).to_batches(batch_rows)
        copied = 0
        for batch in batches:
            if not batch.num_rows:
                continue
            vectors = np.asarray(embed(batch.column("text").to_pylist()), dtype=np.float32)
            if vectors.shape != (batch.num_rows, self.dim):
                raise ValueError(
                    f"re-embedding {batch.num_rows} chunks gave vectors of shape "
                    f"{vectors.shape}, not ({batch.num_rows}, {self.dim})"
                )
            data = {name: batch.column(name) for name in columns}
            data["vector"] = pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), self.dim)
            self._table.add(pa.table({name: data[name] for name in schema.names}, schema=schema))
            copied += batch.num_rows
            if on_event:
                on_event(f"re-embedded {copied} of {total} chunks")
        if copied != total:
            raise RuntimeError(
                f"Re-embedding read {copied} of the {total} chunks of {source.db_path}, so "
                f"the index rebuilt in {self.db_path} would lack some; it is not used."
            )
        self._copy_documents(source)
        self._copy_signatures(source)
        return copied

    def _copy_documents(self, source: Store) -> None:
        """`source`'s document rows, each written while its chunks here carry its stamp."""
        try:
            table = source._document_table(create=False)
            if table is None:
                return
            names = _document_schema().names
            batches = table.search().select(names).limit(_LIST_LIMIT).to_batches(_SIGN_BATCH)
            for batch in batches:
                if batch.num_rows:
                    self._write_documents(batch.to_pylist())
        except Exception as e:
            self._warn_documents("copy the document rows", e)

    def _copy_signatures(self, source: Store) -> None:
        try:
            table = source._signature_table(create=False)
            if table is None:
                return
            data = table.search().select(_signature_schema().names).limit(_LIST_LIMIT).to_arrow()
            if data.num_rows:
                self._signature_table(create=True).add(data)
        except Exception as e:
            self._warn_signature("copy the look-alike signatures", e)

    # Look-alike signatures live in a table of their own, so no write spans both tables
    # in one commit. What keeps the two consistent is the chunk stamp a signature
    # carries: it is written only while the source's chunks still carry that stamp, and
    # used only where they still do. Whatever a crash or a race leaves behind reads as
    # unsigned, which `status` counts and the next sync signs. A failed signature write
    # never fails the chunk write it follows.

    def _signature_table(self, *, create: bool):
        if self._signatures is None:
            try:
                self._signatures = self._db.open_table(SIGNATURE_TABLE)
            except Exception as e:
                if not create:
                    return None
                if not _is_missing_table(e):
                    raise
                self._signatures = self._db.create_table(
                    SIGNATURE_TABLE, schema=_signature_schema(), exist_ok=True
                )
        return self._signatures

    def _warn_signature(self, what: str, e: Exception) -> None:
        warnings.warn(
            f"Could not {what} in the index at {self.db_path}: {e}. Until a sync signs it, "
            "results do not name its look-alikes.",
            RuntimeWarning,
            stacklevel=4,
        )

    def _signature_entry(
        self, source: str, records: Sequence[ChunkRecord]
    ) -> tuple[str, str, Signature]:
        """The signature of `source`, computed from `records` in chunk order."""
        ordered = sorted(records, key=lambda r: r.chunk_index)
        return source, ordered[0].ingested_at, signature(join_document(ordered))

    def _put_signatures(
        self,
        entries: Sequence[tuple[str, str, Signature]],
        *,
        stamps: dict[str, set[str]] | None = None,
    ) -> list[str]:
        """Write each signature whose source's chunks all still carry its stamp, read here
        unless the caller read `stamps` after its own chunk write."""
        if not entries:
            return []
        if stamps is None:
            stamps = self._chunk_stamps(source for source, _, _ in entries)
        current = [e for e in entries if stamps.get(e[0]) == {e[1]}]
        if current:
            self._write_signatures(current)
        return [source for source, _, _ in current]

    def _chunk_stamps(self, sources: Iterable[str]) -> dict[str, set[str]]:
        """The stamps the chunks of each of `sources` carry."""
        rows = (
            self._table.search()
            .where(_in_clause(sources))
            .select(["source", "ingested_at"])
            .limit(_LIST_LIMIT)
            .to_arrow()
        )
        stamps: dict[str, set[str]] = {}
        for source, stamp in zip(
            rows.column("source").to_pylist(), rows.column("ingested_at").to_pylist(), strict=True
        ):
            stamps.setdefault(source, set()).add(stamp)
        return stamps

    def _write_signatures(self, entries: Sequence[tuple[str, str, Signature]]) -> None:
        values = np.concatenate([sig.values for _, _, sig in entries]).astype(np.uint32)
        data = pa.table(
            {
                "source": pa.array([source for source, _, _ in entries], pa.string()),
                "ingested_at": pa.array([stamp for _, stamp, _ in entries], pa.string()),
                "signature": pa.FixedSizeListArray.from_arrays(pa.array(values), NUM_PERM),
                "shingles": pa.array([sig.shingles for _, _, sig in entries], pa.int64()),
            },
            schema=_signature_schema(),
        )
        # Keyed like `replace_source`, so a row of the same source under another stamp,
        # one a racing insert left behind included, is deleted in the same commit.
        (
            self._signature_table(create=True)
            .merge_insert(["source", "ingested_at"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .when_not_matched_by_source_delete(_in_clause(source for source, _, _ in entries))
            .execute(data)
        )

    def _move_signature(self, old: str, new: str) -> None:
        try:
            table = self._signature_table(create=False)
            if table is None:
                return
            rows = (
                table.search()
                .where(f"source = '{sql_str(old)}'")
                .select(_signature_schema().names)
                .limit(_LIST_LIMIT)
                .to_arrow()
            )
            if not rows.num_rows:
                return
            rows = rows.set_column(
                rows.schema.get_field_index("source"),
                "source",
                pa.array([new] * rows.num_rows, pa.string()),
            )
            (
                table.merge_insert(["source", "ingested_at"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(_in_clause((old, new)))
                .execute(rows)
            )
        except Exception as e:
            self._warn_signature(f"move the look-alike signature of {old} to {new}", e)

    def _delete_signature(self, source: str) -> None:
        clause = f"source = '{sql_str(source)}'"
        try:
            table = self._signature_table(create=False)
            if table is not None and table.count_rows(filter=clause):
                table.delete(clause)
        except Exception as e:
            self._warn_signature(f"delete the look-alike signature of {source}", e)

    def _signature_state(self) -> _Signatures:
        """`_read_signature_index`, kept until either table gains a version."""
        try:
            table = self._signature_table(create=False)
            version = None if table is None else table.version
        except Exception:
            version = None  # _read_signature_index warns
        key = (self._table.version, version)
        cached = self._signature_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        state = self._read_signature_index()
        self._signature_cache = (key, state)
        return state

    def _read_signature_index(self) -> _Signatures:
        present = set(self._table.schema.names)
        dated = [c for c in ("doc_date", "doc_date_source") if c in present]
        meta = (
            self._table.search()
            .select(["source", "ingested_at", *dated])
            .limit(_LIST_LIMIT)
            .to_arrow()
            .group_by("source")
            .aggregate(
                [("ingested_at", "min"), ("ingested_at", "max")] + [(c, "max") for c in dated]
            )
        )
        stamp_of: dict[str, str | None] = {}
        dates: dict[str, tuple[str, str]] = {}
        for row in meta.to_pylist():
            low, high = row["ingested_at_min"], row["ingested_at_max"]
            stamp_of[row["source"]] = low if low == high else None
            dates[row["source"]] = (
                row.get("doc_date_max") or "",
                row.get("doc_date_source_max") or "",
            )
        sources: list[str] = []
        picked: list[int] = []
        signed: set[str] = set()
        values = np.zeros((0, NUM_PERM), dtype=np.uint32)
        rows: list[tuple[str, str, int]] = []
        # A signatures table that cannot be read leaves every document unsigned, never
        # takes search, status or sync down with it.
        try:
            table = self._signature_table(create=False)
            if table is not None:
                sigs = (
                    table.search().select(_signature_schema().names).limit(_LIST_LIMIT).to_arrow()
                )
                flat = sigs.column("signature").combine_chunks().flatten()
                values = flat.to_numpy(zero_copy_only=False).astype(np.uint32).reshape(-1, NUM_PERM)
                rows = list(
                    zip(
                        sigs.column("source").to_pylist(),
                        sigs.column("ingested_at").to_pylist(),
                        sigs.column("shingles").to_pylist(),
                        strict=True,
                    )
                )
                if len(rows) != len(values):
                    raise ValueError(f"{len(values)} signatures for {len(rows)} rows")
        except Exception as e:
            self._warn_signature("read the look-alike signatures", e)
            rows, values = [], np.zeros((0, NUM_PERM), dtype=np.uint32)
        for i, (source, stamp, shingles) in enumerate(rows):
            if source in signed or stamp_of.get(source) is None or stamp_of[source] != stamp:
                continue
            signed.add(source)
            if shingles > 0:
                sources.append(source)
                picked.append(i)
        return _Signatures(
            index=SignatureIndex(sources, values[picked]),
            dates=dates,
            unsigned={s: stamp for s, stamp in stamp_of.items() if s not in signed},
        )

    def look_alikes(self, sources: Iterable[str]) -> dict[str, list[LookAlike]]:
        """The look-alikes of each of `sources` that has any, from the whole index."""
        wanted = list(dict.fromkeys(sources))
        if not wanted:
            return {}
        state = self._signature_state()
        out: dict[str, list[LookAlike]] = {}
        for source in wanted:
            if near := state.index.near(source):
                out[source] = [
                    replace(
                        n,
                        doc_date=state.dates[n.source][0],
                        doc_date_source=state.dates[n.source][1],
                    )
                    for n in near
                ]
        return out

    def unsigned_sources(self, scopes: tuple[str, ...] = ()) -> list[str]:
        """Indexed sources in `scopes` without a current look-alike signature."""
        return sorted(
            s
            for s in self._signature_state().unsigned
            if not scopes or any(is_under(s, p) for p in scopes)
        )

    def backfill_signatures(
        self,
        scopes: tuple[str, ...] = (),
        on_event: Callable[[str], None] | None = None,
    ) -> tuple[int, list[dict]]:
        """Sign every source in `scopes` that has no current signature.

        From the stored chunk text, rebuilt as `read_file` rebuilds it: nothing is parsed
        or embedded again. A source whose chunks carry two stamps is left alone; the
        sync's diff re-reads such a source, and that ingest signs it.
        """
        todo = sorted(
            s
            for s, stamp in self._signature_state().unsigned.items()
            if stamp is not None and (not scopes or any(is_under(s, p) for p in scopes))
        )
        signed = 0
        for start in range(0, len(todo), _SIGN_BATCH):
            try:
                written = self._put_signatures(
                    self._stored_signatures(todo[start : start + _SIGN_BATCH])
                )
            except Exception as e:  # every later batch would fail the same way
                return signed, [
                    {
                        "source": str(self.db_path),
                        "error": f"lookAlikes: {type(e).__name__}: {e}; "
                        f"{len(todo) - signed} document(s) left without a look-alike "
                        "signature, which the next sync retries",
                    }
                ]
            signed += len(written)
            if on_event:
                for source in written:
                    on_event(f"signed {source}")
        return signed, []

    def _stored_signatures(self, sources: Sequence[str]) -> list[tuple[str, str, Signature]]:
        present = set(self._table.schema.names)
        wanted = ["source", "chunk_index", "text", "parent_id", "ingested_at"]
        rows = (
            self._table.search()
            .where(_in_clause(sources))
            .select([c for c in wanted if c in present])
            .limit(_LIST_LIMIT)
            .to_list()
        )
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["source"], []).append(row)
        out = []
        for source, group in grouped.items():
            group.sort(key=lambda r: r["chunk_index"])
            units = [_Unit(r["text"], r.get("parent_id") or "") for r in group]
            out.append((source, group[0]["ingested_at"], signature(join_document(units))))
        return out

    # The documents table is kept the way the signature table is: no write spans it and the
    # chunks table in one commit, a row carries the stamp of the chunks it was joined
    # from and is written only while they still carry it, and whatever a crash or a race
    # leaves behind reads as undocumented, which `status` counts and the next sync
    # rebuilds. A failed document write never fails the chunk write it follows.

    def _document_table(self, *, create: bool):
        if self._documents is None:
            try:
                self._documents = self._db.open_table(DOCUMENT_TABLE)
            except Exception as e:
                if not create:
                    return None
                if not _is_missing_table(e):
                    raise
                table = self._db.create_table(
                    DOCUMENT_TABLE, schema=_document_schema(), exist_ok=True
                )
                config = FTS(language=self.fts_index_language)
                missing = []
                for column in FTS_COLUMNS:
                    try:
                        table.create_index(column, config=config, replace=True)
                    except Exception as index_error:
                        missing.append(column)
                        warnings.warn(
                            _uncreated_index_warning(
                                column,
                                f"the documents table of the index at {self.db_path}",
                                f"documents are not ranked as a whole by their {column}",
                                index_error,
                            ),
                            RuntimeWarning,
                            stacklevel=4,
                        )
                self.missing_document_fts_indices = tuple(missing)
                self._documents = table
        return self._documents

    def _warn_documents(
        self,
        what: str,
        e: Exception,
        consequence: str = "Until a sync writes its row again, search does not rank that "
        "document as a whole.",
    ) -> None:
        warnings.warn(
            f"Could not {what} in the index at {self.db_path}: {e}. {consequence}",
            RuntimeWarning,
            stacklevel=4,
        )

    def _document_row(self, source: str, records: Sequence[ChunkRecord]) -> dict:
        """The documents row of `source`, joined from `records` in chunk order."""
        ordered = sorted(records, key=lambda r: r.chunk_index)
        return {
            "source": source,
            "title": ordered[0].title,
            "text": join_document(ordered),
            "file_hash": ordered[0].file_hash,
            "ingested_at": ordered[0].ingested_at,
        }

    def _write_documents(
        self, rows: Sequence[dict], *, stamps: dict[str, set[str]] | None = None
    ) -> None:
        """`_put_documents`, warning instead of raising."""
        try:
            self._put_documents(rows, stamps=stamps)
        except Exception as e:
            self._warn_documents(
                f"write the document row of {_named([r['source'] for r in rows])}", e
            )

    def _put_documents(
        self, rows: Sequence[dict], *, stamps: dict[str, set[str]] | None = None
    ) -> list[str]:
        """Write each row whose source's chunks all still carry its stamp, in one merge
        that also deletes the other rows of those sources; the sources written. The
        stamps are read here unless the caller read `stamps` after its own chunk write."""
        if not rows:
            return []
        if stamps is None:
            stamps = self._chunk_stamps(row["source"] for row in rows)
        current = [row for row in rows if stamps.get(row["source"]) == {row["ingested_at"]}]
        if current:
            # Keyed like `replace_source`, so a row of the same source under another stamp,
            # one a racing insert left behind included, is deleted in the same commit.
            (
                self._document_table(create=True)
                .merge_insert(["source", "ingested_at"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(_in_clause(row["source"] for row in current))
                .execute(pa.Table.from_pylist(current, schema=_document_schema()))
            )
        return [row["source"] for row in current]

    def _move_document(self, old: str, new: str) -> None:
        try:
            table = self._document_table(create=False)
            if table is None:
                return
            rows = (
                table.search()
                .where(f"source = '{sql_str(old)}'")
                .select(_document_schema().names)
                .limit(_LIST_LIMIT)
                .to_arrow()
            )
            if not rows.num_rows:
                return
            rows = rows.set_column(
                rows.schema.get_field_index("source"),
                "source",
                pa.array([new] * rows.num_rows, pa.string()),
            )
            (
                table.merge_insert(["source", "ingested_at"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(_in_clause((old, new)))
                .execute(rows)
            )
        except Exception as e:
            self._warn_documents(f"move the document row of {old} to {new}", e)

    def _delete_documents(self, sources: Sequence[str]) -> None:
        if not sources:
            return
        clause = _in_clause(sources)
        try:
            table = self._document_table(create=False)
            if table is not None and table.count_rows(filter=clause):
                table.delete(clause)
        except Exception as e:
            self._warn_documents(f"delete the document row of {_named(sources)}", e)

    def _document_gaps(self) -> _DocumentGaps:
        documented: set[tuple[str, str]] = set()
        # Read before the chunks: a row is written only after its chunks, so a row whose
        # source has no chunks in the later read has really lost them.
        # A documents table that cannot be read leaves every document undocumented, never
        # takes status or sync down with it.
        try:
            table = self._document_table(create=False)
            if table is not None:
                rows = table.search().select(["source", "ingested_at"]).limit(_LIST_LIMIT)
                rows = rows.to_arrow()
                documented = set(
                    zip(
                        rows.column("source").to_pylist(),
                        rows.column("ingested_at").to_pylist(),
                        strict=True,
                    )
                )
        except Exception as e:
            self._warn_documents(
                "read the document rows",
                e,
                "Every document is counted as without its row until the table can be read.",
            )
        meta = (
            self._table.search()
            .select(["source", "ingested_at"])
            .limit(_LIST_LIMIT)
            .to_arrow()
            .group_by("source")
            .aggregate([("ingested_at", "min"), ("ingested_at", "max")])
        )
        undocumented: dict[str, str | None] = {}
        indexed: set[str] = set()
        for row in meta.to_pylist():
            indexed.add(row["source"])
            low, high = row["ingested_at_min"], row["ingested_at_max"]
            stamp = low if low == high else None
            if stamp is None or (row["source"], stamp) not in documented:
                undocumented[row["source"]] = stamp
        orphans = sorted(pair for pair in documented if pair[0] not in indexed)
        return _DocumentGaps(undocumented, orphans)

    def undocumented_sources(self) -> list[str]:
        """Indexed sources without a current documents row."""
        return sorted(self._document_gaps().undocumented)

    def backfill_documents(
        self,
        scopes: tuple[str, ...] = (),
        on_event: Callable[[str], None] | None = None,
    ) -> tuple[int, list[dict]]:
        """Write the documents row of every source in `scopes` that has no current one,
        and delete the rows in `scopes` whose source has no chunks left.

        From the stored chunk text, as `backfill_signatures` signs: nothing is parsed or
        embedded again. A source whose chunks carry two stamps is left alone; the sync's
        diff re-reads such a source, and that ingest writes its row.
        """

        def in_scope(source: str) -> bool:
            return not scopes or any(is_under(source, p) for p in scopes)

        gaps = self._document_gaps()
        errors: list[dict] = []
        if orphans := [(s, stamp) for s, stamp in gaps.orphans if in_scope(s)]:
            try:
                self._delete_orphans(orphans)
            except Exception as e:
                errors.append(
                    {
                        "source": str(self.db_path),
                        "error": f"documents: {type(e).__name__}: {e}; {len(orphans)} row(s) "
                        "of documents no longer indexed left in the whole-document ranking, "
                        "which the next sync retries",
                    }
                )
            else:
                if on_event:
                    for source in dict.fromkeys(s for s, _ in orphans):
                        on_event(f"removed the document row of {source}")
        todo = sorted(
            s for s, stamp in gaps.undocumented.items() if stamp is not None and in_scope(s)
        )
        written_count = 0
        for start in range(0, len(todo), _SIGN_BATCH):
            try:
                written = self._put_documents(
                    self._stored_documents(todo[start : start + _SIGN_BATCH])
                )
            except Exception as e:  # every later batch would fail the same way
                return written_count, [
                    *errors,
                    {
                        "source": str(self.db_path),
                        "error": f"documents: {type(e).__name__}: {e}; "
                        f"{len(todo) - written_count} document(s) left without the row "
                        "that ranks them as a whole, which the next sync retries",
                    },
                ]
            written_count += len(written)
            if on_event:
                for source in written:
                    on_event(f"documented {source}")
        return written_count, errors

    def _delete_orphans(self, orphans: Sequence[tuple[str, str]]) -> None:
        """Delete exactly these (source, stamp) rows: a row a re-ingest of the same path
        wrote since they were read carries a new stamp and stays."""
        clause = " OR ".join(
            f"(source = '{sql_str(source)}' AND ingested_at = '{sql_str(stamp)}')"
            for source, stamp in orphans
        )
        table = self._document_table(create=False)
        if table is not None:
            table.delete(clause)

    def _stored_documents(self, sources: Sequence[str]) -> list[dict]:
        present = set(self._table.schema.names)
        wanted = ["source", "chunk_index", "text", "parent_id", "title", "file_hash", "ingested_at"]
        rows = (
            self._table.search()
            .where(_in_clause(sources))
            .select([c for c in wanted if c in present])
            .limit(_LIST_LIMIT)
            .to_list()
        )
        grouped: dict[str, list[_StoredChunk]] = {}
        for r in rows:
            grouped.setdefault(r["source"], []).append(
                _StoredChunk(
                    r["chunk_index"],
                    r["text"],
                    r.get("parent_id") or "",
                    r["title"],
                    r["file_hash"],
                    r["ingested_at"],
                )
            )
        return [self._document_row(source, group) for source, group in grouped.items()]

    def _document_ranks(self, query_text: str, scope_clause: str | None) -> dict[str, int]:
        """The 0-based rank of each of the first DOCUMENT_FETCH sources in `scope_clause`
        that `query_text` matches as a whole document; empty when that cannot be asked."""
        try:
            table = self._document_table(create=False)
            if table is None:
                return {}
            q = (
                table.search(query_text, query_type="fts", fts_columns=list(FTS_COLUMNS))
                .select(["source", "_score"])
                .limit(DOCUMENT_FETCH)
            )
            if scope_clause:
                q = q.where(scope_clause)
            rows = sorted(q.to_list(), key=lambda r: (-r["_score"], r["source"]))
        except Exception as e:
            if not self._document_query_warned:
                self._document_query_warned = True
                warnings.warn(
                    f"Ranking documents as a whole failed in the index at {self.db_path}: {e}. "
                    "Search answers without that ranking until it works again.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            return {}
        ranks: dict[str, int] = {}
        for rank, row in enumerate(rows):
            ranks.setdefault(row["source"], rank)
        return ranks

    def neighbors(
        self, source: str, chunk_index: int, before: int = 1, after: int = 1
    ) -> list[SearchResult]:
        lo, hi = max(0, chunk_index - before), chunk_index + after
        clause = f"source = '{sql_str(source)}' AND chunk_index >= {lo} AND chunk_index <= {hi}"
        rows = self._table.search().where(clause).limit(_LIST_LIMIT).to_list()
        rows.sort(key=lambda r: r["chunk_index"])
        return [
            SearchResult(
                text=r["text"],
                source=r["source"],
                title=r["title"],
                chunk_index=r["chunk_index"],
                score=0.0,
                distance=None,
                parent_id=r.get("parent_id", ""),
                pages=r.get("pages") or "",
            )
            for r in rows
        ]

    def all_chunks(self, source: str) -> list[SearchResult]:
        clause = f"source = '{sql_str(source)}'"
        rows = self._table.search().where(clause).limit(_LIST_LIMIT).to_list()
        rows.sort(key=lambda r: r["chunk_index"])
        return [
            SearchResult(
                text=r["text"],
                source=r["source"],
                title=r["title"],
                chunk_index=r["chunk_index"],
                score=0.0,
                distance=None,
                parent_id=r.get("parent_id", ""),
                pages=r.get("pages") or "",
            )
            for r in rows
        ]

    @_refused_while_rebuilt
    def set_document_labels(
        self,
        source: str,
        *,
        labels_json: str,
        label_keys: str,
        label_hash: str,
        doc_date: str,
        doc_date_source: str,
    ) -> int:
        """Write a document's label and date columns on every one of its chunks."""
        result = self._table.update(
            where=f"source = '{sql_str(source)}'",
            values={
                "labels_json": labels_json,
                "label_keys": label_keys,
                "label_hash": label_hash,
                "doc_date": doc_date,
                "doc_date_source": doc_date_source,
            },
        )
        return result.rows_updated

    def _meta_columns(self) -> list[str]:
        """`_META_COLS`, minus anything an un-migrated older index does not have."""
        present = set(self._table.schema.names)
        return [c for c in _META_COLS if c in present]

    def _iter_meta(self, scopes: tuple[str, ...] = (), filter_sql: str | None = None) -> list[dict]:
        q = self._table.search().select(self._meta_columns())
        clause = _and_clauses(sql_clause(scopes), filter_sql)
        if clause:
            q = q.where(clause)
        return q.limit(_LIST_LIMIT).to_list()

    def list_sources(
        self, scopes: tuple[str, ...] = (), filter_sql: str | None = None
    ) -> list[SourceInfo]:
        by_source: dict[str, dict] = {}
        counts: dict[str, int] = {}
        indexes: dict[str, set[int]] = {}
        schemes: dict[str, int] = {}
        engines: dict[str, str] = {}
        unpaged: set[str] = set()
        for row in self._iter_meta(scopes, filter_sql):
            by_source.setdefault(row["source"], row)
            counts[row["source"]] = counts.get(row["source"], 0) + 1
            indexes.setdefault(row["source"], set()).add(row["chunk_index"])
            scheme = _scheme_of(row)
            schemes[row["source"]] = min(schemes.get(row["source"], scheme), scheme)
            engines[row["source"]] = engines.get(row["source"]) or row.get("ocr_engine", "")
            if not _pages_recorded([row]):
                unpaged.add(row["source"])
        return [
            SourceInfo(
                source=src,
                source_type=row["source_type"],
                title=row["title"],
                chunk_count=counts[src],
                file_hash=row["file_hash"],
                mtime=row["mtime"],
                scheme_version=schemes[src],
                ocr_engine=engines[src] or "",
                duplicated=counts[src] > len(indexes[src]),
                **_label_fields(row),
                pages_recorded=src not in unpaged,
                ingested_at=row.get("ingested_at") or "",
            )
            for src, row in sorted(by_source.items())
        ]

    def get_source(self, source: str) -> SourceInfo | None:
        rows = (
            self._table.search()
            .where(f"source = '{sql_str(source)}'")
            .select(self._meta_columns())
            .limit(_LIST_LIMIT)
            .to_list()
        )
        if not rows:
            return None
        r = rows[0]
        engine = next((row.get("ocr_engine", "") for row in rows if row.get("ocr_engine")), "")
        return SourceInfo(
            source=r["source"],
            source_type=r["source_type"],
            title=r["title"],
            chunk_count=len(rows),
            file_hash=r["file_hash"],
            mtime=r["mtime"],
            scheme_version=min(_scheme_of(row) for row in rows),
            ocr_engine=engine,
            duplicated=len(rows) > len({row["chunk_index"] for row in rows}),
            **_label_fields(r),
            pages_recorded=_pages_recorded(rows),
        )

    def unpaged_pdfs(self) -> dict[str, str]:
        """PDF files with a chunk stored before page numbers were, each with its OCR
        engine ("" when OCR read none of it). Asked of the index rather than listed from
        every document, so `status` stays quick on a large one."""
        present = set(self._table.schema.names)
        clauses = ["source_type = 'file'", "lower(source) LIKE '%.pdf'"]
        if "pages" in present:
            clauses.append("pages IS NULL")
        columns = ["source", *(["ocr_engine"] if "ocr_engine" in present else [])]
        rows = (
            self._table.search()
            .where(" AND ".join(clauses))
            .select(columns)
            .limit(_LIST_LIMIT)
            .to_arrow()
        )
        engines = (
            rows.column("ocr_engine").to_pylist()
            if "ocr_engine" in columns
            else [""] * rows.num_rows
        )
        out: dict[str, str] = {}
        for source, engine in zip(rows.column("source").to_pylist(), engines, strict=True):
            out[source] = out.get(source) or engine or ""
        return out

    def stale_chunk_count(self) -> int:
        """Chunks written by an older chunking scheme. Detected, never assumed.

        Their boundaries follow the old rules and their vectors were computed over
        truncated text, so they are not comparable with chunks written now — but they
        are still returned by search, which is why this has to be reported rather than
        silently tolerated.
        """
        if not self.schema_migrated:
            return self._table.count_rows()
        return self._table.count_rows(filter=f"scheme_version < {SCHEME_VERSION}")

    def parent_texts(self, parent_ids: Sequence[str]) -> dict[str, str]:
        """The enclosing section of each chunk, rebuilt from its siblings.

        A section is never stored twice: its text is exactly its chunks, in
        `chunk_index` order, with the header they all repeat emitted once. Chunks from
        an older scheme carry no parent_id and are skipped — grouping them all under
        the empty string would fuse the whole index into one "section".
        """
        wanted = sorted({p for p in parent_ids if p})
        if not wanted or not self.schema_migrated:
            return {}
        values = ", ".join(f"'{sql_str(p)}'" for p in wanted)
        rows = (
            self._table.search()
            .where(f"parent_id IN ({values})")
            .select(["parent_id", "chunk_index", "text"])
            .limit(_LIST_LIMIT)
            .to_list()
        )
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["parent_id"], []).append(row)
        out: dict[str, str] = {}
        for pid, group in grouped.items():
            group.sort(key=lambda r: r["chunk_index"])
            out[pid] = join_parent([r["text"] for r in group])
        return out

    def parent_text(self, parent_id: str) -> str | None:
        """The enclosing section of one chunk, or None when there is none to rebuild."""
        return self.parent_texts([parent_id]).get(parent_id)

    def literal_candidates(
        self,
        terms: Sequence[literal.Term],
        *,
        scopes: tuple[str, ...] = (),
        filter_sql: str | None = None,
    ) -> Iterator[SearchResult]:
        """Every chunk whose text or title may contain one of `terms`, in document order.

        A superset of the matches: the SQL clause cannot see where a word starts, so the
        caller keeps what `literal.matches` accepts. Never capped, because a report that
        a term is absent is only as good as the scan behind it. Rows are handed out a
        batch at a time: a broad stem can select most of the index, and its text sits
        far more compactly in Arrow than as Python strings.
        """
        clause = _and_clauses(
            sql_clause(scopes), filter_sql, literal.sql_clause(terms, FTS_COLUMNS)
        )
        present = set(self._table.schema.names)
        wanted = ["source", "title", "chunk_index", "text", "parent_id", "pages", *LABEL_COLUMNS]
        table = (
            self._table.search()
            .where(clause)
            .select([c for c in wanted if c in present])
            .limit(_LIST_LIMIT)
            .to_arrow()
            .sort_by([("source", "ascending"), ("chunk_index", "ascending")])
        )
        for batch in table.to_batches(max_chunksize=1024):
            for r in batch.to_pylist():
                yield SearchResult(
                    text=r["text"],
                    source=r["source"],
                    title=r["title"],
                    chunk_index=r["chunk_index"],
                    score=0.0,
                    distance=None,
                    parent_id=r.get("parent_id") or "",
                    labels_json=r.get("labels_json") or "",
                    doc_date=r.get("doc_date") or "",
                    doc_date_source=r.get("doc_date_source") or "",
                    pages=r.get("pages") or "",
                )

    def chunk_count(self) -> int:
        return self._table.count_rows()

    def table_version(self) -> int:
        """The table's version: changes with every write, so equal versions mean equal rows."""
        return self._table.version

    def source_count(self) -> int:
        return len({r["source"] for r in self._iter_meta()})

    def search(
        self,
        query_text: str,
        query_vector: list[float],
        *,
        top_k: int = 8,
        hybrid_weight: float = 0.6,
        scopes: tuple[str, ...] = (),
        max_distance: float | None = None,
        grouping: str | None = None,
        max_files: int | None = None,
        include_parents: bool = True,
        filter_sql: str | None = None,
        document_weight: float = 0.0,
        reranker: Callable[[str, list[str]], list[float]] | None = None,
        rerank_top: int = 30,
    ) -> list[SearchResult]:
        """Hybrid search: vector + BM25 fused by weighted RRF, then filtered.

        With `document_weight`, each fused chunk also gains `document_weight / (60 + r)`
        where `r` is its document's 1-based rank among the documents in scope that the
        query matches as a whole (BM25 over the documents table, the first
        DOCUMENT_FETCH of them). The bonus reorders the chunks the two chunk searches
        found and never adds one: a filter or scope that excluded a chunk still does.

        Results are returned in fused-rank order. A result carries a `distance` only
        when it appeared in the vector search's fetch window; a hit found by FTS alone
        has `distance is None`, and the two distance-based filters treat that case
        explicitly rather than letting such rows slip through unjudged:

        - `max_distance` drops every result without a distance. The caller asked for a
          distance bound, and a row that was never ranked by distance cannot be shown
          to satisfy it.
        - `grouping` ("similar" | "related") cuts at a natural gap in the distances
          sorted **ascending**, considering only results that have one; results without
          a distance are kept unconditionally, since a distance-gap rule has nothing to
          judge them by. Surviving results keep their fused-rank order.

        With a `reranker`, the first `rerank_top` results left after those two filters are
        rescored against the query text and re-sorted by descending score (ties keep
        fused order); the rest follow in fused order. A reranker that raises, or returns
        the wrong number of scores, leaves the fused order untouched.

        `max_files` then keeps only the first N distinct sources in rank order, and the
        list is truncated to `top_k` last.

        With `include_parents`, each surviving result also carries `parent_text`: the
        enclosing section, rebuilt from the chunk's siblings. Ranking is unaffected —
        the retrieval unit is what was scored, and the section is only what is read.
        """
        fetch = max(top_k * 4, 50, rerank_top if reranker else 0)
        clause = _and_clauses(sql_clause(scopes), filter_sql)

        vq = self._table.search(query_vector).limit(fetch)
        if clause:
            vq = vq.where(clause, prefilter=True)
        # LanceDB returns rows with equal scores in no fixed order, and the fusion below
        # takes rank as given: ties are broken by id, so one query ranks the same every time.
        vrows = sorted(vq.to_list(), key=lambda r: (r["_distance"], r["id"]))
        dist_by_id = {r["id"]: r["_distance"] for r in vrows}

        fts_rows: list[dict] = []
        rrf_scores: dict[str, float] = {}

        if hybrid_weight <= 0.0 and document_weight <= 0.0:
            ordered = vrows
        else:
            if hybrid_weight > 0.0:
                try:
                    fq = self._table.search(
                        query_text, query_type="fts", fts_columns=list(FTS_COLUMNS)
                    ).limit(fetch)
                    if clause:
                        fq = fq.where(clause)
                    fts_rows = sorted(fq.to_list(), key=lambda r: (-r["_score"], r["id"]))
                except Exception:
                    fts_rows = []  # malformed FTS query degrades to vector-only ranking

            # Fuse vector and FTS results by reciprocal rank
            by_id = {r["id"]: r for r in vrows}
            by_id.update({r["id"]: r for r in fts_rows})
            rrf_scores = _weighted_rrf(
                [r["id"] for r in vrows],
                [r["id"] for r in fts_rows],
                hybrid_weight,
            )
            if document_weight > 0.0:
                # The documents table has no label columns, so only the scope narrows it.
                ranks = self._document_ranks(query_text, sql_clause(scopes))
                for rid in rrf_scores:
                    rank = ranks.get(by_id[rid]["source"])
                    if rank is not None:
                        rrf_scores[rid] += document_weight / (RRF_K + rank + 1)
            ordered = [
                by_id[rid] for rid, _ in sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))
            ]

        results: list[SearchResult] = []
        for r in ordered:
            distance = dist_by_id.get(r["id"], r.get("_distance"))
            if max_distance is not None and (distance is None or distance > max_distance):
                continue  # an FTS-only hit has no distance, so it cannot meet the bound
            score = rrf_scores.get(r["id"])
            if score is None:
                score = 1.0 / (1.0 + distance) if distance is not None else 0.0
            results.append(
                SearchResult(
                    text=r["text"],
                    source=r["source"],
                    title=r["title"],
                    chunk_index=r["chunk_index"],
                    score=float(score),
                    distance=distance,
                    parent_id=r.get("parent_id", ""),
                    labels_json=r.get("labels_json", ""),
                    doc_date=r.get("doc_date", ""),
                    doc_date_source=r.get("doc_date_source", ""),
                    pages=r.get("pages") or "",
                )
            )

        if grouping in ("similar", "related") and results:
            # The gap rule needs an ascending distance ordering, not the RRF order the
            # results are in. Judge only the results that carry a distance; keep the
            # FTS-only ones, which were never ranked by distance at all. Matching on
            # the surviving distances means exact ties across the boundary survive
            # together — identical distances give no basis for splitting them.
            distances = sorted(r.distance for r in results if r.distance is not None)
            if distances:
                kept = set(distances[: relevance_cutoff(distances, grouping)])
                results = [r for r in results if r.distance is None or r.distance in kept]

        if reranker is not None and results:
            results = _rerank(results, query_text, reranker, rerank_top)

        if max_files is not None:
            keep: list[SearchResult] = []
            seen: list[str] = []
            for r in results:
                if r.source not in seen:
                    if len(seen) >= max_files:
                        continue
                    seen.append(r.source)
                keep.append(r)
            results = keep

        results = results[:top_k]
        if include_parents and results:
            # One batched lookup after truncation, so a large fetch window never turns
            # into a large parent scan.
            parents = self.parent_texts([r.parent_id for r in results])
            results = [
                replace(r, parent_text=parents.get(r.parent_id)) if r.parent_id else r
                for r in results
            ]
        return results
