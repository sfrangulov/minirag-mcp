"""LanceDB persistence: one `chunks` table + BM25 FTS index."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.index import FTS

TABLE = "chunks"
_LIST_LIMIT = 2**31 - 1  # LanceDB scalar queries default to limit 10 — always set explicitly
RRF_K = 60  # standard RRF damping constant
GAP_FACTOR = 2.0  # relevance_cutoff boundary must exceed mean gap by this factor


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


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    title: str
    chunk_index: int
    score: float
    distance: float | None


@dataclass(frozen=True)
class SourceInfo:
    source: str
    source_type: str
    title: str
    chunk_count: int
    file_hash: str
    mtime: float


def _sql_str(s: str) -> str:
    return s.replace("'", "''")


def _scope_clause(scopes: tuple[str, ...]) -> str | None:
    if not scopes:
        return None
    return " OR ".join(f"starts_with(source, '{_sql_str(p)}')" for p in scopes)


_META_COLS = ["source", "source_type", "title", "chunk_index", "file_hash", "mtime"]


def relevance_cutoff(distances: Sequence[float], mode: str) -> int:
    """Identify natural gaps in distance ordering for clustering into relevance groups.

    A boundary occurs where gap > mean(gaps) * GAP_FACTOR. This avoids spurious
    cuts from jitter and detects only materially significant distance jumps.

    Args:
        distances: sorted sequence of distances (e.g., from vector search)
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
    boundaries = [i + 1 for i, g in enumerate(gaps) if g > threshold and g > 0]
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


class Store:
    def __init__(self, db_path: Path, dim: int):
        db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(db_path))
        self.dim = dim
        try:
            self._table = self._db.open_table(TABLE)
        except Exception:
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
                ]
            )
            self._table = self._db.create_table(TABLE, schema=schema)
            self._table.create_index("text", config=FTS())

    def replace_source(self, source: str, records: Sequence[ChunkRecord]) -> None:
        self._table.delete(f"source = '{_sql_str(source)}'")
        if records:
            self._table.add([asdict(r) for r in records])

    def delete_source(self, source: str) -> int:
        clause = f"source = '{_sql_str(source)}'"
        before = len(self._table.search().where(clause).select(["id"]).limit(_LIST_LIMIT).to_list())
        if before:
            self._table.delete(clause)
        return before

    def neighbors(
        self, source: str, chunk_index: int, before: int = 1, after: int = 1
    ) -> list[SearchResult]:
        lo, hi = max(0, chunk_index - before), chunk_index + after
        clause = f"source = '{_sql_str(source)}' AND chunk_index >= {lo} AND chunk_index <= {hi}"
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
            )
            for r in rows
        ]

    def all_chunks(self, source: str) -> list[SearchResult]:
        clause = f"source = '{_sql_str(source)}'"
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
            )
            for r in rows
        ]

    def _iter_meta(self, scopes: tuple[str, ...] = ()) -> list[dict]:
        q = self._table.search().select(_META_COLS)
        clause = _scope_clause(scopes)
        if clause:
            q = q.where(clause)
        return q.limit(_LIST_LIMIT).to_list()

    def list_sources(self, scopes: tuple[str, ...] = ()) -> list[SourceInfo]:
        by_source: dict[str, dict] = {}
        counts: dict[str, int] = {}
        for row in self._iter_meta(scopes):
            by_source.setdefault(row["source"], row)
            counts[row["source"]] = counts.get(row["source"], 0) + 1
        return [
            SourceInfo(
                source=src,
                source_type=row["source_type"],
                title=row["title"],
                chunk_count=counts[src],
                file_hash=row["file_hash"],
                mtime=row["mtime"],
            )
            for src, row in sorted(by_source.items())
        ]

    def get_source(self, source: str) -> SourceInfo | None:
        rows = (
            self._table.search()
            .where(f"source = '{_sql_str(source)}'")
            .select(_META_COLS)
            .limit(_LIST_LIMIT)
            .to_list()
        )
        if not rows:
            return None
        r = rows[0]
        return SourceInfo(
            source=r["source"],
            source_type=r["source_type"],
            title=r["title"],
            chunk_count=len(rows),
            file_hash=r["file_hash"],
            mtime=r["mtime"],
        )

    def chunk_count(self) -> int:
        return self._table.count_rows()

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
    ) -> list[SearchResult]:
        fetch = max(top_k * 4, 50)
        clause = _scope_clause(scopes)

        vq = self._table.search(query_vector).limit(fetch)
        if clause:
            vq = vq.where(clause, prefilter=True)
        vrows = vq.to_list()
        dist_by_id = {r["id"]: r["_distance"] for r in vrows}

        fts_rows: list[dict] = []
        rrf_scores: dict[str, float] = {}

        if hybrid_weight <= 0.0:
            ordered = vrows
        else:
            try:
                fq = self._table.search(query_text, query_type="fts").limit(fetch)
                if clause:
                    fq = fq.where(clause)
                fts_rows = fq.to_list()
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
            ordered = [
                by_id[rid]
                for rid, _ in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
            ]

        results: list[SearchResult] = []
        for r in ordered:
            distance = dist_by_id.get(r["id"], r.get("_distance"))
            if max_distance is not None and distance is not None and distance > max_distance:
                continue
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
                )
            )

        if (
            grouping in ("similar", "related")
            and results
            and all(r.distance is not None for r in results)
        ):
            cut = relevance_cutoff([r.distance for r in results], grouping)
            results = results[:cut]

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

        return results[:top_k]
