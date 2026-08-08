"""Wire-shape for search results, shared by the MCP server and the CLI.

Both interfaces must describe a hit the same way — divergence here is how the CLI
ended up returning fewer fields than the server for the same query. Anything that
shapes a SearchResult for output belongs in this module, imported by both sides.
"""

from __future__ import annotations

from collections.abc import Iterable

from minirag_mcp.store import RESYNC_HINT, SearchResult, Store


def result_dict(r: SearchResult) -> dict:
    """One search hit as its JSON object. Keys are camelCase, matching the MCP tools.

    `parentText` is added alongside `text`, never in place of it. `text` is the
    retrieval unit — the thing that was embedded and the thing `score` describes — and
    a client already reading it would silently start receiving a whole section instead,
    many times larger, for a hit it thought it understood. Adding a field breaks
    nobody; redefining one breaks everybody who was right before.
    """
    return {
        "text": r.text,
        "source": r.source,
        "title": r.title,
        "chunkIndex": r.chunk_index,
        "score": r.score,
        "distance": r.distance,
        "parentId": r.parent_id or None,
        "parentText": r.parent_text,
    }


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


def aggregate_sources(results: Iterable[SearchResult]) -> list[dict]:
    """Distinct sources among `results`, in rank order, each with a hit count.

    The first hit of a source fixes that source's position, so the list answers
    "which documents cover this topic" without inspecting individual chunks.
    """
    sources: dict[str, dict] = {}
    for r in results:
        agg = sources.setdefault(r.source, {"source": r.source, "title": r.title, "hits": 0})
        agg["hits"] += 1
    return list(sources.values())
