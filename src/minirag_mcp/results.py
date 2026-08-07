"""Wire-shape for search results, shared by the MCP server and the CLI.

Both interfaces must describe a hit the same way — divergence here is how the CLI
ended up returning fewer fields than the server for the same query. Anything that
shapes a SearchResult for output belongs in this module, imported by both sides.
"""

from __future__ import annotations

from collections.abc import Iterable

from minirag_mcp.store import SearchResult


def result_dict(r: SearchResult) -> dict:
    """One search hit as its JSON object. Keys are camelCase, matching the MCP tools."""
    return {
        "text": r.text,
        "source": r.source,
        "title": r.title,
        "chunkIndex": r.chunk_index,
        "score": r.score,
        "distance": r.distance,
    }


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
