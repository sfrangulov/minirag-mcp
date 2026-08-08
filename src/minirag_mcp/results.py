"""Wire-shape for search results, shared by the MCP server and the CLI.

Both interfaces must describe a hit the same way — divergence here is how the CLI
ended up returning fewer fields than the server for the same query. Anything that
shapes a SearchResult for output belongs in this module, imported by both sides.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from minirag_mcp.chunker import join_parent
from minirag_mcp.store import RESYNC_HINT, SearchResult, Store


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
