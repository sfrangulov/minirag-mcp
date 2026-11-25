"""A client that leaves its session mid-call stops the read it started, over real HTTP."""

from __future__ import annotations

import asyncio
import time

import minirag_mcp.ingest.pipeline as pipeline_mod
import minirag_mcp.literal as literal_mod
from http_helpers import abandon_call, call_tool
from minirag_mcp.store import Store


def _corpus(root) -> str:
    """A Markdown file of many sections, each its own chunk and each mentioning apricot."""
    sections = [
        f"## Section {i}\n\n"
        + " ".join(f"apricot{i} orchard{i} harvest{j}" for j in range(60))
        + f" apricot number {i}.\n"
        for i in range(40)
    ]
    path = root / "fruit.md"
    path.write_text("# Fruit\n\n" + "\n".join(sections), encoding="utf-8")
    return str(path)


def _slow_matches(monkeypatch, calls: list[float], delay: float) -> None:
    real = literal_mod.matches

    def matches(*args, **kwargs):
        calls.append(time.monotonic())
        if delay:
            time.sleep(delay)
        return real(*args, **kwargs)

    monkeypatch.setattr(literal_mod, "matches", matches)


async def test_an_abandoned_find_mentions_stops_scanning(site, indexes_file, monkeypatch):
    path = _corpus(indexes_file.roots["alpha"])
    await call_tool(site, "alpha", "ingest_file", {"filePath": path})
    args = {"terms": ["apricot"]}

    full: list[float] = []
    _slow_matches(monkeypatch, full, 0)
    await call_tool(site, "alpha", "find_mentions", args)
    assert len(full) >= 20

    calls: list[float] = []
    _slow_matches(monkeypatch, calls, 0.05)
    await abandon_call(site, "alpha", "find_mentions", args, after_s=0.3)
    abandoned_at = time.monotonic()
    await asyncio.sleep(1.5)

    assert calls
    assert calls[-1] - abandoned_at < 0.5
    assert len(calls) < len(full)


async def test_an_abandoned_query_does_not_search_after_embedding(
    site, indexes_file, fake_embedder, monkeypatch
):
    path = _corpus(indexes_file.roots["alpha"])
    await call_tool(site, "alpha", "ingest_file", {"filePath": path})

    real_embed = fake_embedder.embed_query

    def slow_embed(text):
        time.sleep(1.0)
        return real_embed(text)

    searched: list[str] = []
    real_search = Store.search

    def search(self, *args, **kwargs):
        searched.append("called")
        return real_search(self, *args, **kwargs)

    monkeypatch.setattr(fake_embedder, "embed_query", slow_embed)
    monkeypatch.setattr(Store, "search", search)

    await abandon_call(site, "alpha", "query_documents", {"query": "apricot"}, after_s=0.3)
    await asyncio.sleep(1.5)

    assert searched == []


async def test_an_abandoned_ingest_still_completes(site, indexes_file, monkeypatch):
    """Guard for the anti-goal: writes are never stopped. It passes without the fix too."""
    path = _corpus(indexes_file.roots["alpha"])
    real_parse = pipeline_mod.parse_file

    def slow_parse(*args, **kwargs):
        time.sleep(1.0)
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(pipeline_mod, "parse_file", slow_parse)

    await abandon_call(site, "alpha", "ingest_file", {"filePath": path}, after_s=0.3)
    await asyncio.sleep(2.0)

    listed = await call_tool(site, "alpha", "list_files", {})
    assert "fruit.md" in str(listed.structuredContent)
    assert "ingested" in str(listed.structuredContent)
