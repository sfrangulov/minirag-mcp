"""Chunking: structure-first sectioning under a token budget.

Two units, deliberately separated (the small-to-big / parent-document pattern):

- the **retrieval unit** is what gets embedded and ranked, and it is sized in the
  embedding model's own tokens, because the model's 128-token ceiling is a token
  ceiling. Text past it is not ranked badly, it is not seen at all.
- the **parent section** is what a caller reads: a transcript time window, a heading
  section, a slide, a table. It is never stored twice — the units of a section share a
  `parent_id`, and `join_parent` rebuilds the section from them in order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from minirag_mcp.chunker.pack import join_parent, pack_section
from minirag_mcp.chunker.sections import detect_category, split_sections
from minirag_mcp.chunker.tokens import CountTokens, estimate_tokens

# Bumped whenever chunk boundaries or chunk text change meaning. Stored on every row so
# an index built by an older scheme is *detected* rather than assumed: its vectors were
# computed over different text and are not comparable with new ones.
SCHEME_VERSION = 3
# The trained sequence length of the multilingual MiniLM models this tool ships with.
MODEL_MAX_TOKENS = 128
# Left under the ceiling on purpose: prefixes, table headers and mixed-script text all
# tokenize worse than the corpus average, and a unit that spills over loses its tail.
DEFAULT_TOKEN_BUDGET = 110

__all__ = [
    "DEFAULT_TOKEN_BUDGET",
    "MODEL_MAX_TOKENS",
    "SCHEME_VERSION",
    "Chunk",
    "chunk_markdown",
    "detect_category",
    "estimate_tokens",
    "join_parent",
]


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit, tagged with the section it was cut from."""

    text: str
    parent_index: int


def _seed_title(body: str, title: str) -> str:
    """Put the document's title at the head of its first section, once.

    Keyword search reads the title column directly, but the vector side only ever sees
    chunk text — so without this a filename-derived title would be invisible to
    semantic search. It goes in before packing, not after, because the budget has to
    count it: seeding a finished chunk pushed 3 of 64,692 chunks on the real corpus
    past the budget, all of them chunk 0. A section that already carries the title is
    left alone, which is also what keeps re-ingest idempotent.
    """
    return body if not title.strip() or title in body else f"# {title}\n\n{body}"


def chunk_markdown(
    markdown: str,
    *,
    count_tokens: CountTokens = estimate_tokens,
    title: str = "",
    budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[Chunk]:
    """Split Markdown into retrieval units, grouped by parent section.

    `count_tokens` must be the embedding model's own tokenizer; the default estimate is
    a fallback for callers that have no model to hand. `title` is written into the
    first section's text and into every transcript window's label — pass it empty when
    the caller has no real title, only a source id or a URL.
    """
    sections = split_sections(markdown, title=title)
    if sections:
        sections[0] = replace(sections[0], body=_seed_title(sections[0].body, title))
    chunks: list[Chunk] = []
    parent = 0
    for section in sections:
        units = pack_section(section, count_tokens, budget)
        if not units:
            continue
        chunks += [Chunk(text=unit, parent_index=parent) for unit in units]
        parent += 1
    return chunks
