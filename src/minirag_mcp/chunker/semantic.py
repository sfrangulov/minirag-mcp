"""Stage 2 chunking: merge adjacent structural blocks by embedding similarity."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from minirag_mcp.chunker.structural import Block

SEPARATOR = "\n\n"


@dataclass(frozen=True)
class MergedChunk:
    """A chunk, with the embedding already computed for it when one exists.

    `vector` is set only when `text` is one original block, unchanged — the vector
    merging computed for that block is then still the vector of the chunk. It is None
    for every chunk built from more than one block, and callers that alter `text`
    afterwards must drop the vector too: a stale vector indexes the wrong text.
    """

    text: str
    vector: list[float] | None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def merge_blocks(
    blocks: list[Block],
    embed: Callable[[Sequence[str]], list[list[float]]],
    *,
    max_chars: int = 1500,
    min_length: int = 50,
    similarity_threshold: float = 0.60,
) -> list[MergedChunk]:
    """Merge adjacent blocks by embedding similarity, with size and code-block rules.

    Every block is embedded here to decide the merges, so each unmerged chunk comes
    back carrying that embedding: the caller embeds only what merging changed.

    The final fold pass prioritizes "never emit a sub-min_length chunk" over the
    max_chars target: folding may exceed max_chars by up to min_length-1 chars
    (plus whatever an atomic code block already exceeds it by). max_chars is a
    sizing heuristic, not a hard limit."""
    if not blocks:
        return []
    vectors = embed([b.text for b in blocks])

    chunks: list[MergedChunk] = [MergedChunk(blocks[0].text, vectors[0])]
    prev_vec = vectors[0]
    for block, vec in zip(blocks[1:], vectors[1:], strict=True):
        candidate = chunks[-1].text + SEPARATOR + block.text
        fits = len(candidate) <= max_chars
        similar = block.is_code or cosine(prev_vec, vec) >= similarity_threshold
        if fits and similar:
            chunks[-1] = MergedChunk(candidate, None)
        else:
            chunks.append(MergedChunk(block.text, vec))
        prev_vec = vec

    # Fold under-length chunks into a neighbor (previous if any, else next).
    folded: list[MergedChunk] = []
    for chunk in chunks:
        if folded and len(folded[-1].text) < min_length:
            folded[-1] = MergedChunk(folded[-1].text + SEPARATOR + chunk.text, None)
        else:
            folded.append(chunk)
    if len(folded) >= 2 and len(folded[-1].text) < min_length:
        tail = folded.pop()
        folded[-1] = MergedChunk(folded[-1].text + SEPARATOR + tail.text, None)
    return folded
