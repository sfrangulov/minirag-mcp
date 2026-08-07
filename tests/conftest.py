"""Shared fixtures: deterministic fake embedder (no model download)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import pytest


class FakeEmbedder:
    """Deterministic 8-dim embeddings from sha256 — same text, same vector."""

    dim = 8
    model_name = "fake"

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [int.from_bytes(h[i : i + 4], "big") / 2**32 - 0.5 for i in range(0, 32, 4)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
