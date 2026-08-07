"""Shared fixtures: deterministic fake embedder (no model download), offline DNS."""

from __future__ import annotations

import hashlib
import math
import socket
from collections.abc import Sequence

import pytest

PUBLIC_IP = "93.184.216.34"


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


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every hostname to one fixed public address, without a resolver.

    ingest_url validates the host before fetching, so a test that ingests a URL
    would otherwise depend on a working resolver and on what a real name happens
    to resolve to today.
    """

    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port or 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
