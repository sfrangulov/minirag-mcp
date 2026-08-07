"""Slow tests: real model download (~220 MB). Run with: uv run pytest -m slow"""

import pytest

from minirag_mcp.config import DEFAULT_MODEL
from minirag_mcp.embedder import Embedder

pytestmark = pytest.mark.slow


def test_real_embeddings_shape_and_similarity(tmp_path):
    emb = Embedder(DEFAULT_MODEL, cache_dir=tmp_path / "models")
    texts = [
        "Аутентификация через токен",
        "Token-based authentication",
        "Рецепт борща",
    ]
    vecs = emb.embed_documents(texts)
    assert len(vecs) == 3 and all(len(v) == 384 for v in vecs)
    from minirag_mcp.chunker.semantic import cosine

    ru_en = cosine(vecs[0], vecs[1])
    ru_off = cosine(vecs[0], vecs[2])
    assert ru_en > ru_off  # multilingual model: RU/EN same topic closer than unrelated RU
    q = emb.embed_query("аутентификация")
    assert len(q) == 384
