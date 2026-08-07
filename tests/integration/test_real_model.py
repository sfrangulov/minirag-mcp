"""Slow tests: real model download (~220 MB). Run with: uv run pytest -m slow"""

import pytest

from minirag_mcp.config import DEFAULT_MODEL
from minirag_mcp.embedder import Embedder

pytestmark = pytest.mark.slow


def test_real_embeddings_shape_and_similarity(tmp_path):
    emb = Embedder(DEFAULT_MODEL, cache_dir=tmp_path / "models")

    # Short 2-3 word fragments do NOT reliably show cross-lingual alignment in
    # this model — measured 0.44 same-topic vs 0.56 unrelated on fragments
    # like "Аутентификация через токен" / "Token-based authentication". The
    # signal only emerges with sentence-length input, which is also what real
    # chunks look like, so that's what this test uses.
    ru_auth = "Аутентификация пользователя выполняется через OAuth2 токен доступа."
    en_auth = "User authentication is performed with an OAuth2 access token."
    ru_cook = "Борщ варят из свёклы, капусты и говядины на медленном огне."
    vecs = emb.embed_documents([ru_auth, en_auth, ru_cook])
    assert len(vecs) == 3 and all(len(v) == 384 for v in vecs)
    assert all(abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-6 for v in vecs)  # unit length

    from minirag_mcp.chunker.semantic import cosine

    cross_lingual = cosine(vecs[0], vecs[1])
    unrelated = cosine(vecs[0], vecs[2])
    assert cross_lingual > 0.7, f"RU/EN same topic should match strongly, got {cross_lingual}"
    assert cross_lingual - unrelated > 0.5, (
        f"topic must dominate language, got {cross_lingual} vs {unrelated}"
    )

    q = emb.embed_query("аутентификация")
    assert len(q) == 384
