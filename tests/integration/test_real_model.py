"""Slow tests: real model download (~220 MB). Run with: uv run pytest -m slow"""

import os
from pathlib import Path

import pytest

from minirag_mcp.chunker import DEFAULT_TOKEN_BUDGET, MODEL_MAX_TOKENS, chunk_markdown
from minirag_mcp.config import DEFAULT_MODEL
from minirag_mcp.embedder import Embedder, cosine

pytestmark = pytest.mark.slow

# CI points this at a restored cache so the download happens once, not once per run.
CACHE_ENV = "MINIRAG_TEST_MODEL_CACHE"


@pytest.fixture
def model_cache(tmp_path) -> Path:
    shared = os.environ.get(CACHE_ENV)
    return Path(shared) if shared else tmp_path / "models"


def test_real_embeddings_shape_and_similarity(model_cache):
    emb = Embedder(DEFAULT_MODEL, cache_dir=model_cache)

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

    cross_lingual = cosine(vecs[0], vecs[1])
    unrelated = cosine(vecs[0], vecs[2])
    assert cross_lingual > 0.7, f"RU/EN same topic should match strongly, got {cross_lingual}"
    assert cross_lingual - unrelated > 0.5, (
        f"topic must dominate language, got {cross_lingual} vs {unrelated}"
    )

    q = emb.embed_query("аутентификация")
    assert len(q) == 384


def test_real_token_counter_is_the_models_own(model_cache):
    """The budget is only meaningful if the counter is the encoder's own tokenizer."""
    emb = Embedder(DEFAULT_MODEL, cache_dir=model_cache)
    assert DEFAULT_TOKEN_BUDGET < MODEL_MAX_TOKENS

    # Russian tokenizes into far more tokens than a character count suggests: this is
    # the measurement the whole redesign rests on (~3.3 chars/token on the corpus).
    ru = "Отражение ставки капитализации затрат на вскрышу в документе учетной политики."
    assert emb.count_tokens(ru) > len(ru) / 6
    assert emb.count_tokens("") < emb.count_tokens(ru)

    # Saturation at the model's ceiling is documented behaviour, and is exactly why the
    # budget has to sit below it for the count to be usable as a comparison.
    assert emb.count_tokens("слово " * 500) == MODEL_MAX_TOKENS

    # A chunk packed to the budget must survive the encoder without truncation.
    packed = chunk_markdown(
        "# Раздел 1\n\n" + " ".join(f"термин{i} определение" for i in range(200)),
        count_tokens=emb.count_tokens,
        title="Спецификация",
    )
    assert packed
    assert all(emb.count_tokens(c.text) <= DEFAULT_TOKEN_BUDGET for c in packed)
