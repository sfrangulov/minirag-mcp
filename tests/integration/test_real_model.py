"""Slow tests: real model download (~220 MB). Run with: uv run pytest -m slow"""

import os
from pathlib import Path

import pytest

from corpus_samples import SLIDES, SPEC, SPREADSHEET, TRANSCRIPT
from minirag_mcp.chunker import DEFAULT_TOKEN_BUDGET, MODEL_MAX_TOKENS, chunk_markdown
from minirag_mcp.chunker.tokens import estimate_tokens
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

    # The count must NOT saturate at the model's ceiling. fastembed's own
    # `TextEmbedding.token_count` does — the tokenizer it hands out has truncation
    # enabled — and a budget compared against a saturating count stops discriminating as
    # it approaches that ceiling, which is what made CHUNK_TOKEN_BUDGET=128 mean "no
    # budget at all".
    long = "слово " * 500
    assert emb._load().token_count(long) == MODEL_MAX_TOKENS  # the saturating counter
    assert emb.count_tokens(long) > 500  # ours keeps counting

    # A chunk packed to the budget must survive the encoder without truncation — at the
    # default budget and at the largest one the configuration accepts.
    for budget in (DEFAULT_TOKEN_BUDGET, MODEL_MAX_TOKENS):
        packed = chunk_markdown(
            "# Раздел 1\n\n" + " ".join(f"термин{i} определение" for i in range(200)),
            count_tokens=emb.count_tokens,
            title="Спецификация",
            budget=budget,
        )
        assert packed
        assert all(emb.count_tokens(c.text) <= budget for c in packed), budget


def test_the_fallback_estimate_stays_on_the_safe_side_of_the_real_tokenizer(model_cache):
    """The estimate is what the chunker falls back to when the tokenizer breaks, so it
    has to err towards *more* tokens — otherwise the fallback quietly produces the
    truncated vectors this whole design exists to prevent.

    The design sketch put the ratio at 6.33 chars/token, which fails both assertions
    below by a wide margin: that number came from a count the tokenizer had already
    truncated at 128. Measured untruncated, this corpus runs at 3.45 chars/token
    overall and 2.2 for markdown table rows.
    """
    emb = Embedder(DEFAULT_MODEL, cache_dir=model_cache)
    pieces = [
        line
        for sample in (SPEC, TRANSCRIPT, SPREADSHEET, SLIDES)
        for line in sample.split("\n")
        if line.strip()
    ]
    assert len(pieces) > 40
    estimated = [estimate_tokens(p) for p in pieces]
    actual = [emb.count_tokens(p) for p in pieces]
    # in aggregate — which is what the packer accumulates — it must never run short
    assert sum(estimated) > sum(actual)
    # and no single piece may be badly under-estimated. Not a bound: no character
    # heuristic can be exact about `ax_ИсторияЗаменыТМЗКЗакупу`.
    worst = min(e / a for e, a in zip(estimated, actual, strict=True))
    assert worst > 0.7, (
        worst,
        min(zip(pieces, estimated, actual, strict=True), key=lambda t: t[1] / t[2]),
    )
