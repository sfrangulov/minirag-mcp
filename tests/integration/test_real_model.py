"""Slow tests: real model downloads (~2.5 GB, e5-large and MiniLM).

Run with: uv run pytest -m slow
"""

import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from corpus_samples import SLIDES, SPEC, SPREADSHEET, TRANSCRIPT
from minirag_mcp.chunker import DEFAULT_TOKEN_BUDGET, MODEL_MAX_TOKENS, chunk_markdown
from minirag_mcp.chunker.tokens import estimate_tokens
from minirag_mcp.config import load_config
from minirag_mcp.embedder import (
    E5_LARGE,
    MINILM,
    PASSAGE_PREFIX,
    Embedder,
    cosine,
    default_budget,
)
from minirag_mcp.endpoints import EmbeddingEndpoint, Endpoint
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.store import Store

pytestmark = pytest.mark.slow

# CI points this at a restored cache so the download happens once, not once per run.
CACHE_ENV = "MINIRAG_TEST_MODEL_CACHE"


@pytest.fixture
def model_cache(tmp_path) -> Path:
    shared = os.environ.get(CACHE_ENV)
    return Path(shared) if shared else tmp_path / "models"


# Short 2-3 word fragments do NOT reliably show cross-lingual alignment in MiniLM —
# measured 0.44 same-topic vs 0.56 unrelated on fragments like "Аутентификация через
# токен" / "Token-based authentication". The signal only emerges with sentence-length
# input, which is also what real chunks look like, so that's what these tests use.
RU_AUTH = "Аутентификация пользователя выполняется через OAuth2 токен доступа."
EN_AUTH = "User authentication is performed with an OAuth2 access token."
RU_COOK = "Борщ варят из свёклы, капусты и говядины на медленном огне."
CORPUS = (SPEC, TRANSCRIPT, SPREADSHEET, SLIDES)
# Never called: an endpoint embedder counts and cuts with the tokenizer file alone.
UNUSED_ENDPOINT = Endpoint("http://127.0.0.1:9/v1/embeddings", "e5")


def test_real_embeddings_shape_and_similarity(model_cache):
    emb = Embedder(MINILM, cache_dir=model_cache)
    vecs = emb.embed_documents([RU_AUTH, EN_AUTH, RU_COOK])
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
    emb = Embedder(MINILM, cache_dir=model_cache)
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
    emb = Embedder(MINILM, cache_dir=model_cache)
    pieces = [line for sample in CORPUS for line in sample.split("\n") if line.strip()]
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


def test_real_e5_vectors_prefixes_and_window(model_cache):
    emb = Embedder(E5_LARGE, cache_dir=model_cache)
    v = emb.embed_documents([RU_AUTH, EN_AUTH, RU_COOK])
    assert all(len(x) == 1024 and abs(math.hypot(*x) - 1) < 1e-6 for x in v)
    assert cosine(v[0], v[1]) > cosine(v[0], v[2])
    # Without the prefixes a query and a passage of the same text are one vector.
    assert cosine(emb.embed_query(RU_AUTH), v[0]) < 0.999
    assert emb.max_tokens == 512 and emb.prefix_tokens > 0
    assert emb._load().token_count("слово " * 1000) == 512


def test_real_e5_tokenizer_file_counts_like_the_loaded_model(model_cache):
    in_process = Embedder(E5_LARGE, cache_dir=model_cache)
    by_file = Embedder(E5_LARGE, cache_dir=model_cache, endpoint=EmbeddingEndpoint(UNUSED_ENDPOINT))
    lines = [line for sample in CORPUS for line in sample.split("\n") if line.strip()]
    lines.append("\n\n".join(CORPUS))  # over the window: neither count may saturate
    assert [by_file.count_tokens(t) for t in lines] == [in_process.count_tokens(t) for t in lines]
    assert by_file.count_tokens(lines[-1]) > 512
    assert by_file.prefix_tokens == in_process.prefix_tokens
    assert by_file._model is None  # counted without the ONNX model


def test_the_endpoint_gets_the_tokens_fastembed_embeds(model_cache):
    """Review focus: text over the window is cut where fastembed truncates, so both
    backends embed the same tokens and llama.cpp never sees more than the window."""
    in_process = Embedder(E5_LARGE, cache_dir=model_cache)
    by_file = Embedder(E5_LARGE, cache_dir=model_cache, endpoint=EmbeddingEndpoint(UNUSED_ENDPOINT))
    loaded = in_process._load().model.tokenizer  # truncates at the window, as fastembed embeds
    counting, _ = by_file._tokenizers()
    for text in ("\n\n".join(CORPUS), "\n".join(reversed(CORPUS)), "слово " * 1000):
        full = PASSAGE_PREFIX + text
        cut = by_file._fit_window(full)
        assert len(cut) < len(full)
        assert counting.encode(cut).ids == loaded.encode(full).ids


def test_e5_chunks_fit_the_window_with_the_prefix(model_cache, tmp_path):
    emb = Embedder(E5_LARGE, cache_dir=model_cache)
    markdown = "# Раздел 1\n\n" + " ".join(f"термин{i} определение" for i in range(400))
    base = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    for budget in (default_budget(E5_LARGE), 512):
        pipeline = Pipeline(
            Store(tmp_path / f"db-{budget}", dim=emb.dim), emb, replace(base, token_budget=budget)
        )
        pipeline.ingest_data(markdown, source="s", fmt="markdown")
        chunks = pipeline.store.all_chunks("s")
        assert len(chunks) > 1, budget
        for c in chunks:
            assert emb.count_tokens(c.text) + emb.prefix_tokens <= budget, budget
            assert emb.count_tokens(PASSAGE_PREFIX + c.text) <= budget, budget
