import math

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from minirag_mcp.chunker import MODEL_MAX_TOKENS, chunk_markdown
from minirag_mcp.config import DEFAULT_MODEL, load_config
from minirag_mcp.embedder import Embedder, UnknownModelError, _unit


def test_dim_from_registry_without_download(tmp_path):
    emb = Embedder(DEFAULT_MODEL, cache_dir=tmp_path)
    assert emb.dim == 384
    assert not any(tmp_path.iterdir())  # nothing downloaded


def test_unknown_model_dim_raises(tmp_path):
    emb = Embedder("no-such/model", cache_dir=tmp_path)
    with pytest.raises(UnknownModelError):
        _ = emb.dim


def test_lazy_no_model_instantiation_on_init(tmp_path, monkeypatch):
    import minirag_mcp.embedder as mod

    def boom(*a, **k):
        raise AssertionError("TextEmbedding must not be constructed in __init__")

    monkeypatch.setattr(mod, "TextEmbedding", boom)
    Embedder(DEFAULT_MODEL, cache_dir=tmp_path)  # must not raise


def test_unit_scales_to_norm_one():
    scaled = _unit([3.0, 4.0])
    norm = math.sqrt(sum(x * x for x in scaled))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)
    assert scaled == pytest.approx([0.6, 0.8])


def test_unit_leaves_zero_vector_alone():
    assert _unit([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


class _FakeBackend:
    """fastembed's ONNX backend, reduced to the one attribute the counter reads."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer


class _FakeTextEmbedding:
    """A stand-in for `TextEmbedding` whose tokenizer truncates like the real one.

    `fastembed.common.preprocessor_utils.load_tokenizer` calls
    `enable_truncation(max_length=<the model's trained sequence length>)`, and
    `TextEmbedding.token_count` sums the attention mask of that truncated encoding —
    so it can never report a number above the ceiling. This reproduces both, over a
    real `tokenizers.Tokenizer`, without the 220 MB download.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.model = _FakeBackend(tokenizer)

    def token_count(self, text: str) -> int:
        return sum(self.model.tokenizer.encode(text).attention_mask)


def _truncating_embedder(tmp_path, monkeypatch) -> tuple[Embedder, Tokenizer]:
    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_truncation(max_length=MODEL_MAX_TOKENS)
    emb = Embedder(DEFAULT_MODEL, cache_dir=tmp_path)
    monkeypatch.setattr(emb, "_load", lambda: _FakeTextEmbedding(tokenizer))
    honest = Tokenizer.from_str(tokenizer.to_str())
    honest.no_truncation()
    return emb, honest


def test_the_token_counter_does_not_saturate_at_the_model_ceiling(tmp_path, monkeypatch):
    """fastembed's own `token_count` cannot report more than the trained sequence
    length, which makes it useless as a budget comparison near that length. The
    chunker's counter has to keep counting past it."""
    emb, honest = _truncating_embedder(tmp_path, monkeypatch)
    long = " ".join(f"слово{i}" for i in range(500))
    assert emb._load().token_count(long) == MODEL_MAX_TOKENS  # what fastembed reports
    assert len(honest.encode(long).ids) == 500  # what is actually there
    assert emb.count_tokens(long) == 500


def test_chunks_fit_the_encoder_at_the_documented_maximum_budget(tmp_path, monkeypatch):
    """The maximum `CHUNK_TOKEN_BUDGET` the configuration accepts must still be a
    budget. Against a counter that saturates at the encoder's ceiling it is not: every
    `count <= budget` is trivially true there, the packer stops cutting, and the
    document comes out as one chunk many times the ceiling."""
    emb, honest = _truncating_embedder(tmp_path, monkeypatch)
    cfg = load_config({"CHUNK_TOKEN_BUDGET": str(MODEL_MAX_TOKENS)}, cwd=tmp_path)
    budget = cfg.token_budget
    assert budget == MODEL_MAX_TOKENS, "the documented maximum has to be accepted"

    markdown = "# Раздел 1\n\n" + " ".join(f"термин{i} определение{i}" for i in range(400))
    chunks = chunk_markdown(markdown, count_tokens=emb.count_tokens, budget=budget)
    assert len(chunks) > 1, "800 words cannot be one chunk of at most 128 tokens"
    for c in chunks:
        assert len(honest.encode(c.text).ids) <= MODEL_MAX_TOKENS, repr(c.text[:120])
