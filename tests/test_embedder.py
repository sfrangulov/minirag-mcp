import math
import pickle
import re

import pytest
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import Unigram, WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import minirag_mcp.embedder as embedder_mod
from minirag_mcp.chunker import MODEL_MAX_TOKENS, chunk_markdown
from minirag_mcp.config import DEFAULT_MODEL, load_config
from minirag_mcp.embedder import (
    E5_LARGE,
    MINILM,
    MODEL_WINDOWS,
    Embedder,
    EmbedderUnavailableError,
    UnknownModelError,
    _unit,
    default_budget,
    known_model,
    model_window,
    prefix_rule,
)
from minirag_mcp.endpoints import EmbeddingEndpoint, Endpoint, EndpointError


def test_dim_from_registry_without_download(tmp_path):
    emb = Embedder(MINILM, cache_dir=tmp_path)
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
    emb = Embedder(MINILM, cache_dir=tmp_path)
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
    cfg = load_config(
        {"MODEL_NAME": MINILM, "CHUNK_TOKEN_BUDGET": str(MODEL_MAX_TOKENS)}, cwd=tmp_path
    )
    budget = cfg.token_budget
    assert budget == MODEL_MAX_TOKENS, "the documented maximum has to be accepted"

    markdown = "# Раздел 1\n\n" + " ".join(f"термин{i} определение{i}" for i in range(400))
    chunks = chunk_markdown(markdown, count_tokens=emb.count_tokens, budget=budget)
    assert len(chunks) > 1, "800 words cannot be one chunk of at most 128 tokens"
    for c in chunks:
        assert len(honest.encode(c.text).ids) <= MODEL_MAX_TOKENS, repr(c.text[:120])


def _unloadable(*args, **kwargs):
    # What fastembed raises with no network and nothing in the cache, measured.
    raise ValueError(f"Could not load model {DEFAULT_MODEL} from any source.")


def test_a_model_that_cannot_load_says_what_failed_and_what_unblocks_it(tmp_path, monkeypatch):
    """fastembed's own text names neither the cache it looked in nor what would fix it,
    and nothing in it tells a caller that a reworded query fails the same way."""
    import minirag_mcp.embedder as mod

    monkeypatch.setattr(mod, "TextEmbedding", _unloadable)
    emb = Embedder(DEFAULT_MODEL, cache_dir=tmp_path)
    with pytest.raises(EmbedderUnavailableError) as raised:
        emb.embed_query("anything")
    message = str(raised.value)
    assert "from any source" in message  # the cause survives
    assert str(tmp_path) in message
    assert "huggingface.co" in message
    assert "not a problem with the query" in message


def test_a_failed_load_does_not_leave_the_token_counter_degraded(tmp_path, monkeypatch):
    """A model that cannot load cannot embed either, so falling back to the character
    estimate buys nothing — and the fallback is permanent, so once the download works
    every later document would still be cut by the estimate."""
    import minirag_mcp.embedder as mod

    tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    monkeypatch.setattr(mod, "TextEmbedding", _unloadable)
    emb = Embedder(DEFAULT_MODEL, cache_dir=tmp_path)
    with pytest.raises(EmbedderUnavailableError):
        emb.count_tokens("one two three")
    monkeypatch.setattr(mod, "TextEmbedding", lambda **kw: _FakeTextEmbedding(tokenizer))
    assert emb.count_tokens("one two three") == 3


def test_prefix_rule_marks_the_e5_family():
    assert prefix_rule(E5_LARGE) == prefix_rule("intfloat/multilingual-e5-small") == "e5"
    assert prefix_rule(MINILM) == prefix_rule("BAAI/bge-base-en") == "none"
    assert prefix_rule("other/multilingual-e5-large") == "none"
    assert prefix_rule("intfloat/bge-x") == "none"


def test_model_window_and_default_budget():
    assert (model_window(E5_LARGE), model_window(MINILM), model_window("x/y")) == (512, 128, 128)
    assert default_budget("x/y") == default_budget(MINILM) == 110


def e5(tmp_path, fake_llama, **kwargs) -> Embedder:
    fake_llama.dim = 1024
    endpoint = EmbeddingEndpoint(Endpoint(fake_llama.url("/v1/embeddings"), "e5"))
    return Embedder(E5_LARGE, tmp_path, endpoint=endpoint, **kwargs)


def dead(tmp_path, dead_url, model_name=E5_LARGE, **kwargs) -> Embedder:
    endpoint = EmbeddingEndpoint(Endpoint(dead_url + "/v1/embeddings", "e5"))
    return Embedder(model_name, tmp_path, endpoint=endpoint, **kwargs)


def inputs(fake_llama) -> list[list[str]]:
    return [r.body["input"] for r in fake_llama.requests]


def test_e5_texts_reach_the_endpoint_prefixed(tmp_path, fake_llama, tokenizer_file):
    emb = e5(tmp_path, fake_llama)
    emb.embed_documents(["alpha"])
    emb.embed_query("beta")
    assert inputs(fake_llama) == [["passage: alpha"], ["query: beta"]]
    assert emb.backend == "endpoint" and emb.warning is None


def test_minilm_texts_reach_the_endpoint_unchanged(tmp_path, fake_llama, tokenizer_file):
    fake_llama.dim = 384
    endpoint = EmbeddingEndpoint(Endpoint(fake_llama.url("/v1/embeddings"), "minilm"))
    emb = Embedder(MINILM, tmp_path, endpoint=endpoint)
    emb.embed_documents(["alpha"])
    emb.embed_query("beta")
    assert inputs(fake_llama) == [["alpha"], ["beta"]]
    assert emb.prefixes == "none"


def test_an_explicit_prefix_rule_overrides_the_models_own(tmp_path, fake_llama, tokenizer_file):
    emb = e5(tmp_path, fake_llama, prefixes="none")
    emb.embed_query("beta")
    assert inputs(fake_llama) == [["beta"]]


def test_in_process_e5_uses_the_same_prefixes_and_threads(tmp_path, stub_text_embedding):
    emb = Embedder(E5_LARGE, tmp_path, threads=2)
    emb.embed_documents(["alpha"])
    emb.embed_query("beta")
    assert stub_text_embedding.batches == [["passage: alpha"], ["query: beta"]]
    assert stub_text_embedding.model_names == [E5_LARGE]
    assert stub_text_embedding.threads == [2] and emb.threads == 2
    assert emb.backend == "in-process" and emb.warning is None


def test_an_unreachable_endpoint_falls_back_in_process_loudly(
    tmp_path, dead_url, stub_text_embedding, tokenizer_file
):
    emb = dead(tmp_path, dead_url)
    with pytest.warns(RuntimeWarning, match="unreachable"):
        assert len(emb.embed_documents(["alpha"])) == 1
    assert stub_text_embedding.model_names == [E5_LARGE]
    assert stub_text_embedding.batches == [["passage: alpha"]]
    assert emb.backend == "in-process (endpoint unreachable)"
    assert dead_url in emb.warning and "every 60 s" in emb.warning


def test_a_fallback_warns_once_until_the_endpoint_is_back(
    tmp_path, fake_llama, stub_text_embedding, tokenizer_file, monkeypatch, recwarn
):
    monkeypatch.setattr(embedder_mod, "ENDPOINT_RETRY_SECONDS", 0.0)
    emb = e5(tmp_path, fake_llama)
    fake_llama.fail["/v1/embeddings"] = (503, "Loading model")
    emb.embed_documents(["alpha"])
    emb.embed_query("beta")
    assert len(fake_llama.requests) == 2  # tried again, and failed again
    assert len([w for w in recwarn if issubclass(w.category, RuntimeWarning)]) == 1
    del fake_llama.fail["/v1/embeddings"]
    emb.embed_query("gamma")
    fake_llama.fail["/v1/embeddings"] = (503, "Loading model")
    emb.embed_query("delta")
    assert len([w for w in recwarn if issubclass(w.category, RuntimeWarning)]) == 2
    assert stub_text_embedding.batches == [["passage: alpha"], ["query: beta"], ["query: delta"]]


def test_a_fallback_that_cannot_load_names_the_endpoint_too(
    tmp_path, dead_url, tokenizer_file, monkeypatch
):
    real = embedder_mod.TextEmbedding

    class Unloadable:
        list_supported_models = staticmethod(real.list_supported_models)

        def __init__(self, *args, **kwargs):
            raise ValueError(f"Could not load model {E5_LARGE} from any source.")

    monkeypatch.setattr(embedder_mod, "TextEmbedding", Unloadable)
    emb = dead(tmp_path, dead_url)
    with pytest.warns(RuntimeWarning), pytest.raises(EmbedderUnavailableError) as raised:
        emb.embed_query("alpha")
    message = str(raised.value)
    assert dead_url in message and "unreachable" in message
    assert "from any source" in message and str(tmp_path) in message


def test_the_endpoint_is_tried_again_after_the_interval(
    tmp_path, fake_llama, stub_text_embedding, tokenizer_file, monkeypatch
):
    emb = e5(tmp_path, fake_llama)
    fake_llama.fail["/v1/embeddings"] = (503, "Loading model")
    with pytest.warns(RuntimeWarning, match="503"):
        emb.embed_documents(["alpha"])
    del fake_llama.fail["/v1/embeddings"]
    emb.embed_documents(["beta"])
    assert len(fake_llama.requests) == 1, "tried again before the interval passed"
    assert emb.backend == "in-process (endpoint unreachable)"

    monkeypatch.setattr(embedder_mod, "ENDPOINT_RETRY_SECONDS", 0.0)
    emb.embed_documents(["gamma"])
    assert inputs(fake_llama)[-1] == ["passage: gamma"]
    assert emb.backend == "endpoint" and emb.warning is None
    assert stub_text_embedding.batches == [["passage: alpha"], ["passage: beta"]]


def test_without_a_fallback_an_unreachable_endpoint_is_an_error(
    tmp_path, dead_url, stub_text_embedding, tokenizer_file
):
    emb = dead(tmp_path, dead_url, allow_fallback=False)
    with pytest.raises(EndpointError, match=re.escape(dead_url)):
        emb.embed_documents(["alpha"])
    assert stub_text_embedding.model_names == []
    assert emb.backend == "endpoint" and emb.warning is None


def test_an_endpoint_of_another_width_is_refused_loudly(
    tmp_path, fake_llama, stub_text_embedding, tokenizer_file
):
    emb = e5(tmp_path, fake_llama)
    fake_llama.dim = 8
    with pytest.raises(EmbedderUnavailableError, match=r"(?s)8.*1024|1024.*8") as raised:
        emb.embed_documents(["a"])
    message = str(raised.value)
    assert fake_llama.url("/v1/embeddings") in message
    assert "'e5'" in message and E5_LARGE in message
    assert stub_text_embedding.batches == [] and stub_text_embedding.model_names == []


def test_text_over_the_window_is_cut_before_the_endpoint_sees_it(
    tmp_path, fake_llama, tokenizer_file, monkeypatch
):
    monkeypatch.setitem(MODEL_WINDOWS, E5_LARGE, 8)
    emb = e5(tmp_path, fake_llama)
    emb.embed_query(" ".join(f"w{i}" for i in range(50)))
    emb.embed_query("w0 w1 w2 w3")
    # [CLS] query : w0 w1 w2 w3 [SEP]: the tokenizer's own truncation keeps room for [SEP].
    assert inputs(fake_llama) == [["query: w0 w1 w2 w3"], ["query: w0 w1 w2 w3"]]
    assert emb.backend == "endpoint"


def test_a_cut_is_never_longer_than_the_window(tmp_path, fake_llama, tokenizer_file, monkeypatch):
    """XLM-R's tokenizer keeps a lone "▁" whose offsets cover the first character of the
    word it was split from. Cut after it, and that character comes back as a token of its
    own, one over the window."""
    tokenizer = Tokenizer(
        Unigram(
            [("<unk>", 0.0), ("[CLS]", 0.0), ("[SEP]", 0.0), ("▁", -2.0), ("東京", -1.0)]
            + [("▁w", -1.0)],
            unk_id=0,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Metaspace()]
    )
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer_file.write(tokenizer)
    assert tokenizer.encode("w w 東").tokens == ["[CLS]", "▁w", "▁w", "▁", "東", "[SEP]"]
    monkeypatch.setitem(MODEL_WINDOWS, MINILM, 5)
    fake_llama.dim = 384
    endpoint = EmbeddingEndpoint(Endpoint(fake_llama.url("/v1/embeddings"), "minilm"))
    Embedder(MINILM, tmp_path, endpoint=endpoint).embed_query("w w 東京 w")
    assert inputs(fake_llama) == [["w w"]]


def test_the_endpoint_backend_counts_with_the_tokenizer_file_only(
    tmp_path, fake_llama, tokenizer_file, monkeypatch
):
    real = embedder_mod.TextEmbedding

    class Boom:
        list_supported_models = staticmethod(real.list_supported_models)

        def __init__(self, *args, **kwargs):
            raise AssertionError("the ONNX model must not be built to count tokens")

    monkeypatch.setattr(embedder_mod, "TextEmbedding", Boom)
    emb = e5(tmp_path, fake_llama)
    assert emb.count_tokens("a b c") == 5  # 3 + 2 specials, prefix not counted
    assert emb.count_tokens(" ".join(["w"] * 2000)) == 2002  # not truncated at the window
    assert [(c["repo_id"], c["filename"]) for c in tokenizer_file.calls] == [
        ("qdrant/multilingual-e5-large-onnx", "tokenizer.json")
    ]
    assert tokenizer_file.calls[0]["cache_dir"] == str(tmp_path)
    assert tokenizer_file.calls[0]["local_files_only"] is True  # a warm cache asks no one
    assert fake_llama.requests == []


def test_the_tokenizer_file_is_downloaded_when_it_is_not_cached(
    tmp_path, fake_llama, tokenizer_file, monkeypatch
):
    def cold_cache(**kwargs):
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("not in the cache")
        return tokenizer_file.download(**kwargs)

    monkeypatch.setattr(embedder_mod, "hf_hub_download", cold_cache)
    assert e5(tmp_path, fake_llama).count_tokens("a b c") == 5
    assert [c.get("local_files_only", False) for c in tokenizer_file.calls] == [False]


def test_prefix_tokens_is_what_the_prefix_adds(tmp_path, fake_llama, tokenizer_file):
    assert e5(tmp_path, fake_llama).prefix_tokens == 2  # "passage", ":"
    assert Embedder(MINILM, tmp_path).prefix_tokens == 0
    assert e5(tmp_path, fake_llama, prefixes="none").prefix_tokens == 0


def test_a_missing_tokenizer_file_is_loud(tmp_path, fake_llama, monkeypatch):
    def offline(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr(embedder_mod, "hf_hub_download", offline)
    emb = e5(tmp_path, fake_llama)
    with pytest.raises(EmbedderUnavailableError) as raised:
        emb.count_tokens("a b c")
    message = str(raised.value)
    assert "qdrant/multilingual-e5-large-onnx" in message and str(tmp_path) in message
    assert "needed to size chunks" in message and "offline" in message
    with pytest.raises(EmbedderUnavailableError):
        emb.embed_query("a")
    assert fake_llama.requests == []


def test_max_tokens_is_the_models_window(tmp_path):
    assert Embedder(E5_LARGE, tmp_path).max_tokens == 512
    assert Embedder(MINILM, tmp_path).max_tokens == 128


def test_known_model_reads_the_registry(tmp_path):
    assert known_model(E5_LARGE) and known_model(MINILM)
    assert not known_model("x/unknown")
    assert not any(tmp_path.iterdir())


def test_an_endpoint_embedder_survives_pickling(tmp_path, fake_llama, tokenizer_file):
    emb = e5(tmp_path, fake_llama)
    before = emb.embed_documents(["alpha"])  # session and counter now exist
    clone = pickle.loads(pickle.dumps(emb))
    assert clone.embed_documents(["alpha"]) == before
    assert clone.prefixes == "e5" and clone.backend == "endpoint"
    assert clone.endpoint.endpoint == emb.endpoint.endpoint


def test_an_in_process_embedder_survives_pickling_without_its_model(tmp_path, stub_text_embedding):
    emb = Embedder(E5_LARGE, tmp_path, threads=2)
    before = emb.embed_documents(["alpha"])
    clone = pickle.loads(pickle.dumps(emb))
    assert clone.embed_documents(["alpha"]) == before
    assert stub_text_embedding.threads == [2, 2]  # rebuilt, with the same threads


def test_pickling_keeps_the_configuration_and_drops_the_fallback(
    tmp_path, dead_url, stub_text_embedding, tokenizer_file
):
    emb = dead(tmp_path, dead_url, model_name=MINILM, prefixes="e5", threads=3)
    with pytest.warns(RuntimeWarning):
        emb.embed_query("x")
    clone = pickle.loads(pickle.dumps(emb))
    assert (clone.model_name, clone.cache_dir, clone.prefixes, clone.threads) == (
        MINILM,
        tmp_path,
        "e5",
        3,
    )
    assert clone.allow_fallback and clone.backend == "endpoint" and clone.warning is None
    strict = pickle.loads(pickle.dumps(dead(tmp_path, dead_url, allow_fallback=False)))
    assert strict.allow_fallback is False
