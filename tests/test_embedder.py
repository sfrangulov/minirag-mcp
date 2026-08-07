import pytest

from minirag_mcp.config import DEFAULT_MODEL
from minirag_mcp.embedder import Embedder, UnknownModelError


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
