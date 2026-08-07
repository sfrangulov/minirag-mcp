import math

import pytest

from minirag_mcp.config import DEFAULT_MODEL
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
