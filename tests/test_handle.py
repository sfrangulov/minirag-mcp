"""One place opens an index: `IndexHandle`, and `make_embedder` behind it."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import minirag_mcp.handle as handle_mod
from conftest import FakeEmbedder
from minirag_mcp.config import load_config
from minirag_mcp.embedder import E5_LARGE, MINILM, Embedder, UnknownModelError
from minirag_mcp.endpoints import EmbeddingEndpoint, Endpoint
from minirag_mcp.handle import IndexHandle, configured_scheme, make_embedder
from minirag_mcp.identity import EmbeddingScheme
from minirag_mcp.store import IndexOpenError
from minirag_mcp.sync import SyncManager

UNUSED_URL = "http://127.0.0.1:9/v1/embeddings"


def cfg(tmp_path, **env):
    return load_config({"BASE_DIR": str(tmp_path), **env}, cwd=tmp_path)


def endpoint_cfg(tmp_path, **env):
    return replace(cfg(tmp_path, **env), embed_endpoints={E5_LARGE: Endpoint(UNUSED_URL, "e5")})


@pytest.fixture
def closed(monkeypatch) -> list[EmbeddingEndpoint]:
    """Every endpoint client closed from here on."""
    seen: list[EmbeddingEndpoint] = []
    monkeypatch.setattr(EmbeddingEndpoint, "close", lambda self: seen.append(self))
    return seen


@pytest.fixture
def built(monkeypatch) -> list[Embedder]:
    """Every embedder the handle's own factory builds."""
    made: list[Embedder] = []
    real = handle_mod.make_embedder

    def recording(config, scheme, **kwargs):
        made.append(real(config, scheme, **kwargs))
        return made[-1]

    monkeypatch.setattr(handle_mod, "make_embedder", recording)
    return made


def test_get_returns_one_open_index(tmp_path, fake_embedder):
    h = IndexHandle(
        load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path),
        embedder_factory=lambda s: fake_embedder,
    )
    a, b = h.get(), h.get()
    assert a is b and a.pipeline.store is a.store and a.embedder is fake_embedder


def test_a_failed_open_is_retried_by_the_next_call(tmp_path, fake_embedder):
    (tmp_path / "db").write_text("not a folder")
    h = IndexHandle(
        load_config({"BASE_DIR": str(tmp_path), "DB_PATH": str(tmp_path / "db")}, cwd=tmp_path),
        embedder_factory=lambda s: fake_embedder,
    )
    with pytest.raises(IndexOpenError):
        h.get()
    (tmp_path / "db").unlink()
    assert h.get().store.chunk_count() == 0


def test_make_embedder_takes_each_models_own_endpoint(tmp_path):
    ep = Endpoint(UNUSED_URL, "e5")
    cfg = replace(load_config({}, cwd=tmp_path), embed_endpoints={E5_LARGE: ep})
    assert configured_scheme(cfg) == EmbeddingScheme(E5_LARGE, 1024, "e5")
    assert make_embedder(cfg, configured_scheme(cfg)).backend == "endpoint"
    assert make_embedder(cfg, EmbeddingScheme(MINILM, 384, "none")).backend == "in-process"
    both = replace(cfg, embed_endpoints={E5_LARGE: ep, MINILM: ep})
    assert make_embedder(both, EmbeddingScheme(MINILM, 384, "none")).backend == "endpoint"


def test_make_embedder_takes_the_schemes_prefixes_and_the_threads(tmp_path):
    """A recorded scheme's prefixes win over what the model name would suggest."""
    config = load_config({}, cwd=tmp_path)
    e = make_embedder(config, EmbeddingScheme(E5_LARGE, 1024, "none"), threads=2)
    assert (e.model_name, e.prefixes, e.threads, e.cache_dir) == (
        E5_LARGE,
        "none",
        2,
        config.cache_dir,
    )


def test_configured_scheme_names_an_unknown_model(tmp_path):
    config = load_config({"MODEL_NAME": "nobody/no-such-model"}, cwd=tmp_path)
    with pytest.raises(UnknownModelError, match="nobody/no-such-model"):
        configured_scheme(config)


def test_the_embedder_is_built_for_the_configured_scheme(tmp_path, fake_embedder):
    seen: list[EmbeddingScheme] = []

    def factory(scheme):
        seen.append(scheme)
        return fake_embedder

    IndexHandle(cfg(tmp_path, MODEL_NAME=MINILM), embedder_factory=factory).get()
    assert seen == [EmbeddingScheme(MINILM, 384, "none")]


def test_without_a_factory_the_handle_builds_through_make_embedder(tmp_path):
    e = IndexHandle(endpoint_cfg(tmp_path)).get().embedder
    assert isinstance(e, Embedder)
    assert (e.model_name, e.prefixes, e.backend) == (E5_LARGE, "e5", "endpoint")


def test_concurrent_first_calls_open_the_index_once(tmp_path, fake_embedder):
    calls: list[EmbeddingScheme] = []

    def slow(scheme):
        calls.append(scheme)
        time.sleep(0.2)
        return fake_embedder

    h = IndexHandle(cfg(tmp_path), embedder_factory=slow)
    with ThreadPoolExecutor(2) as pool:
        a, b = pool.map(lambda _: h.get(), range(2))
    assert a is b and len(calls) == 1


def _finished(manager: SyncManager, job_id: str) -> str:
    deadline = time.monotonic() + 30
    while (state := manager.status(job_id).state) not in ("succeeded", "failed"):
        assert time.monotonic() < deadline, "sync did not finish"
        time.sleep(0.01)
    return state


def test_the_sync_manager_comes_with_the_first_successful_open(tmp_path, fake_embedder):
    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha body about tokens.")
    db = tmp_path / "db"
    db.write_text("not a folder")
    h = IndexHandle(cfg(tmp_path, DB_PATH=str(db)), embedder_factory=lambda s: fake_embedder)
    with pytest.raises(IndexOpenError):
        h.sync  # noqa: B018 — the property opens the index
    db.unlink()
    manager = h.sync
    assert isinstance(manager, SyncManager) and h.sync is manager
    assert _finished(manager, manager.start(None)) == "succeeded"
    assert h.get().store.source_count() == 1


def test_notes_are_the_open_embedders_warnings(tmp_path):
    emb = FakeEmbedder()
    h = IndexHandle(cfg(tmp_path), embedder_factory=lambda s: emb)
    h.get()
    assert h.notes() == []
    emb.warning = "W"
    assert h.notes() == ["W"]


def test_the_ingest_pipeline_writes_to_the_open_index(tmp_path, fake_embedder):
    h = IndexHandle(cfg(tmp_path), embedder_factory=lambda s: fake_embedder)
    h.ingest_pipeline().ingest_data("Alpha body.", source="s")
    assert h.ingest_pipeline() is h.get().pipeline
    assert h.get().store.source_count() == 1


def test_a_failed_open_closes_the_endpoint_client_it_built(tmp_path, built, closed):
    (tmp_path / "db").write_text("not a folder")
    h = IndexHandle(endpoint_cfg(tmp_path, DB_PATH=str(tmp_path / "db")))
    with pytest.raises(IndexOpenError):
        h.get()
    assert len(built) == 1 and closed == [built[0].endpoint]


def test_close_closes_the_endpoint_client_it_built_and_the_next_get_reopens(
    tmp_path, built, closed
):
    h = IndexHandle(endpoint_cfg(tmp_path))
    first = h.get()
    manager = h.sync
    h.close()
    assert closed == [first.embedder.endpoint]
    second = h.get()
    assert second is not first and len(built) == 2
    assert h.sync is not manager


def test_embedders_from_an_injected_factory_are_left_open(tmp_path, closed):
    """They belong to whoever supplied the factory, and may serve other indexes too."""
    shared = make_embedder(endpoint_cfg(tmp_path), EmbeddingScheme(E5_LARGE, 1024, "e5"))
    db = tmp_path / "db"
    db.write_text("not a folder")
    h = IndexHandle(endpoint_cfg(tmp_path, DB_PATH=str(db)), embedder_factory=lambda s: shared)
    with pytest.raises(IndexOpenError):
        h.get()
    db.unlink()
    h.get()
    h.close()
    assert closed == []


def test_classify_url_selects_the_served_model(tmp_path, fake_embedder):
    from minirag_mcp.results import labels_status

    config = replace(cfg(tmp_path), classify_endpoint=Endpoint("http://127.0.0.1:9", "plumb"))
    opened = IndexHandle(config, embedder_factory=lambda scheme: fake_embedder).get()
    assert opened.labels.backend == "endpoint"
    opened.labels.path.parent.mkdir(parents=True, exist_ok=True)
    opened.labels.path.write_text(
        '[facets.project]\ndefault = "core"\nvalues.core = { description = "Core." }\n',
        encoding="utf-8",
    )
    assert labels_status(opened.labels, opened.store)["labels"]["backend"] == "endpoint"

    plain = IndexHandle(cfg(tmp_path), embedder_factory=lambda scheme: fake_embedder)
    assert plain.get().labels.backend == "in-process"
