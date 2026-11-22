"""One place opens an index: `IndexHandle`, and `make_embedder` behind it."""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

import minirag_mcp.handle as handle_mod
from conftest import FakeEmbedder
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import default_db_path, load_config
from minirag_mcp.embedder import (
    E5_LARGE,
    MINILM,
    Embedder,
    ProbeMismatchError,
    UnknownModelError,
)
from minirag_mcp.endpoints import EmbeddingEndpoint, Endpoint
from minirag_mcp.handle import (
    IndexHandle,
    IndexUnavailableError,
    RebuildPendingError,
    configured_scheme,
    make_embedder,
)
from minirag_mcp.identity import (
    INDEX_FILE,
    PROBE_TEXT,
    ChunkingScheme,
    EmbeddingScheme,
    IndexIdentity,
    IndexState,
    begin_staging,
    previous_dir,
    read_identity,
    staging_dir,
    write_identity,
)
from minirag_mcp.lock import sync_lock
from minirag_mcp.results import run_query
from minirag_mcp.store import ChunkRecord, IndexBusyError, IndexOpenError, Store
from minirag_mcp.sync import SyncManager
from minirag_mcp.sync_report import REPORT_FILE, read_report

UNUSED_URL = "http://127.0.0.1:9/v1/embeddings"


def cfg(tmp_path, **env):
    return load_config({"BASE_DIR": str(tmp_path), **env}, cwd=tmp_path)


def endpoint_cfg(tmp_path, **env):
    return replace(cfg(tmp_path, **env), embed_endpoints={E5_LARGE: Endpoint(UNUSED_URL, "e5")})


def handle(tmp_path, embedder, **env) -> IndexHandle:
    return IndexHandle(cfg(tmp_path, **env), embedder_factory=lambda s: embedder)


def rec(source, i, text):
    return ChunkRecord(
        id=f"{source}#{i}",
        source=source,
        source_type="file",
        title="T",
        chunk_index=i,
        text=text,
        vector=[0.1] * 8,
        file_hash="h",
        mtime=1.0,
        ingested_at="2026-11-19T00:00:00+00:00",
    )


def recorded(config, scheme: EmbeddingScheme) -> IndexIdentity:
    """`config`'s index stamped as built with `scheme`."""
    identity = IndexIdentity.fresh(
        scheme, ChunkingScheme(SCHEME_VERSION, config.token_budget), config.fts_language
    )
    write_identity(config.db_path, identity)
    return identity


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


def test_a_fresh_folder_is_stamped_with_the_configured_scheme(tmp_path, fake_embedder):
    h = handle(tmp_path, fake_embedder)
    h.get()
    ident = read_identity(h.config.db_path)
    assert (ident.generation, ident.embedding, ident.probe) == (
        1,
        EmbeddingScheme("fake", 8, "none"),
        (),
    )
    assert ident.chunking == ChunkingScheme(SCHEME_VERSION, h.config.token_budget)
    assert ident.fts_language == h.config.fts_language and ident.document_table is True
    assert h.get().identity == ident
    assert h.state() == (IndexState.READY, "")


def test_a_fresh_folder_is_stamped_before_its_table_is_created(
    tmp_path, fake_embedder, monkeypatch
):
    """A crash between the two must not leave a folder that reads as a 1.x index."""
    stamped_first: list[bool] = []
    real = handle_mod.Store

    def spy(db_path, **kwargs):
        stamped_first.append(read_identity(db_path) is not None)
        return real(db_path, **kwargs)

    monkeypatch.setattr(handle_mod, "Store", spy)
    handle(tmp_path, fake_embedder).get()
    assert stamped_first == [True]


def test_a_1x_index_is_refused_and_left_untouched(tmp_path, fake_embedder):
    db = tmp_path / "db"
    Store(db, dim=8).replace_source("/a.md", [rec("/a.md", 0, "x")])
    before = sorted(p.relative_to(db) for p in db.rglob("*"))
    h = handle(tmp_path, fake_embedder, DB_PATH=str(db))
    with pytest.raises(
        IndexUnavailableError, match=re.escape(str(default_db_path(h.config.roots)))
    ):
        h.get()
    state, why = h.state()
    assert state is IndexState.UNAVAILABLE and "1.x" in why
    assert sorted(p.relative_to(db) for p in db.rglob("*")) == before


def test_an_unreadable_identity_is_unavailable_and_never_overwritten(tmp_path, fake_embedder):
    h = handle(tmp_path, fake_embedder)
    db = h.config.db_path
    db.mkdir(parents=True)
    (db / INDEX_FILE).write_text("{", encoding="utf-8")
    with pytest.raises(IndexUnavailableError, match=re.escape(str(db / INDEX_FILE))):
        h.get()
    state, why = h.state()
    assert state is IndexState.UNAVAILABLE and str(db / INDEX_FILE) in why
    assert (db / INDEX_FILE).read_text(encoding="utf-8") == "{"
    assert sorted(p.name for p in db.iterdir()) == [INDEX_FILE]


def test_an_interrupted_swap_is_finished_on_open(tmp_path, fake_embedder):
    """Opening the missing folder as it is would answer "nothing indexed"."""
    db = cfg(tmp_path).db_path
    handle(tmp_path, fake_embedder, DB_PATH=str(previous_dir(db))).ingest_pipeline().ingest_data(
        "Old text.", source="old"
    )
    staged = handle(tmp_path, fake_embedder, DB_PATH=str(staging_dir(db)))
    staged.ingest_pipeline().ingest_data("Rebuilt text.", source="rebuilt")
    write_identity(staging_dir(db), replace(read_identity(staging_dir(db)), generation=2))
    assert not db.exists()

    opened = handle(tmp_path, fake_embedder).get()
    assert opened.identity.generation == 2
    assert [s.source for s in opened.store.list_sources()] == ["rebuilt"]
    assert not staging_dir(db).exists()


def _recording_queries(models: dict) -> list[str]:
    seen: list[str] = []
    for name, model in models.items():
        real = model.embed_query
        model.embed_query = lambda text, real=real, name=name: seen.append(name) or real(text)
    return seen


def test_a_changed_model_is_pending_and_queries_use_the_recorded_model(tmp_path):
    models = {MINILM: FakeEmbedder(MINILM, salt="a"), E5_LARGE: FakeEmbedder(E5_LARGE, salt="b")}

    def factory(scheme):
        return models[scheme.model]

    a = IndexHandle(cfg(tmp_path, MODEL_NAME=MINILM), embedder_factory=factory)
    a.get().pipeline.ingest_data("budget review", source="s")
    b = IndexHandle(cfg(tmp_path), embedder_factory=factory)  # the default model, e5
    state, why = b.state()
    assert state is IndexState.REBUILD_PENDING and f"model {MINILM} -> {E5_LARGE}" in why
    opened = b.get()
    assert opened.embedder is models[MINILM]
    assert opened.identity.embedding == EmbeddingScheme(MINILM, 8, "none")
    seen = _recording_queries(models)
    assert run_query(opened.store, opened.embedder, b.config, "budget review")["sources"]
    assert seen == [MINILM]


def test_a_pending_index_embeds_through_the_recorded_models_endpoint(
    tmp_path, fake_llama, tokenizer_file
):
    fake_llama.dim = 1024
    config = replace(
        cfg(tmp_path, MODEL_NAME=MINILM),
        embed_endpoints={E5_LARGE: Endpoint(fake_llama.url("/v1/embeddings"), "e5")},
    )
    recorded(config, EmbeddingScheme(E5_LARGE, 1024, "e5"))
    h = IndexHandle(config)
    state, why = h.state()
    assert state is IndexState.REBUILD_PENDING
    assert why == f"model {E5_LARGE} -> {MINILM}; query/passage prefixes e5 -> none"
    opened = h.get()
    assert (opened.embedder.model_name, opened.embedder.backend) == (E5_LARGE, "endpoint")
    run_query(opened.store, opened.embedder, config, "budget review")
    sent = [r.body["input"] for r in fake_llama.requests if r.path == "/v1/embeddings"]
    assert sent[-1] == ["query: budget review"]


def test_a_changed_budget_is_pending_with_a_chunking_reason(tmp_path, fake_embedder):
    a = handle(tmp_path, fake_embedder)
    a.ingest_pipeline().ingest_data("Alpha body.", source="s")
    bigger = a.config.token_budget + 90
    b = handle(tmp_path, fake_embedder, CHUNK_TOKEN_BUDGET=str(bigger))
    assert b.state() == (
        IndexState.REBUILD_PENDING,
        f"chunk budget {a.config.token_budget} -> {bigger} tokens",
    )
    assert b.get().identity.chunking.token_budget == a.config.token_budget


def test_a_pending_index_refuses_ingests_and_still_deletes(tmp_path, fake_embedder):
    a = handle(tmp_path, fake_embedder)
    a.ingest_pipeline().ingest_data("Alpha body.", source="s")
    bigger = a.config.token_budget + 90
    b = handle(tmp_path, fake_embedder, CHUNK_TOKEN_BUDGET=str(bigger))
    with pytest.raises(RebuildPendingError) as refused:
        b.ingest_pipeline()
    assert "index needs a rebuild: run an unscoped sync" in str(refused.value)
    assert f"chunk budget {a.config.token_budget} -> {bigger} tokens" in str(refused.value)
    assert b.get().store.delete_source("s") > 0
    assert a.ingest_pipeline() is a.get().pipeline


def test_a_changed_fts_language_is_rebuilt_in_place_and_not_pending(tmp_path, fake_embedder):
    a = handle(tmp_path, fake_embedder)
    a.ingest_pipeline().ingest_data("Подписание договоров с поставщиками.", source="s")
    before = read_identity(a.config.db_path)
    assert before.fts_language == "English"
    b = handle(tmp_path, fake_embedder, RAG_FTS_LANGUAGE="Russian")
    assert b.state() == (IndexState.READY, "")
    opened = b.get()
    assert opened.store.fts_index_language == "Russian"
    assert opened.identity == replace(before, fts_language="Russian")
    assert read_identity(b.config.db_path) == opened.identity


def test_an_fts_language_that_cannot_be_recorded_warns_and_still_opens(
    tmp_path, fake_embedder, monkeypatch
):
    a = handle(tmp_path, fake_embedder)
    a.get()

    def refused(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(handle_mod, "update_identity", refused)
    b = handle(tmp_path, fake_embedder, RAG_FTS_LANGUAGE="Russian")
    with pytest.warns(RuntimeWarning, match="Permission denied"):
        opened = b.get()
    assert opened.store.fts_index_language == "Russian"
    assert read_identity(b.config.db_path).fts_language == "English"


def test_a_recorded_model_that_cannot_be_built_is_unavailable(tmp_path):
    config = cfg(tmp_path)
    recorded(config, EmbeddingScheme("x/unknown", 8, "none"))
    before = (config.db_path / INDEX_FILE).read_bytes()
    h = IndexHandle(config)
    with pytest.raises(IndexUnavailableError, match="x/unknown"):
        h.get()
    state, why = h.state()
    assert state is IndexState.UNAVAILABLE and "x/unknown" in why
    assert (config.db_path / INDEX_FILE).read_bytes() == before
    assert not (config.db_path / "chunks.lance").exists()


def test_a_recorded_model_of_another_width_here_is_unavailable(tmp_path):
    config = cfg(tmp_path)
    recorded(config, EmbeddingScheme(MINILM, 16, "none"))
    h = IndexHandle(config, embedder_factory=lambda s: FakeEmbedder(s.model))
    with pytest.raises(IndexUnavailableError) as refused:
        h.get()
    assert MINILM in str(refused.value) and "16" in str(refused.value)
    assert "8-dimensional" in str(refused.value)
    assert h.state()[0] is IndexState.UNAVAILABLE


class _Blocking(FakeEmbedder):
    """Holds every document embedding until released."""

    def __init__(self) -> None:
        super().__init__()
        self.entered, self.released = threading.Event(), threading.Event()

    def embed_documents(self, texts):
        self.entered.set()
        assert self.released.wait(30)
        return super().embed_documents(texts)


def test_a_running_sync_makes_the_index_syncing(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha body about tokens.")
    emb = _Blocking()
    h = handle(tmp_path, emb)
    assert h.sync.running is False
    job = h.sync.start(None)
    assert emb.entered.wait(30)
    assert h.sync.running is True
    assert h.state() == (IndexState.SYNCING, "")
    emb.released.set()
    assert _finished(h.sync, job) == "succeeded"
    assert h.sync.running is False
    assert h.state() == (IndexState.READY, "")


def test_sync_running_says_whether_a_job_runs_and_opens_nothing(tmp_path):
    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha body about tokens.")
    emb = _Blocking()
    made = []
    h = IndexHandle(cfg(tmp_path), embedder_factory=lambda s: made.append(s) or emb)
    assert h.sync_running is False
    assert made == [] and not (tmp_path / ".minirag").exists()
    job = h.start_sync()
    assert emb.entered.wait(30)
    assert h.sync_running is True
    emb.released.set()
    assert _finished(h.sync, job) == "succeeded"
    assert h.sync_running is False


def test_state_never_raises(tmp_path, fake_embedder):
    db = tmp_path / "db"
    db.write_text("not a folder")
    h = handle(tmp_path, fake_embedder, DB_PATH=str(db))
    state, why = h.state()
    assert state is IndexState.UNAVAILABLE and "DB_PATH" in why

    def unknown(scheme):
        raise UnknownModelError("Model 'nobody/x' is not in the fastembed registry.")

    h = IndexHandle(cfg(tmp_path), embedder_factory=unknown)
    assert h.state() == (
        IndexState.UNAVAILABLE,
        "Model 'nobody/x' is not in the fastembed registry.",
    )


def test_a_failed_pipeline_build_closes_the_endpoint_client(tmp_path, built, closed, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no pipeline")

    monkeypatch.setattr(handle_mod, "Pipeline", broken)
    h = IndexHandle(endpoint_cfg(tmp_path))
    with pytest.raises(RuntimeError, match="no pipeline"):
        h.get()
    assert len(built) == 1 and closed == [built[0].endpoint]


def test_close_closes_the_clients_of_both_embedders_of_a_pending_index(tmp_path, built, closed):
    ep = Endpoint(UNUSED_URL, "e5")
    config = replace(cfg(tmp_path, MODEL_NAME=MINILM), embed_endpoints={E5_LARGE: ep, MINILM: ep})
    recorded(config, EmbeddingScheme(E5_LARGE, 1024, "e5"))
    h = IndexHandle(config)
    assert h.get().embedder.model_name == E5_LARGE
    assert sorted(e.model_name for e in built) == sorted([E5_LARGE, MINILM])
    h.close()
    assert sorted(map(id, closed)) == sorted(id(e.endpoint) for e in built)


def test_status_of_an_open_index_gone_unavailable_reports_only_why(tmp_path, fake_embedder):
    from minirag_mcp.results import status_payload

    h = handle(tmp_path, fake_embedder)
    h.ingest_pipeline().ingest_data("Alpha body.", source="s")
    (h.config.db_path / INDEX_FILE).write_text("{", encoding="utf-8")
    st = status_payload(h.config, h)
    assert st["indexState"] == "unavailable"
    assert str(h.config.db_path / INDEX_FILE) in st["indexError"]
    assert not {"chunkCount", "indexGeneration", "embedding"} & set(st)
    with pytest.raises(IndexUnavailableError, match=re.escape(str(h.config.db_path / INDEX_FILE))):
        h.ingest_pipeline()


LONG = " ".join(
    f"Sentence {i} " + " ".join(f"w{i}x{j}" for j in range(10)) + "." for i in range(40)
)


def test_an_index_is_chunked_at_its_recorded_budget(tmp_path, fake_embedder):
    """A pending index's pipeline must not add chunks cut at the configured budget."""
    small = 32
    a = handle(tmp_path, fake_embedder, CHUNK_TOKEN_BUDGET=str(small))
    a.ingest_pipeline().ingest_data(LONG, source="recorded")
    b = handle(tmp_path, fake_embedder)
    assert b.config.token_budget > small
    assert b.state()[0] is IndexState.REBUILD_PENDING
    b.get().pipeline.ingest_data(LONG, source="pending")
    other = handle(tmp_path, fake_embedder, DB_PATH=str(tmp_path / "other"))
    other.ingest_pipeline().ingest_data(LONG, source="configured")

    def chunks(h, source):
        return len(h.get().store.all_chunks(source))

    assert chunks(b, "pending") == chunks(b, "recorded")
    assert chunks(b, "pending") != chunks(other, "configured")


@pytest.mark.parametrize("model, dim", [("x/unknown", 8), (MINILM, 16)])
def test_a_recorded_embedder_refused_at_open_has_its_client_closed(
    tmp_path, built, closed, model, dim
):
    ep = Endpoint(UNUSED_URL, "e5")
    config = replace(cfg(tmp_path), embed_endpoints={E5_LARGE: ep, model: ep})
    recorded(config, EmbeddingScheme(model, dim, "none"))
    with pytest.raises(IndexUnavailableError, match=re.escape(model)):
        IndexHandle(config).get()
    assert [e.model_name for e in built] == [E5_LARGE, model]
    assert sorted(map(id, closed)) == sorted(id(e.endpoint) for e in built)


def _served(tmp_path, fake_llama) -> IndexHandle:
    fake_llama.dim = 1024
    config = replace(
        cfg(tmp_path),
        embed_endpoints={E5_LARGE: Endpoint(fake_llama.url("/v1/embeddings"), "e5")},
    )
    return IndexHandle(config)


def test_the_first_ingest_records_the_probe(tmp_path, fake_embedder):
    h = handle(tmp_path, fake_embedder)
    h.get().pipeline.ingest_data("x", source="s")
    assert read_identity(h.config.db_path).probe == pytest.approx(fake_embedder._vec(PROBE_TEXT))


def test_a_matching_endpoint_is_ready(fake_llama, tmp_path, tokenizer_file):
    first = _served(tmp_path, fake_llama)
    first.get().pipeline.ingest_data("x", source="s")
    first.close()
    again = _served(tmp_path, fake_llama)
    assert again.state() == (IndexState.READY, "")
    again.get().pipeline.ingest_data("y", source="t")


def test_an_endpoint_serving_another_model_is_unavailable_naming_both(
    fake_llama, tmp_path, tokenizer_file
):
    first = _served(tmp_path, fake_llama)
    first.get().pipeline.ingest_data("x", source="s")
    first.close()
    fake_llama.vector_salt = "other"
    h = _served(tmp_path, fake_llama)
    state, why = h.state()
    assert state is IndexState.UNAVAILABLE and E5_LARGE in why and "'e5'" in why
    with pytest.raises(IndexUnavailableError):
        h.get()


def test_an_open_index_turns_unavailable_when_the_endpoint_changes_model(
    fake_llama, tmp_path, tokenizer_file
):
    import minirag_mcp.embedder as embedder_mod

    h = _served(tmp_path, fake_llama)
    pipeline = h.get().pipeline
    pipeline.ingest_data("x", source="s")
    fake_llama.vector_salt = "other"
    with pytest.raises(embedder_mod.ProbeMismatchError):
        pipeline.ingest_data("y", source="t")
    assert h.state()[0] is IndexState.UNAVAILABLE
    with pytest.raises(IndexUnavailableError):
        h.get()
    fake_llama.vector_salt = ""
    assert h.state() == (IndexState.READY, "")


def test_the_store_hook_is_bound_to_the_generation_it_was_opened_with(tmp_path, fake_embedder):
    """Re-resolved through the handle, a pipeline taken before a swap would pass its check
    against the new generation and write chunks made the old way into it."""
    h = handle(tmp_path, fake_embedder)
    opened = h.get()
    hook = opened.pipeline.before_store
    db = h.config.db_path
    write_identity(db, replace(read_identity(db), generation=2))
    reopened = h.get()
    assert reopened is not opened and reopened.pipeline.before_store is not hook
    with pytest.raises(IndexBusyError, match="replaced by a rebuilt index"):
        hook()
    reopened.pipeline.before_store()
    h.ensure_probe()
    assert read_identity(db).probe == pytest.approx(fake_embedder._vec(PROBE_TEXT))


def test_a_same_width_wrong_model_is_refused_on_a_fresh_folder(
    fake_llama, tmp_path, tokenizer_file
):
    fake_llama.vector_salt = "bge-m3"
    h = _served(tmp_path, fake_llama)
    with pytest.raises(IndexUnavailableError) as refused:
        h.get().pipeline.ingest_data("x", source="s")
    why = str(refused.value)
    assert E5_LARGE in why and "'e5'" in why and fake_llama.url("/v1/embeddings") in why
    assert read_identity(h.config.db_path).probe == ()
    assert h.get().store.get_source("s") is None
    fake_llama.vector_salt = ""
    h.get().pipeline.ingest_data("x", source="s")
    assert read_identity(h.config.db_path).probe


def test_a_wrong_model_that_appears_after_open_is_refused_before_the_probe_is_recorded(
    fake_llama, tmp_path, tokenizer_file
):
    h = _served(tmp_path, fake_llama)
    pipeline = h.get().pipeline
    fake_llama.vector_salt = "bge-m3"
    with pytest.raises(IndexUnavailableError, match=E5_LARGE):
        pipeline.ingest_data("x", source="s")
    assert read_identity(h.config.db_path).probe == ()
    assert h.get().store.get_source("s") is None


def test_a_model_without_a_reference_warns_once_and_is_recorded(
    fake_llama, tmp_path, tokenizer_file, reference_probes
):
    reference_probes.clear()
    h = _served(tmp_path, fake_llama)
    with pytest.warns(RuntimeWarning, match="cannot be verified"):
        h.get().pipeline.ingest_data("x", source="s")
    assert read_identity(h.config.db_path).probe


def test_the_rebuilt_generation_embeds_with_the_rebuilds_embedder(
    tmp_path, fake_llama, tokenizer_file, built, closed, reference_probes
):
    """The one the handle built for the configuration, not another copy of that model; the
    one it replaces has its client closed."""
    fake_llama.dim = 1024
    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha body about budgets.")
    config = replace(
        cfg(tmp_path), embed_endpoints={E5_LARGE: Endpoint(fake_llama.url("/v1/embeddings"), "e5")}
    )
    unprefixed = EmbeddingScheme(E5_LARGE, 1024, "none")
    shipped = dict(reference_probes)
    reference_probes.clear()
    with pytest.warns(RuntimeWarning, match="cannot be verified"):
        IndexHandle(config, embedder_factory=lambda s: make_embedder(config, unprefixed)).run_sync(
            None
        )
    reference_probes.update(shipped)
    built.clear()
    closed.clear()
    h = IndexHandle(config)
    replaced = h.get().embedder

    _, _, info = h.run_sync(None)

    assert info.kind == "reembed" and len(built) == 2 and built[1] is replaced
    assert h.get().embedder is built[0] and h.get().identity.generation == 2
    assert closed == [replaced.endpoint]


def test_the_hook_refuses_a_write_when_the_identity_it_reads_is_another_generation(
    tmp_path, fake_embedder, monkeypatch
):
    """Even when its quick look misses a swap: it never lets the write go on unchecked."""
    h = handle(tmp_path, fake_embedder)
    opened = h.get()
    db = h.config.db_path
    monkeypatch.setattr(handle_mod._Watch, "replaced", lambda self: False)
    write_identity(db, replace(read_identity(db), generation=2))
    with pytest.raises(IndexBusyError, match="replaced by a rebuilt index"):
        opened.pipeline.before_store()
    with pytest.raises(IndexBusyError, match="replaced by a rebuilt index"):
        opened.store.delete_source("s")
    assert read_identity(db).probe == ()


def test_a_probe_another_process_records_is_checked_before_the_next_query(
    fake_llama, tmp_path, tokenizer_file
):
    """A server opened a fresh folder, so it held no probe to check; a terminal sync then
    recorded one. The server's next call checks its endpoint against that probe."""
    server = _served(tmp_path, fake_llama)
    server.get()
    _served(tmp_path, fake_llama).ingest_pipeline().ingest_data("Budgets.", source="b")
    fake_llama.vector_salt = "bge-m3"
    with pytest.raises(IndexUnavailableError, match=E5_LARGE):
        server.get()
    assert server.state()[0] is IndexState.UNAVAILABLE


def test_a_write_prepared_before_another_process_recorded_the_probe_is_checked(
    fake_llama, tmp_path, tokenizer_file
):
    """Its vectors were made before the embedder was armed with that probe."""
    server = _served(tmp_path, fake_llama)
    pipeline = server.ingest_pipeline()
    terminal = _served(tmp_path, fake_llama)
    terminal.ingest_pipeline().ingest_data("Budgets.", source="b")
    fake_llama.vector_salt = "bge-m3"
    with pytest.raises(IndexUnavailableError, match=E5_LARGE):
        pipeline.ingest_data("Written by another model.", source="a")
    assert terminal.get().store.get_source("a") is None


def test_two_first_writers_record_one_probe(tmp_path):
    """The second to measure arms itself with the probe the first recorded."""
    first, second = FakeEmbedder(), FakeEmbedder()
    a, b = handle(tmp_path, first), handle(tmp_path, second)
    pipeline = a.ingest_pipeline()
    b.get()
    recorded = first.probe_vector()

    def measured_while_the_other_records() -> list[float]:
        b.ingest_pipeline().ingest_data("Beta.", source="b")
        return [x + 1e-4 for x in recorded]  # the same model, rounded another way

    first.probe_vector = measured_while_the_other_records
    pipeline.ingest_data("Alpha.", source="a")
    assert read_identity(a.config.db_path).probe == pytest.approx(recorded)
    assert first.expected_probe == pytest.approx(recorded)


def test_every_ingest_checks_the_endpoint_before_it_starts(fake_llama, tmp_path, tokenizer_file):
    """A router restarted with its preset pointing at another model of the same width
    answers without a failure, within the interval an embedding would wait out."""
    h = _served(tmp_path, fake_llama)
    h.ingest_pipeline().ingest_data("Budgets.", source="s")
    fake_llama.vector_salt = "bge-m3"
    with pytest.raises(IndexUnavailableError, match=E5_LARGE):
        h.ingest_pipeline()


def test_every_sync_checks_the_endpoint_before_it_starts(fake_llama, tmp_path, tokenizer_file):
    h = _served(tmp_path, fake_llama)
    h.ingest_pipeline().ingest_data("Budgets.", source="s")
    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha body about budgets.")
    fake_llama.vector_salt = "bge-m3"
    with pytest.raises(IndexUnavailableError, match=E5_LARGE):
        h.run_sync(None)
    assert Store(h.config.db_path, dim=1024).get_source(str(tmp_path / "a.md")) is None


def test_an_index_is_available_again_once_its_endpoint_is_back(
    fake_llama, tmp_path, tokenizer_file, stub_text_embedding, reference_probes, monkeypatch
):
    """A mismatch of the in-process fallback, met while the endpoint was down, says nothing
    about the endpoint that serves again."""
    import minirag_mcp.embedder as embedder_mod

    reference_probes.clear()
    fake_llama.vector_salt = "gguf"  # recorded through the endpoint; fastembed differs
    h = _served(tmp_path, fake_llama)
    with pytest.warns(RuntimeWarning, match="cannot be verified"):
        h.ingest_pipeline().ingest_data("Budgets.", source="s")
    opened = h.get()
    fake_llama.fail["/v1/embeddings"] = (503, "Loading model")
    with pytest.warns(RuntimeWarning, match="503"), pytest.raises(ProbeMismatchError):
        run_query(opened.store, opened.embedder, h.config, "budgets")
    assert h.state()[0] is IndexState.UNAVAILABLE
    del fake_llama.fail["/v1/embeddings"]
    monkeypatch.setattr(embedder_mod, "ENDPOINT_RETRY_SECONDS", 0.0)
    assert h.state() == (IndexState.READY, "")
    run_query(h.get().store, h.get().embedder, h.config, "budgets")


def test_a_read_checks_a_slow_endpoint_for_the_query_timeout_only(
    fake_llama, tmp_path, tokenizer_file, stub_text_embedding, monkeypatch
):
    """A check that timed out is made again before the endpoint's next use, so a read
    neither waits the endpoint's own timeout nor sends embeddings in-process. A write
    still waits that timeout."""
    _served(tmp_path, fake_llama).ingest_pipeline().ingest_data("Budgets.", source="s")
    monkeypatch.setattr(handle_mod, "QUERY_TIMEOUT", 0.2)
    h = _served(tmp_path, fake_llama)
    fake_llama.delay = 1.0
    started = time.monotonic()
    embedder = h.get().embedder
    assert time.monotonic() - started < 0.8
    assert embedder.warning is None and embedder.backend == "endpoint"
    started = time.monotonic()
    h.ingest_pipeline()
    assert time.monotonic() - started >= 0.9
    assert embedder.warning is None


def test_a_known_mismatch_stays_known_when_a_read_cannot_check_in_time(
    fake_llama, tmp_path, tokenizer_file, stub_text_embedding, monkeypatch
):
    h = _served(tmp_path, fake_llama)
    h.ingest_pipeline().ingest_data("Budgets.", source="s")
    embedder = h.get().embedder
    fake_llama.vector_salt = "bge-m3"
    with pytest.raises(IndexUnavailableError, match=E5_LARGE) as checked:
        h.ingest_pipeline()
    assert "re-checked" not in str(checked.value)
    fake_llama.vector_salt = ""
    monkeypatch.setattr(handle_mod, "QUERY_TIMEOUT", 0.2)
    fake_llama.delay = 1.0
    started = time.monotonic()
    with pytest.raises(IndexUnavailableError, match=E5_LARGE) as unchecked:
        h.get()
    assert time.monotonic() - started < 0.8
    assert "could not be re-checked" in str(unchecked.value)
    assert "within 0.2 s" in str(unchecked.value)
    assert embedder.warning is None
    fake_llama.delay = 0.0
    assert h.get().embedder is embedder and embedder.probe_error is None


@pytest.mark.parametrize(
    "cached", ["nothing", "tokenizer only", "the model"], ids=lambda c: f"{c} cached"
)
def test_status_during_a_fallback_says_whether_the_model_can_run_in_process(
    fake_llama, dead_url, tmp_path, tokenizer_file, stub_text_embedding, cached
):
    """The index is opened, its endpoint found down at open and fallen back from, and no
    embedding asked for yet: nothing has tried to load the model."""
    from minirag_mcp.results import status_payload

    _served(tmp_path, fake_llama).ingest_pipeline().ingest_data("Budgets.", source="s")
    cache = tmp_path / "models"
    stub_text_embedding.on_disk = True
    if cached != "nothing":
        stub_text_embedding.put(cache, E5_LARGE, tokenizer_only=cached == "tokenizer only")
    config = replace(
        cfg(tmp_path, CACHE_DIR=str(cache)),
        embed_endpoints={E5_LARGE: Endpoint(dead_url + "/v1/embeddings", "e5")},
    )
    with pytest.warns(RuntimeWarning, match="unreachable"):
        st = status_payload(config, IndexHandle(config))
    assert st["embeddingBackend"] == "in-process (endpoint unreachable)"
    if cached == "the model":
        assert "embeddingFallback" not in st
    else:
        assert st["embeddingFallback"] == f"no local copy in {cache}"
    assert stub_text_embedding.model_names == [] and stub_text_embedding.downloads == []


def _sync_job(h: IndexHandle) -> None:
    job = h.start_sync()
    h.sync.wait()
    assert h.sync.status(job).error is None


@pytest.mark.parametrize(
    "write",
    [
        lambda h: h.ingest_pipeline(),
        lambda h: h.run_sync(None),
        _sync_job,
        # What the job's thread runs, after a mismatch may have become known.
        lambda h: h._run_held(None, None, None, None),
        lambda h: h.ensure_probe(),
    ],
    ids=["ingest", "sync", "sync-job", "sync-job-body", "before-store"],
)
def test_a_write_waits_out_a_slow_endpoint_to_clear_a_known_mismatch(
    fake_llama, tmp_path, tokenizer_file, stub_text_embedding, monkeypatch, write
):
    """A router fixed but slow to answer is checked within its own timeout before a write,
    not refused after QUERY_TIMEOUT for what an earlier check found."""
    h = _served(tmp_path, fake_llama)
    h.ingest_pipeline().ingest_data("Budgets.", source="s")
    embedder = h.get().embedder
    fake_llama.vector_salt = "bge-m3"
    with pytest.raises(IndexUnavailableError, match=E5_LARGE):
        h.ingest_pipeline()
    fake_llama.vector_salt = ""
    monkeypatch.setattr(handle_mod, "QUERY_TIMEOUT", 0.2)
    fake_llama.delay = 0.6
    write(h)
    assert embedder.probe_error is None and embedder.warning is None


def test_a_rebuild_by_another_process_makes_the_index_syncing(tmp_path, fake_embedder):
    h = handle(tmp_path, fake_embedder)
    h.get()
    db = h.config.db_path
    with sync_lock(db):
        begin_staging(db)
        assert h.state() == (IndexState.SYNCING, "")
    assert h.state() == (IndexState.READY, "")


def test_a_budget_only_rebuild_embeds_with_the_serving_embedder(
    tmp_path, fake_llama, tokenizer_file, built, closed
):
    """The embedding scheme is unchanged, so the model it serves with is the one it needs."""
    fake_llama.dim = 1024
    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha body about budgets.")
    ep = {E5_LARGE: Endpoint(fake_llama.url("/v1/embeddings"), "e5")}
    IndexHandle(replace(cfg(tmp_path, CHUNK_TOKEN_BUDGET="32"), embed_endpoints=ep)).run_sync(None)
    built.clear()
    closed.clear()
    h = IndexHandle(replace(cfg(tmp_path), embed_endpoints=ep))
    serving = h.get().embedder

    _, _, info = h.run_sync(None)

    assert info.kind == "reingest" and built == [serving] and closed == []
    assert h.get().embedder is serving and h.get().identity.generation == 2


def _synced_handle(tmp_path, fake_embedder, **env) -> IndexHandle:
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "a.md").write_text("# Alpha\n\nAlpha body text.\n")
    return handle(tmp_path, fake_embedder, **env)


def test_a_cli_sync_writes_the_report(tmp_path, fake_embedder):
    h = _synced_handle(tmp_path, fake_embedder)
    counts, _, _ = h.run_admitted(None, h.admit_sync(None))
    latest = read_report(h.config.db_path)["latest"]
    assert latest["ranBy"] == "cli" and latest["outcome"] == "succeeded"
    assert latest["counts"] == counts


def test_a_background_sync_writes_the_report(tmp_path, fake_embedder):
    h = _synced_handle(tmp_path, fake_embedder)
    _sync_job(h)
    report = read_report(h.config.db_path)
    assert report["latest"]["ranBy"] == "server"
    assert report["coverage"]["ranBy"] == "server"


def test_a_sync_that_raises_writes_a_failed_latest(tmp_path, fake_embedder, monkeypatch):
    h = _synced_handle(tmp_path, fake_embedder)

    def boom(*a, **k):
        raise RuntimeError("scan exploded")

    monkeypatch.setattr(handle_mod, "_run_sync_unlocked", boom)
    with pytest.raises(RuntimeError, match="scan exploded"):
        h.run_admitted(None, h.admit_sync(None))
    latest = read_report(h.config.db_path)["latest"]
    assert latest["outcome"] == "failed" and "scan exploded" in latest["error"]


def test_a_rebuild_report_lands_in_the_live_folder(tmp_path, fake_embedder):
    a = _synced_handle(tmp_path, fake_embedder)
    a.run_sync(None)
    bigger = a.config.token_budget + 90
    b = handle(tmp_path, fake_embedder, CHUNK_TOKEN_BUDGET=str(bigger))
    assert b.state()[0] is IndexState.REBUILD_PENDING
    b.run_sync(None)
    report = read_report(b.config.db_path)
    assert report["latest"]["rebuild"]["kind"]
    nxt = b.config.db_path.with_name(b.config.db_path.name + ".next")
    assert not (nxt / REPORT_FILE).exists()


def test_a_report_that_cannot_be_written_leaves_the_sync_result_unchanged(
    tmp_path, fake_embedder, monkeypatch
):
    h = _synced_handle(tmp_path, fake_embedder)
    expected = h.run_sync(None)
    (h.config.db_path / REPORT_FILE).unlink()

    def crash(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(os, "replace", crash)
    events: list[str] = []
    again = h.run_sync(None, events.append)
    assert again[0].keys() == expected[0].keys() and again[1] == expected[1]
    assert any("last-sync report" in e and "read-only" in e for e in events)
