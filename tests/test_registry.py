"""The index registry: which index a call reaches, and why a call is refused."""

from __future__ import annotations

import threading
import time

import pytest

from conftest import FakeEmbedder
from index_helpers import file_registry
from minirag_mcp.config import ConfigError, load_config
from minirag_mcp.embedder import E5_LARGE, MINILM, EmbedderUnavailableError
from minirag_mcp.handle import IndexHandle
from minirag_mcp.identity import PROBE_TEXT, IndexState, record_probe
from minirag_mcp.registry import (
    SINGLE_INDEX_NAME,
    UNRESTRICTED,
    Connection,
    IndexRefused,
    IndexRegistry,
    IndexUnavailable,
    Use,
)


def test_a_registry_opens_no_index_until_one_is_used(indexes_file, fake_embedder):
    made = []
    reg = file_registry(indexes_file, lambda c, s: made.append(c.index_name) or fake_embedder)
    assert reg.names == ("alpha", "beta", "gamma") and made == []
    reg.open(Connection("alpha", None), None, Use.READ)
    assert made == ["alpha"]
    assert not (indexes_file.roots["beta"] / ".minirag").exists()


def test_indexes_of_one_model_share_its_loaded_model(indexes_file, stub_text_embedding):
    indexes_file.write('[server]\ncache_dir = "~/models"\n\n' + indexes_file.text)
    reg = IndexRegistry.from_file(indexes_file.path, indexes_file.env)
    embedders = [
        reg.resolve(UNRESTRICTED, n).runtime().embedder for n in ("alpha", "beta", "gamma")
    ]
    for embedder in embedders:
        embedder.embed_query("apricots")
    assert stub_text_embedding.model_names == [E5_LARGE, MINILM]
    # Each index checks its own recorded probe, so none shares another's embedder.
    assert len({id(e) for e in embedders}) == 3


def test_concurrent_first_use_opens_an_index_once(indexes_file, fake_embedder):
    calls = []

    def slow(config, scheme):
        time.sleep(0.2)
        calls.append(config.index_name)
        return fake_embedder

    reg = file_registry(indexes_file, slow)
    start = threading.Barrier(8)
    runtimes = []

    def first_use() -> None:
        start.wait()
        runtimes.append(reg.resolve(UNRESTRICTED, "alpha").runtime())

    threads = [threading.Thread(target=first_use) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == ["alpha"] and len(runtimes) == 8 and len({id(r) for r in runtimes}) == 1


def test_opening_one_index_holds_up_no_other(indexes_file, fake_embedder):
    entered, release = threading.Event(), threading.Event()

    def factory(config, scheme):
        if config.index_name == "alpha":
            entered.set()
            release.wait(10)
        return fake_embedder

    reg = file_registry(indexes_file, factory)
    opening = threading.Thread(target=reg.open, args=(UNRESTRICTED, "alpha", Use.READ))
    opening.start()
    try:
        assert entered.wait(10)
        entry, _ = reg.open(UNRESTRICTED, "beta", Use.READ)
        assert entry.name == "beta" and opening.is_alive()
    finally:
        release.set()
        opening.join()


def test_a_connection_reaches_its_default_and_its_allow_list(indexes_file, fake_embedder):
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    assert reg.reachable(UNRESTRICTED) == ("alpha", "beta", "gamma")
    assert reg.reachable(Connection("gamma", frozenset({"alpha", "delta"}))) == ("alpha", "gamma")
    assert reg.reachable(Connection(None, frozenset())) == ()


@pytest.mark.parametrize(
    "conn, requested, message",
    [
        (
            Connection("alpha", None),
            "delta",
            "Unknown index 'delta'. Reachable indexes: alpha, beta, gamma.",
        ),
        (
            Connection("delta", None),
            None,
            "Unknown index 'delta' (this connection's default). Reachable indexes: alpha, beta, "
            "gamma.",
        ),
        (
            Connection("alpha", frozenset({"beta"})),
            "gamma",
            "Index 'gamma' is not allowed on this connection (allow=beta). Reachable indexes: "
            "alpha, beta.",
        ),
        (
            Connection("alpha", frozenset()),
            "beta",
            "Index 'beta' is not allowed on this connection (allow=). Reachable indexes: alpha.",
        ),
        (
            Connection(None, frozenset({"gamma", "beta"})),
            "alpha",
            "Index 'alpha' is not allowed on this connection (allow=beta,gamma). Reachable "
            "indexes: beta, gamma.",
        ),
        (
            Connection(None, frozenset()),
            "alpha",
            "Index 'alpha' is not allowed on this connection (allow=). Reachable indexes: none.",
        ),
        (
            UNRESTRICTED,
            None,
            "This connection has no default index, so name one with `index`. Reachable "
            "indexes: alpha, beta, gamma.",
        ),
    ],
)
def test_refusals_name_the_index_and_the_reachable_ones(
    indexes_file, fake_embedder, conn, requested, message
):
    with pytest.raises(IndexRefused) as e:
        file_registry(indexes_file, lambda c, s: fake_embedder).open(conn, requested, Use.READ)
    assert str(e.value) == message


def test_a_failed_open_is_unavailable_and_retried(indexes_file, fake_embedder):
    fail = {"on": True}

    def factory(config, scheme):
        if fail["on"]:
            raise RuntimeError("probe mismatch")
        return fake_embedder

    reg = file_registry(indexes_file, factory)
    with pytest.raises(IndexUnavailable) as e:
        reg.open(Connection("alpha", None), None, Use.READ)
    assert str(e.value) == (
        "Index 'alpha' is unavailable: probe mismatch. Other reachable indexes: beta, gamma."
    )
    assert (e.value.name, e.value.reason) == ("alpha", "probe mismatch")
    assert reg.resolve(UNRESTRICTED, "alpha").state() == (IndexState.UNAVAILABLE, "probe mismatch")
    fail["on"] = False
    reg.open(Connection("alpha", None), None, Use.READ)


@pytest.mark.parametrize(
    "conn, others",
    [
        (Connection("alpha", frozenset({"gamma"})), " Other reachable indexes: gamma."),
        (Connection("alpha", frozenset()), ""),
    ],
)
def test_an_unavailable_index_ends_its_reason_once_and_names_only_the_reachable_others(
    indexes_file, conn, others
):
    def factory(config, scheme):
        raise RuntimeError("The endpoint serves another model.")

    with pytest.raises(IndexUnavailable) as e:
        file_registry(indexes_file, factory).open(conn, None, Use.READ)
    assert str(e.value) == (
        f"Index 'alpha' is unavailable: The endpoint serves another model.{others}"
    )


def test_an_unknown_model_names_where_the_known_ones_are_listed(tmp_path):
    config = load_config({"BASE_DIR": str(tmp_path), "MODEL_NAME": "x/unknown"})
    with pytest.raises(IndexUnavailable) as e:
        IndexRegistry.single(config).open(Connection(SINGLE_INDEX_NAME, None), None, Use.READ)
    message = str(e.value)
    assert message.startswith("Index 'default' is unavailable: Model 'x/unknown' is not in")
    assert message.endswith(" — see fastembed's supported models.")


def test_an_embedder_of_the_configured_model_at_another_width_is_refused(indexes_file):
    reg = file_registry(indexes_file, lambda c, s: FakeEmbedder(s.model))
    with pytest.raises(IndexUnavailable) as e:
        reg.open(UNRESTRICTED, "alpha", Use.READ)
    assert str(e.value) == (
        f"Index 'alpha' is unavailable: {E5_LARGE} makes 1024-dimensional vectors, but the "
        "embedder supplied for it makes 8-dimensional ones. Other reachable indexes: beta, gamma."
    )
    assert not (indexes_file.roots["alpha"] / ".minirag").exists()
    fitting = file_registry(indexes_file, lambda c, s: FakeEmbedder(s.model, dim=s.dim))
    _, rt = fitting.open(UNRESTRICTED, "alpha", Use.READ)
    assert (rt.identity.embedding.model, rt.identity.embedding.dim) == (E5_LARGE, 1024)


def test_a_rebuild_pending_index_answers_reads_and_syncs_but_refuses_ingests(
    indexes_file, fake_embedder, index_states
):
    index_states["beta"] = (IndexState.REBUILD_PENDING, "chunk budget 300 -> 400 tokens")
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    reg.open(UNRESTRICTED, "beta", Use.READ)
    reg.open(UNRESTRICTED, "beta", Use.SYNC)
    with pytest.raises(IndexRefused) as e:
        reg.open(UNRESTRICTED, "beta", Use.INGEST)
    message = str(e.value)
    assert not isinstance(e.value, IndexUnavailable)
    assert message.startswith("Index 'beta': ") and "needs a rebuild" in message
    assert "(chunk budget 300 -> 400 tokens)" in message
    assert message.endswith(" Other reachable indexes: alpha, gamma.")
    reg.open(UNRESTRICTED, "alpha", Use.INGEST)


def test_an_ingest_the_handle_finds_unavailable_is_refused_as_unavailable(
    indexes_file, fake_embedder, index_states
):
    index_states["gamma"] = (IndexState.UNAVAILABLE, "index.json is unreadable")
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    with pytest.raises(IndexUnavailable) as e:
        reg.open(Connection("gamma", None), None, Use.INGEST)
    assert str(e.value) == (
        "Index 'gamma' is unavailable: index.json is unreadable. Other reachable indexes: "
        "alpha, beta."
    )


def test_only_an_ingest_is_admitted_by_the_handle(indexes_file, fake_embedder, monkeypatch):
    admitted = []
    real = IndexHandle.ingest_pipeline

    def spy(self):
        admitted.append(real(self))
        return admitted[-1]

    monkeypatch.setattr(IndexHandle, "ingest_pipeline", spy)
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    reg.open(UNRESTRICTED, "alpha", Use.READ)
    reg.open(UNRESTRICTED, "alpha", Use.SYNC)
    assert admitted == []
    _, rt = reg.open(UNRESTRICTED, "alpha", Use.INGEST)
    assert len(admitted) == 1 and rt.pipeline is admitted[0]


def _reopened(where: str, every: bool = False):
    """`IndexHandle.ingest_pipeline` with the handle let go of its open index `where` the
    admission is: before it or after it. The next `get()` then opens the index again, which
    hands out another OpenIndex, as a swapped generation does. Only the first admission,
    unless `every`."""
    real = IndexHandle.ingest_pipeline
    calls = []

    def admit(self):
        calls.append(where)
        if where == "before" and (every or len(calls) == 1):
            self.close()
        pipeline = real(self)
        if where == "after" and (every or len(calls) == 1):
            self.close()
        return pipeline

    return admit


@pytest.mark.parametrize("where", ["before", "after"])
def test_an_ingest_gets_the_open_index_its_admitted_pipeline_belongs_to(
    indexes_file, fake_embedder, monkeypatch, where
):
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    reg.open(UNRESTRICTED, "alpha", Use.READ)
    monkeypatch.setattr(IndexHandle, "ingest_pipeline", _reopened(where))
    entry, rt = reg.open(UNRESTRICTED, "alpha", Use.INGEST)
    assert rt is entry.handle.get()
    assert rt.pipeline.store is rt.store and rt.pipeline.embedder is rt.embedder


def test_an_ingest_into_an_index_swapped_at_every_admission_is_refused(
    indexes_file, fake_embedder, monkeypatch
):
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    monkeypatch.setattr(IndexHandle, "ingest_pipeline", _reopened("after", every=True))
    with pytest.raises(IndexRefused) as e:
        reg.open(Connection("alpha", frozenset()), None, Use.INGEST)
    assert str(e.value) == (
        "Index 'alpha' changed twice while this ingest was being admitted (another process "
        "rebuilt it, or recorded its probe), so nothing was written. Retry it."
    )


def test_one_index_failing_its_probe_on_a_shared_model_leaves_the_other_answering(
    indexes_file, stub_text_embedding, reference_probes
):
    indexes_file.write('[server]\ncache_dir = "~/models"\n\n' + indexes_file.text)
    reg = IndexRegistry.from_file(indexes_file.path, indexes_file.env)
    alpha, beta = (reg.resolve(UNRESTRICTED, n) for n in ("alpha", "beta"))
    wrong = tuple(FakeEmbedder(dim=1024, salt="another model")._vec(PROBE_TEXT))
    for entry, probe in ((alpha, wrong), (beta, reference_probes[E5_LARGE])):
        recorded = record_probe(entry.config.db_path, entry.runtime().identity.generation, probe)
        assert recorded is not None and recorded.probe == probe
    mismatch = "differs from the one recorded when this index was built"

    beta.runtime().embedder.embed_query("bananas")
    with pytest.raises(EmbedderUnavailableError, match=mismatch):
        alpha.runtime().embedder.embed_query("apricots")
    with pytest.raises(IndexUnavailable, match=mismatch):
        reg.open(UNRESTRICTED, "alpha", Use.READ)
    _, rt = reg.open(UNRESTRICTED, "beta", Use.READ)
    rt.embedder.embed_query("budgets")
    assert beta.state() == (IndexState.READY, "") and rt.embedder.probe_error is None
    state, why = alpha.state()
    assert state is IndexState.UNAVAILABLE and mismatch in why
    assert stub_text_embedding.model_names == [E5_LARGE]


class HeldEmbedder(FakeEmbedder):
    """Embeds documents only once `release` is set; `entered` says one batch is waiting."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed_documents(self, texts):
        self.entered.set()
        self.release.wait(10)
        return super().embed_documents(texts)


def test_a_running_sync_makes_the_index_syncing(indexes_file):
    held = HeldEmbedder()
    entry = file_registry(indexes_file, lambda c, s: held).resolve(UNRESTRICTED, "alpha")
    job = entry.handle.start_sync(None)
    try:
        assert held.entered.wait(10)
        assert entry.state() == (IndexState.SYNCING, "")
    finally:
        held.release.set()
        entry.handle.sync.wait()
    assert entry.handle.sync.status(job).state == "succeeded"
    assert entry.state() == (IndexState.READY, "")


def test_every_open_goes_through_the_handle(indexes_file, fake_embedder, monkeypatch):
    got = []
    real = IndexHandle.get

    def spy(self):
        got.append(real(self))
        return got[-1]

    monkeypatch.setattr(IndexHandle, "get", spy)
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    runtimes = [reg.open(UNRESTRICTED, "alpha", Use.READ)[1] for _ in range(3)]
    assert len(got) == 3 and all(r is got[0] for r in runtimes)


def test_a_file_that_does_not_load_is_a_config_error(indexes_file):
    indexes_file.write("[server]\nport 8765\n")
    with pytest.raises(ConfigError, match=str(indexes_file.path)):
        IndexRegistry.from_file(indexes_file.path, indexes_file.env)


def test_the_file_mode_keeps_its_file_and_server_settings(indexes_file, fake_embedder):
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    assert reg.path == indexes_file.path and reg.config_error is None
    assert list(reg.server.indexes) == ["alpha", "beta", "gamma"]
    entry = reg.resolve(UNRESTRICTED, "beta")
    assert (entry.name, entry.description, entry.config.index_name) == (
        "beta",
        "Beta documents",
        "beta",
    )


def test_the_environment_mode_is_one_index_named_default(tmp_path, fake_embedder):
    reg = IndexRegistry.single(
        load_config({"BASE_DIR": str(tmp_path)}), embedder_factory=lambda c, s: fake_embedder
    )
    assert reg.names == ("default",) and (reg.path, reg.server, reg.config_error) == (
        None,
        None,
        None,
    )
    entry, _ = reg.open(Connection(SINGLE_INDEX_NAME, None), None, Use.READ)
    assert entry.name == "default"
    broken = IndexRegistry.single(None, error="BASE_DIRS must be a JSON array")
    assert broken.names == ()
    with pytest.raises(IndexRefused, match="Configuration error: BASE_DIRS"):
        broken.open(Connection("default", None), None, Use.READ)
