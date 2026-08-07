import pytest

from minirag_mcp.config import load_config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.store import Store
from minirag_mcp.sync import SyncBusyError, SyncManager, run_sync


@pytest.fixture
def env(tmp_path, fake_embedder):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    store = Store(tmp_path / ".minirag" / "lancedb", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)
    return cfg, store, pipeline, tmp_path


def seed(root):
    (root / "sub").mkdir(exist_ok=True)
    (root / "a.md").write_text("# A\n\nAlpha body text for the corpus.")
    (root / "sub" / "b.md").write_text("# B\n\nBeta body text for the corpus.")


def test_run_sync_ingests_and_deletes(env):
    cfg, store, pipeline, root = env
    seed(root)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 2 and counts["scanned"] == 2 and errors == []
    (root / "a.md").unlink()
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["deleted"] == 1 and counts["skipped"] == 1
    assert store.get_source(str(root / "a.md")) is None


def test_run_sync_collects_per_file_errors(env, monkeypatch):
    cfg, store, pipeline, root = env
    seed(root)

    real = pipeline.ingest_file

    def flaky(path):
        if path.name == "b.md":
            raise RuntimeError("boom")
        return real(path)

    monkeypatch.setattr(pipeline, "ingest_file", flaky)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 1 and counts["failed"] == 1
    assert errors and "boom" in errors[0]["error"]


def test_manager_lifecycle(env):
    cfg, store, pipeline, root = env
    seed(root)
    mgr = SyncManager(pipeline, store, cfg)
    job_id = mgr.start()
    mgr.wait()
    job = mgr.status(job_id)
    assert job.state == "succeeded"
    assert job.counts["ingested"] == 2
    d = job.to_dict()
    assert d["jobId"] == job_id and d["startedAt"] and d["finishedAt"]


def test_manager_rejects_concurrent_and_forgets_old(env):
    import threading

    cfg, store, pipeline, root = env
    seed(root)
    gate = threading.Event()
    real = pipeline.ingest_file

    def slow(path):
        gate.wait(5)
        return real(path)

    pipeline.ingest_file = slow
    mgr = SyncManager(pipeline, store, cfg)
    first = mgr.start()
    with pytest.raises(SyncBusyError):
        mgr.start()
    gate.set()
    mgr.wait()
    second = mgr.start()
    mgr.wait()
    with pytest.raises(KeyError):
        mgr.status(first)  # only the latest job is retained
    assert mgr.status(second).state == "succeeded"


def test_unknown_job_id(env):
    cfg, store, pipeline, _ = env
    mgr = SyncManager(pipeline, store, cfg)
    with pytest.raises(KeyError):
        mgr.status("nope")


def test_run_sync_scope_single_file(env):
    cfg, store, pipeline, root = env
    seed(root)
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size, scope=root / "a.md")
    assert counts["ingested"] == 1


def test_run_sync_scope_through_symlink(env, tmp_path):
    """Regression: an unresolved scope silently matched zero files (roots are resolved)."""
    cfg, store, pipeline, root = env
    seed(root)
    link = tmp_path / "link_root"
    link.symlink_to(root, target_is_directory=True)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size, scope=link / "sub")
    assert counts["ingested"] == 1 and errors == []


def test_run_sync_never_indexes_symlink_escaping_the_root(env, tmp_path):
    """Security: a symlink inside a root pointing outside it must not be ingested.

    The extension whitelist sees the link name (.md) while the parser reads the
    target, so without a containment check any readable file — extension-less
    private keys included — lands in the index.
    """
    cfg, store, pipeline, root = env
    outside = tmp_path.parent / f"{tmp_path.name}_outside"  # sibling of the document root
    outside.mkdir()
    secret = outside / "id_rsa"
    secret.write_text("-----BEGIN PRIVATE KEY-----\nSUPERSECRET\n-----END PRIVATE KEY-----\n")
    (root / "readme.md").symlink_to(secret)
    (root / "real.md").write_text("# Real\n\nOrdinary body text that belongs in the corpus.")

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["ingested"] == 1 and errors == []
    assert store.get_source(str(root / "readme.md")) is None
    assert all("SUPERSECRET" not in c.text for c in store.all_chunks(str(root / "real.md")))
    indexed = {s.source for s in store.list_sources()}
    assert indexed == {str(root / "real.md")}


def test_run_sync_purges_content_leaked_by_an_escaping_symlink(env, tmp_path):
    """An index written before the containment check is cleaned up by the next sync."""
    cfg, store, pipeline, root = env
    outside = tmp_path.parent / f"{tmp_path.name}_leak"
    outside.mkdir()
    secret = outside / "id_rsa"
    secret.write_text("-----BEGIN PRIVATE KEY-----\nLEAKED\n-----END PRIVATE KEY-----\n")
    link = root / "readme.md"
    link.symlink_to(secret)
    pipeline.ingest_file(link)  # simulate the pre-fix state: the target got indexed
    assert store.get_source(str(link)) is not None

    run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert store.get_source(str(link)) is None
    assert store.chunk_count() == 0


def test_finished_at_set_when_state_terminal(env):
    cfg, store, pipeline, root = env
    seed(root)
    mgr = SyncManager(pipeline, store, cfg)
    job_id = mgr.start()
    mgr.wait()
    job = mgr.status(job_id)
    assert job.state in ("succeeded", "failed") and job.finished_at is not None
