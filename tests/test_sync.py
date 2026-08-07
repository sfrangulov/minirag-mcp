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
