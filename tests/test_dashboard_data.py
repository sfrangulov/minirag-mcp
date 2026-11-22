from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pytest

from index_helpers import file_registry
from minirag_mcp.dashboard.data import DashboardData
from minirag_mcp.identity import IndexState
from minirag_mcp.store import Store
from minirag_mcp.sync_report import REPORT_FILE, SyncRecord, record_sync

ROOT_FILES = "(files at the root)"
FOLDER_FILES = "(files in this folder)"


def _write(root: Path, rel: str, words: int = 3) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {path.stem}\n\n" + " ".join(["word"] * words) + "\n", encoding="utf-8")
    return path


def _ingest(registry, name: str, *paths: Path) -> None:
    pipeline = next(e for e in registry.entries if e.name == name).runtime().pipeline
    for path in paths:
        pipeline.ingest_file(path)


@pytest.fixture
def registry(indexes_file, fake_embedder):
    return file_registry(indexes_file, lambda c, s: fake_embedder)


def _folder(root, name):
    return next(f for f in root.folders if f.name == name)


def test_documents_and_chunks_add_up_per_root_and_folder(indexes_file, registry):
    root = indexes_file.roots["alpha"]
    other = root.parent / "second"
    other.mkdir()
    paths = [
        _write(root, "a/x.md"),
        _write(root, "a/b/y.md"),
        _write(root, "a/b/c/z.md"),
        _write(root, "d/w.md"),
        _write(other, "e/v.md"),
    ]
    text = indexes_file.text.replace(
        'roots = ["~/corpora/alpha"]', f'roots = ["~/corpora/alpha", "{other}"]'
    )
    indexes_file.write(text)
    registry.reload()
    _ingest(registry, "alpha", *paths, root / "alpha.md")
    store = registry.entries[0].runtime().store
    per = {i.source: i.chunk_count for i in store.list_sources()}

    view = DashboardData(registry).index("alpha")

    assert view.documents == 6 and view.chunks == sum(per.values())
    by_path = {r.path: r for r in view.roots}
    first = by_path[str(root.resolve())]
    assert first.documents == 5
    assert first.chunks == sum(v for k, v in per.items() if k.startswith(str(root.resolve())))
    a = _folder(first, "a")
    assert a.documents == 3
    assert (
        a.chunks
        == per[str(paths[0].resolve())]
        + per[str(paths[1].resolve())]
        + per[str(paths[2].resolve())]
    )
    assert a.path == str(root.resolve() / "a")
    assert {c.name: c.documents for c in a.children} == {FOLDER_FILES: 1, "b": 2}
    assert _folder(first, "d").children[0].name == FOLDER_FILES
    second = by_path[str(other.resolve())]
    assert second.documents == 1 and _folder(second, "e").documents == 1


def test_folders_are_ranked_by_chunks(indexes_file, registry):
    root = indexes_file.roots["alpha"]
    _ingest(
        registry,
        "alpha",
        _write(root, "small/s.md", words=2),
        _write(root, "big/b.md", words=400),
        _write(root, "big/sub1/p.md", words=2),
        _write(root, "big/sub2/q.md", words=300),
    )
    view = DashboardData(registry).index("alpha")
    folders = view.roots[0].folders
    assert [f.name for f in folders][0] == "big"
    assert [f.chunks for f in folders] == sorted((f.chunks for f in folders), reverse=True)
    children = folders[0].children
    assert [c.chunks for c in children] == sorted((c.chunks for c in children), reverse=True)
    assert len(children) == 3


def test_nfd_and_nfc_spellings_of_a_folder_are_one_folder(indexes_file, registry):
    root = indexes_file.roots["alpha"]
    nfc = unicodedata.normalize("NFC", "Отчёт")
    nfd = unicodedata.normalize("NFD", "Отчёт")
    assert nfc != nfd
    _ingest(registry, "alpha", _write(root, f"{nfc}/one.md"))
    # A cloud drive hands the same folder back decomposed; the store keeps whatever it saw.
    entry = registry.entries[0]
    pipeline = entry.runtime().pipeline
    pipeline.ingest_data("# Two\n\nbody text", f"{root.resolve()}/{nfd}/two.md", "text")
    view = DashboardData(registry).index("alpha")
    names = [f.name for f in view.roots[0].folders]
    assert names == [nfc]
    assert view.roots[0].folders[0].documents == 2


def test_files_at_a_root_get_their_own_row(indexes_file, registry):
    root = indexes_file.roots["alpha"]
    _ingest(registry, "alpha", root / "alpha.md", _write(root, "sub/s.md"))
    view = DashboardData(registry).index("alpha")
    folders = {f.name: f for f in view.roots[0].folders}
    assert folders[ROOT_FILES].documents == 1
    assert folders[ROOT_FILES].path == str(root.resolve())
    assert folders["sub"].children[0].name == FOLDER_FILES


def test_an_unavailable_index_gets_its_reason_and_no_statistics(
    indexes_file, registry, index_states
):
    _ingest(registry, "alpha", indexes_file.roots["alpha"] / "alpha.md")
    index_states["beta"] = (IndexState.UNAVAILABLE, "index.json is unreadable")
    views = {v.name: v for v in DashboardData(registry).overview()}
    assert list(views) == ["alpha", "beta", "gamma"]
    beta = views["beta"]
    assert beta.state == "unavailable" and beta.why == "index.json is unreadable"
    assert beta.documents == beta.chunks == 0 and beta.roots == ()
    assert views["alpha"].documents == 1 and views["alpha"].state == "ready"
    assert DashboardData(registry).folder("beta", str(indexes_file.roots["beta"])) is None


def test_folder_lists_the_files_under_a_path(indexes_file, registry):
    root = indexes_file.roots["alpha"]
    _ingest(registry, "alpha", _write(root, "a/x.md"), _write(root, "a/b/y.md"))
    data = DashboardData(registry)
    rows = data.folder("alpha", str(root.resolve() / "a"))
    assert sorted(Path(r.source).name for r in rows) == ["x.md", "y.md"]
    assert all(r.chunks >= 1 and r.ingested for r in rows)
    assert len(data.folder("alpha", str(root.resolve()))) == 2
    assert data.folder("alpha", str(root.resolve() / "a" / "b"))[0].title == "y"
    assert data.folder("alpha", str(root.resolve() / "nothing")) is None
    assert data.folder("nope", str(root)) is None


def test_folder_outside_every_root_or_with_dotdot_gives_none(indexes_file, registry, monkeypatch):
    root = indexes_file.roots["alpha"]
    _ingest(registry, "alpha", _write(root, "a/x.md"))
    data = DashboardData(registry)
    data.overview()

    corpus = str(root.parent.resolve())
    real_stat, real_scandir = Path.stat, os.scandir

    # The index's own files are read to tell whether it changed; the documents never are.
    def guarded(real):
        def call(path, *a, **k):
            if str(path).startswith((corpus, str(root.parent))) and ".minirag" not in str(path):
                raise AssertionError(f"the disk was touched: {path}")
            return real(path, *a, **k)

        return call

    resolved = root.resolve()
    with monkeypatch.context() as patched:
        patched.setattr(Path, "stat", guarded(real_stat))
        patched.setattr(os, "scandir", guarded(real_scandir))
        for bad in (
            "../x",
            "/etc",
            str(resolved.parent),
            f"{resolved}/../beta",
            f"{resolved}/a/..",
        ):
            assert data.folder("alpha", bad) is None
        assert data.folder("alpha", str(resolved / "a")) is not None


def test_statistics_are_cached_until_the_index_changes(indexes_file, registry, monkeypatch):
    root = indexes_file.roots["alpha"]
    _ingest(registry, "alpha", root / "alpha.md")
    calls = []
    real = Store.list_sources

    def spy(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(Store, "list_sources", spy)
    data = DashboardData(registry)
    data.overview()
    first = len(calls)
    data.overview()
    data.folder("alpha", str(root.resolve()))
    assert len(calls) == first
    _ingest(registry, "alpha", _write(root, "n.md"))
    assert data.index("alpha").documents == 2
    assert len(calls) > first


def test_the_report_is_attached_to_its_index(indexes_file, registry):
    db = registry.entries[0].config.db_path
    data = DashboardData(registry)
    assert data.index("alpha").latest is None and data.index("alpha").report_problem is None
    record = SyncRecord(
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:05+00:00",
        ran_by="cli",
        scope=None,
        outcome="succeeded",
        error=None,
        counts={"ingested": 1},
        errors=[{"source": "a.pdf", "error": "boom"}],
        rebuild=None,
    )
    assert record_sync(db, record) is None
    view = data.index("alpha")
    assert view.latest["outcome"] == "succeeded"
    assert view.coverage["errors"] == [{"source": "a.pdf", "error": "boom"}]
    assert view.report_problem is None
    path = db / REPORT_FILE
    path.write_text("{truncated", encoding="utf-8")
    os.utime(path, ns=(1, 1))
    view = data.index("alpha")
    assert "last-sync.json" in view.report_problem
    assert view.latest is None and view.coverage is None
    assert data.index("beta").report_problem is None


def test_one_index_computing_never_blocks_the_others(indexes_file, registry, monkeypatch):
    import threading

    _ingest(registry, "alpha", indexes_file.roots["alpha"] / "alpha.md")
    _ingest(registry, "beta", indexes_file.roots["beta"] / "beta.md")
    data = DashboardData(registry)
    data.index("alpha")  # cached
    entered, release = threading.Event(), threading.Event()
    beta_store = next(e for e in registry.entries if e.name == "beta").runtime().store
    real = Store.list_sources

    def stub(self, *a, **k):
        if self is beta_store:
            entered.set()
            assert release.wait(30)
        return real(self, *a, **k)

    monkeypatch.setattr(Store, "list_sources", stub)
    slow = threading.Thread(target=data.index, args=("beta",))
    slow.start()
    try:
        assert entered.wait(30)
        done: dict = {}

        def others():
            done["hit"] = data.index("alpha")
            done["fresh"] = data.index("gamma")

        t = threading.Thread(target=others)
        t.start()
        t.join(10)
        finished = not t.is_alive()
    finally:
        release.set()
        slow.join(30)
    assert finished and done["hit"].documents == 1 and done["fresh"].name == "gamma"
