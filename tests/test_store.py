import os

import pytest

from minirag_mcp.store import ChunkRecord, DimensionMismatchError, Store


def rec(source, i, text, vec=None, source_type="file", title="T"):
    return ChunkRecord(
        id=f"{source}#{i}",
        source=source,
        source_type=source_type,
        title=title,
        chunk_index=i,
        text=text,
        vector=vec or [0.1 * (i + 1)] * 8,
        file_hash="h",
        mtime=1.0,
        ingested_at="2026-08-07T00:00:00+00:00",
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db", dim=8)


def test_empty_store_counts(store):
    assert store.chunk_count() == 0
    assert store.source_count() == 0
    assert store.list_sources() == []
    assert store.get_source("/nope.md") is None


def test_replace_and_get(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "alpha"), rec("/a.md", 1, "beta")])
    assert store.chunk_count() == 2
    info = store.get_source("/a.md")
    assert info.chunk_count == 2 and info.file_hash == "h" and info.source_type == "file"


def test_replace_is_atomic_swap(store):
    store.replace_source("/a.md", [rec("/a.md", i, f"t{i}") for i in range(3)])
    store.replace_source("/a.md", [rec("/a.md", 0, "new")])
    assert store.chunk_count() == 1
    assert store.get_source("/a.md").chunk_count == 1


def test_delete_source_returns_count(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "x")])
    assert store.delete_source("/a.md") == 1
    assert store.delete_source("/a.md") == 0
    assert store.chunk_count() == 0


def test_sql_injection_safe_source_names(store):
    evil = "/o'brien's notes.md"
    store.replace_source(evil, [rec(evil, 0, "x")])
    assert store.get_source(evil) is not None
    assert store.delete_source(evil) == 1


def test_neighbors_window_and_order(store):
    store.replace_source("/a.md", [rec("/a.md", i, f"chunk {i}") for i in range(30)])
    got = store.neighbors("/a.md", 15, before=2, after=2)
    # >10 rows must survive default limits
    assert [g.chunk_index for g in got] == [13, 14, 15, 16, 17]
    edge = store.neighbors("/a.md", 0, before=3, after=1)
    assert [g.chunk_index for g in edge] == [0, 1]


def test_all_chunks_full_order(store):
    store.replace_source("/a.md", [rec("/a.md", i, f"chunk {i}") for i in range(25)])
    got = store.all_chunks("/a.md")
    assert [g.chunk_index for g in got] == list(range(25))  # all rows, ordered
    assert store.all_chunks("/missing.md") == []


def test_list_sources_scopes_and_many_rows(store):
    for n in range(15):
        src = f"/docs/api/f{n:02d}.md"
        store.replace_source(src, [rec(src, 0, "x")])
    store.replace_source("/other/z.md", [rec("/other/z.md", 0, "x")])
    store.replace_source("https://x.io/p", [rec("https://x.io/p", 0, "x", source_type="url")])
    assert store.source_count() == 17
    scoped = store.list_sources(scopes=("/docs/api",))
    assert len(scoped) == 15 and all(s.source.startswith("/docs/api") for s in scoped)
    everything = store.list_sources()
    assert len(everything) == 17
    assert everything == sorted(everything, key=lambda s: s.source)


def test_persistence_across_instances(tmp_path):
    s1 = Store(tmp_path / "db", dim=8)
    s1.replace_source("/a.md", [rec("/a.md", 0, "x")])
    s2 = Store(tmp_path / "db", dim=8)  # reopen, no create conflict
    assert s2.chunk_count() == 1


def test_reopening_with_a_different_dim_names_both_dimensions(tmp_path):
    """A MODEL_NAME change against an existing DB_PATH must say so, not blame the schema."""
    Store(tmp_path / "db", dim=8).replace_source("/a.md", [rec("/a.md", 0, "x")])
    with pytest.raises(DimensionMismatchError) as exc:
        Store(tmp_path / "db", dim=16)
    msg = str(exc.value)
    assert "8" in msg and "16" in msg
    assert "no vector column" not in msg
    assert "DB_PATH" in msg or "re-ingest" in msg  # names a remedy


def test_reopening_with_the_same_dim_is_fine(tmp_path):
    Store(tmp_path / "db", dim=8).replace_source("/a.md", [rec("/a.md", 0, "x")])
    assert Store(tmp_path / "db", dim=8).chunk_count() == 1


def _fts_columns(store) -> set[str]:
    return {c for idx in store._table.list_indices() for c in idx.columns}


def test_new_table_indexes_both_text_and_title(store):
    assert _fts_columns(store) == {"text", "title"}


def test_reopening_an_older_index_adds_the_missing_title_index(tmp_path):
    """A DB written before titles were searchable must gain the index on open —
    without that, existing users would need a full re-ingest to search filenames."""
    s1 = Store(tmp_path / "db", dim=8)
    s1._table.drop_index("title_idx")  # simulate a table created by the older version
    assert _fts_columns(s1) == {"text"}
    s1.replace_source("/a.md", [rec("/a.md", 0, "alpha", title="Sputnik Handbook")])

    s2 = Store(tmp_path / "db", dim=8)
    assert _fts_columns(s2) == {"text", "title"}
    assert s2.chunk_count() == 1  # the upgrade did not disturb the data
    rows = s2._table.search("Sputnik", query_type="fts", fts_columns=["title"]).to_list()
    assert [r["id"] for r in rows] == ["/a.md#0"]


def _chmod_tree(root, dir_mode: int, file_mode: int) -> None:
    for path, dirs, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(os.path.join(path, name), file_mode)
        for name in dirs:
            os.chmod(os.path.join(path, name), dir_mode)
    os.chmod(root, dir_mode)


def test_opening_a_read_only_database_serves_reads(tmp_path):
    """Reading a database must never require write access — the title-index upgrade is
    an opportunistic extra, not a precondition for opening."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses file permissions")
    db = tmp_path / "db"
    s1 = Store(db, dim=8)
    s1._table.drop_index("title_idx")  # a table written by the older version
    s1.replace_source("/a.md", [rec("/a.md", 0, "alpha body", title="Sputnik Handbook")])

    _chmod_tree(db, 0o555, 0o444)
    try:
        with pytest.warns(RuntimeWarning, match="title"):
            s2 = Store(db, dim=8)
        assert s2.missing_fts_indices == ("title",)
        assert s2.chunk_count() == 1
        got = s2.search("alpha", [0.1] * 8, top_k=5, hybrid_weight=0.6)
        assert [r.source for r in got] == ["/a.md"]
    finally:
        _chmod_tree(db, 0o755, 0o644)


def test_index_creation_failure_does_not_propagate(tmp_path, monkeypatch):
    """Concurrent opens of a not-yet-upgraded database race on the same index write;
    the loser gets a commit conflict and must still hand back a usable Store."""
    db = tmp_path / "db"
    s1 = Store(db, dim=8)
    s1._table.drop_index("title_idx")
    s1.replace_source("/a.md", [rec("/a.md", 0, "alpha body", title="Sputnik Handbook")])

    def boom(*args, **kwargs):
        raise RuntimeError("Retryable commit conflict for version 65")

    monkeypatch.setattr(type(s1._table), "create_index", boom)
    with pytest.warns(RuntimeWarning, match="title"):
        s2 = Store(db, dim=8)
    assert s2.missing_fts_indices == ("title",)
    assert _fts_columns(s2) == {"text"}
    monkeypatch.undo()
    got = s2.search("alpha", [0.1] * 8, top_k=5, hybrid_weight=0.6)
    assert [r.source for r in got] == ["/a.md"]


def test_scope_stops_at_a_path_component_boundary(store):
    for src in ("/data/proj/f.md", "/data/project-secret/f.md"):
        store.replace_source(src, [rec(src, 0, "x")])
    assert [s.source for s in store.list_sources(scopes=("/data/proj",))] == ["/data/proj/f.md"]
    # a trailing separator names the same scope
    assert [s.source for s in store.list_sources(scopes=("/data/proj/",))] == ["/data/proj/f.md"]


def test_scope_matches_the_scoped_source_itself(store):
    store.replace_source("/data/proj", [rec("/data/proj", 0, "x")])
    store.replace_source("/data/project-secret", [rec("/data/project-secret", 0, "x")])
    assert [s.source for s in store.list_sources(scopes=("/data/proj",))] == ["/data/proj"]


def test_url_scope_stops_at_a_path_component_boundary(store):
    for src in ("https://example.com/docs/page", "https://example.com/docs-private/page"):
        store.replace_source(src, [rec(src, 0, "x", source_type="url")])
    scoped = store.list_sources(scopes=("https://example.com/docs",))
    assert [s.source for s in scoped] == ["https://example.com/docs/page"]


def test_scope_prefix_with_like_wildcards_not_overmatching(store):
    for src in ("/docs_api/f1.md", "/docsXapi/f2.md", "/100%done/f4.md", "/100Xdone/f5.md"):
        store.replace_source(src, [rec(src, 0, "x")])
    assert [s.source for s in store.list_sources(scopes=("/docs_api",))] == ["/docs_api/f1.md"]
    assert [s.source for s in store.list_sources(scopes=("/100%done",))] == ["/100%done/f4.md"]
