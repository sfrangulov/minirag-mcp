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


def test_scope_prefix_with_like_wildcards_not_overmatching(store):
    for src in ("/docs_api/f1.md", "/docsXapi/f2.md", "/100%done/f4.md", "/100Xdone/f5.md"):
        store.replace_source(src, [rec(src, 0, "x")])
    assert [s.source for s in store.list_sources(scopes=("/docs_api",))] == ["/docs_api/f1.md"]
    assert [s.source for s in store.list_sources(scopes=("/100%done",))] == ["/100%done/f4.md"]
