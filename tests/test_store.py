import os
from dataclasses import asdict, replace

import lancedb.db
import pytest

from minirag_mcp.store import ChunkRecord, DimensionMismatchError, Store
from store_helpers import _older_table, _raise_commit_conflict, _row


def rec(source, i, text, vec=None, source_type="file", title="T", ocr_engine=""):
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
        ocr_engine=ocr_engine,
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


def test_replacing_a_source_never_leaves_a_version_without_it(store):
    """Every table version is what some reader may see.

    As a delete followed by an add, a re-ingest left one version in which the source
    was gone: a concurrent reader saw it missing, and a second writer whose delete and
    add interleaved with these left both copies, which no later sync repairs because
    the hash and mtime match.
    """
    store.replace_source("/a.md", [rec("/a.md", i, f"old {i}") for i in range(3)])
    before = store._table.version
    store.replace_source("/a.md", [rec("/a.md", i, f"new {i}") for i in range(2)])
    after = store._table.version

    clause = "source = '/a.md'"
    rows_per_version = {}
    for version in range(before, after + 1):
        store._table.checkout(version)
        rows_per_version[version] = store._table.count_rows(filter=clause)
    store._table.checkout_latest()

    assert rows_per_version == {before: 3, after: 2}
    texts = sorted(r.text for r in store.all_chunks("/a.md"))
    assert texts == ["new 0", "new 1"]


def test_replacing_one_source_leaves_the_others_alone(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "a")])
    store.replace_source("/a.md#b.md", [rec("/a.md#b.md", 0, "b0"), rec("/a.md#b.md", 1, "b1")])
    store.replace_source("/a.md", [rec("/a.md", 0, "a2")])
    store.replace_source("/a.md", [])
    assert [(s.source, s.chunk_count) for s in store.list_sources()] == [("/a.md#b.md", 2)]


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


def test_an_open_handle_sees_what_another_handle_writes(tmp_path):
    """The MCP server keeps one Store for its lifetime while a terminal `sync` writes.

    A handle that read the table pinned that version: it went on listing, counting and
    searching the old rows until it wrote something itself, and a sync planned from
    that stale listing duplicated every re-pathed source.
    """
    resident = Store(tmp_path / "db", dim=8)
    resident.replace_source("/old.md", [rec("/old.md", 0, "old text")])
    assert resident.chunk_count() == 1

    other = Store(tmp_path / "db", dim=8)
    other.delete_source("/old.md")
    other.replace_source("/new.md", [rec("/new.md", 0, "new text"), rec("/new.md", 1, "more")])

    assert [s.source for s in resident.list_sources()] == ["/new.md"]
    assert resident.chunk_count() == 2
    hits = resident.search("new text", [0.1] * 8, top_k=5, include_parents=False)
    assert {h.source for h in hits} == {"/new.md"}


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


def test_ocr_engine_round_trips_and_aggregates(store):
    store.replace_source(
        "/s.md",
        [
            rec("/s.md", 0, "alpha", ocr_engine="rapidocr"),
            rec("/s.md", 1, "beta"),
        ],
    )
    info = store.get_source("/s.md")
    assert info.ocr_engine == "rapidocr"
    assert [s.ocr_engine for s in store.list_sources()] == ["rapidocr"]


def test_reopening_a_table_missing_the_ocr_column_migrates_it(tmp_path):
    """A pre-OCR index must gain this column on open: builds a `chunks` table with the
    create-table schema minus `ocr_engine` — a genuine pre-OCR index, the same shape
    `_ensure_scheme_columns` already had to migrate for `parent_id`/`scheme_version` —
    then opens it through `Store` and proves both the column and the two read paths
    that depend on it survive."""
    db_path = tmp_path / "db"
    _older_table(db_path, {"ocr_engine"}, [_row("/a.md", 0, "x")])

    store = Store(db_path, dim=8)
    assert "ocr_engine" in store._table.schema.names
    assert store.missing_columns == ()
    assert store.get_source("/a.md").ocr_engine == ""
    assert [s.ocr_engine for s in store.list_sources()] == [""]


def test_a_failed_ocr_migration_does_not_make_the_scheme_look_stale(tmp_path, monkeypatch):
    """`ocr_engine` is carried through storage and read by nothing that judges a chunk's
    age. An index that cannot gain it — a read-only copy, or one with a concurrent
    writer — must keep reporting scheme staleness from the chunks themselves and must
    keep rebuilding parent sections."""
    db_path = tmp_path / "db"
    rows = [_row("/a.md", 0, "первый кусок", "/a.md#p0"), _row("/a.md", 1, "второй", "/a.md#p0")]
    table = _older_table(db_path, {"ocr_engine"}, rows)
    monkeypatch.setattr(type(table), "add_columns", _raise_commit_conflict)

    with pytest.warns(RuntimeWarning, match="ocr_engine") as caught:
        store = Store(db_path, dim=8)
    assert not [w for w in caught if "predates the current chunking scheme" in str(w.message)]

    assert store.schema_migrated
    assert store.missing_columns == ("ocr_engine",)
    assert store.stale_chunk_count() == 0
    assert store.parent_text("/a.md#p0") == "первый кусок\n\nвторой"
    assert [s.ocr_engine for s in store.list_sources()] == [""]


def test_a_failed_scheme_migration_still_reports_every_chunk_stale(tmp_path, monkeypatch):
    """The converse: without `parent_id`/`scheme_version` there is nothing to tell an
    old chunk from a new one and no section to rebuild, so the pessimistic reading is
    the only honest one."""
    db_path = tmp_path / "db"
    table = _older_table(
        db_path,
        {"parent_id", "scheme_version", "ocr_engine"},
        [_row("/a.md", i, f"кусок {i}") for i in range(3)],
    )
    monkeypatch.setattr(type(table), "add_columns", _raise_commit_conflict)

    with pytest.warns(RuntimeWarning, match="predates the current chunking scheme"):
        store = Store(db_path, dim=8)

    assert not store.schema_migrated
    assert store.missing_columns == ("ocr_engine", "parent_id", "scheme_version")
    assert store.stale_chunk_count() == 3
    assert store.parent_texts(["/a.md#p0"]) == {}


def test_open_failure_that_is_not_a_missing_table_propagates(tmp_path, monkeypatch):
    db_path = tmp_path / "db"
    Store(db_path, dim=8)  # the table now exists

    def unreadable(self, name, **kwargs):
        raise RuntimeError("lance error: LanceError(IO): Permission denied (os error 13)")

    monkeypatch.setattr(lancedb.db.LanceDBConnection, "open_table", unreadable)
    with pytest.raises(RuntimeError, match="Permission denied"):
        Store(db_path, dim=8)


@pytest.mark.parametrize(
    "old, new",
    [
        ("/docs/a.md", "/docs/sub/a.md"),
        ("/Документы/Счёт.pdf", "/Документы/Архив 2026/Счёт.pdf"),
        # decomposed й, as macOS folder names often are: substring arithmetic in
        # bytes rather than characters would cut the id in the wrong place
        ("/docs/Пйсьмо.md", "/d/Пйсьмо.md"),
        ("/it's/a#1.md", "/moved/it's/a#1.md"),
    ],
)
def test_move_source_repaths_every_row_in_one_commit_and_keeps_the_vectors(store, old, new):
    from dataclasses import replace as dc_replace

    rows = [
        dc_replace(rec(old, i, f"text {i}", vec=[0.1 * (i + 1)] * 8), parent_id=f"{old}#p{i // 2}")
        for i in range(3)
    ]
    store.replace_source(old, rows)
    store.replace_source("/other.md", [rec("/other.md", 0, "other")])
    stored = sorted(
        store._table.search().where(f"source = '{old.replace(chr(39), chr(39) * 2)}'").to_list(),
        key=lambda r: r["chunk_index"],
    )
    version = store._table.version

    assert store.move_source(old, new, mtime=42.5) == 3

    assert store._table.version == version + 1
    assert [s.source for s in store.list_sources()] == sorted([new, "/other.md"])
    moved = store._table.search().where(f"source = '{new.replace(chr(39), chr(39) * 2)}'")
    got = sorted(moved.limit(100).to_list(), key=lambda r: r["chunk_index"])
    assert [r["id"] for r in got] == [f"{new}#0", f"{new}#1", f"{new}#2"]
    assert [r["parent_id"] for r in got] == [f"{new}#p0", f"{new}#p0", f"{new}#p1"]
    assert [r["text"] for r in got] == ["text 0", "text 1", "text 2"]
    assert [list(r["vector"]) for r in got] == [list(r["vector"]) for r in stored]
    assert {r["mtime"] for r in got} == {42.5}
    assert store.get_source("/other.md").chunk_count == 1


def test_move_source_of_an_unknown_source_moves_nothing(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "x")])
    assert store.move_source("/nope.md", "/b.md", mtime=1.0) == 0
    assert [s.source for s in store.list_sources()] == ["/a.md"]


def test_replacing_a_source_clears_chunks_a_racing_first_ingest_stored_twice(store):
    """Two first ingests of one new file both see it absent, so both insert.

    A merge keyed on id alone updates both copies of every matched id, and the
    document kept double weight in ranking and every chunk twice in its sections.
    """
    doubled = [rec("/a.md", i, f"old {i}") for i in range(3)]
    store._table.add([asdict(r) for r in doubled + doubled])
    assert store.chunk_count() == 6

    store.replace_source(
        "/a.md", [replace(rec("/a.md", i, f"new {i}"), ingested_at="later") for i in range(2)]
    )

    assert sorted(c.text for c in store.all_chunks("/a.md")) == ["new 0", "new 1"]
