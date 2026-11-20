"""The `documents` table: one row per source holding its whole text, kept beside the
chunks it was joined from, and the whole-document BM25 rank it adds to search."""

import os
from dataclasses import asdict

import pytest
from lancedb.table import LanceTable

from minirag_mcp.chunker import join_document
from minirag_mcp.results import fts_status
from minirag_mcp.store import DOCUMENT_TABLE, FTS_LANGUAGE_FILENAME, ChunkRecord, Store

STAMP = "2026-11-19T00:00:00+00:00"
LATER = "2026-11-20T00:00:00+00:00"


def rec(source, i, text, vec=None, *, title="T", stamp=STAMP, parent_id="", file_hash="h"):
    return ChunkRecord(
        id=f"{source}#{i}",
        source=source,
        source_type="file",
        title=title,
        chunk_index=i,
        text=text,
        vector=vec or [0.1 * (i + 1)] * 8,
        file_hash=file_hash,
        mtime=1.0,
        ingested_at=stamp,
        parent_id=parent_id,
    )


def V(x):  # 8-dim vector pointing "x of the way" between axis 0 and axis 1
    return [1.0 - x, x] + [0.0] * 6


def rows(store) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in store._db.open_table(DOCUMENT_TABLE).to_arrow().to_pylist():
        out.setdefault(row["source"], []).append(row)
    return out


def doc_hits(store, query: str) -> list[str]:
    table = store._db.open_table(DOCUMENT_TABLE)
    found = table.search(query, query_type="fts", fts_columns=["text", "title"]).to_list()
    return [r["source"] for r in found]


def unindexed(table) -> dict[str, int]:
    return {i.name: table.index_stats(i.name).num_unindexed_rows for i in table.list_indices()}


def ranked(results) -> list[tuple[str, int]]:
    return [(r.source, r.chunk_index) for r in results]


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db", dim=8)


SECTIONED = [
    rec("/a.md", 0, "# Guide\n\nfirst part", title="Guide Book", parent_id="/a.md#p0"),
    rec("/a.md", 1, "# Guide\n\nsecond part", title="Guide Book", parent_id="/a.md#p0"),
    rec("/a.md", 2, "# Annex\n\nthird part", title="Guide Book", parent_id="/a.md#p1"),
]


def test_replace_source_writes_one_row_with_the_documents_joined_text(store):
    store.replace_source("/a.md", SECTIONED[::-1])
    assert rows(store) == {
        "/a.md": [
            {
                "source": "/a.md",
                "title": "Guide Book",
                "text": join_document(SECTIONED),
                "file_hash": "h",
                "ingested_at": STAMP,
            }
        ]
    }
    # The heading each unit repeats is written once, as `read_file` shows the text.
    assert join_document(SECTIONED) != "\n\n".join(r.text for r in SECTIONED)


def test_reingest_delete_and_move_keep_one_row_per_source(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "first version")])
    store.replace_source("/a.md", [rec("/a.md", 0, "second version", stamp=LATER)])
    [row] = rows(store)["/a.md"]
    assert (row["text"], row["ingested_at"]) == ("second version", LATER)

    assert store.move_source("/a.md", "/sub/a.md", mtime=2.0) == 1
    assert list(rows(store)) == ["/sub/a.md"]
    [moved] = rows(store)["/sub/a.md"]
    assert moved == {**row, "source": "/sub/a.md"}
    assert store.undocumented_sources() == []

    assert store.delete_source("/sub/a.md") == 1
    assert rows(store) == {}

    store.replace_source("/b.md", [rec("/b.md", 0, "b")])
    store.replace_source("/b.md", [])
    assert rows(store) == {}


def test_one_write_of_several_rows_matches_one_replace_each(tmp_path):
    a = [rec("/a.md", i, f"alpha part {i}") for i in range(3)]
    b = [rec("/b.md", 0, "beta", title="B", file_hash="hb", stamp=LATER)]
    one_each = Store(tmp_path / "one", dim=8)
    one_each.replace_source("/a.md", a)
    one_each.replace_source("/b.md", b)

    batched = Store(tmp_path / "batched", dim=8)
    batched._table.add([asdict(r) for r in a + b])
    table = batched._db.open_table(DOCUMENT_TABLE)
    before = table.version
    batched._write_documents([batched._document_row("/a.md", a), batched._document_row("/b.md", b)])
    assert table.version == before + 1
    assert rows(batched) == rows(one_each)


def test_document_rows_match_with_the_index_stemmer(tmp_path):
    russian = Store(tmp_path / "ru", dim=8, fts_language="Russian")
    russian.replace_source("/c.md", [rec("/c.md", 0, "договор поставки")])
    assert doc_hits(russian, "договоры") == ["/c.md"]

    english = Store(tmp_path / "en", dim=8)
    english.replace_source("/c.md", [rec("/c.md", 0, "договор поставки")])
    assert doc_hits(english, "договоры") == []


def test_a_language_change_rebuilds_the_document_index_too(tmp_path):
    db = tmp_path / "db"
    english = Store(db, dim=8)
    english.replace_source("/c.md", [rec("/c.md", 0, "договор поставки")])
    english.update_indices()
    assert doc_hits(english, "договоры") == []

    russian = Store(db, dim=8, fts_language="Russian")
    assert doc_hits(russian, "договоры") == ["/c.md"]
    assert russian.fts_index_language == "Russian"
    assert (db / FTS_LANGUAGE_FILENAME).read_text(encoding="utf-8").strip() == "Russian"


def test_a_failed_document_index_rebuild_is_retried_on_the_next_open(tmp_path, monkeypatch):
    db = tmp_path / "db"
    Store(db, dim=8).replace_source("/c.md", [rec("/c.md", 0, "договор поставки")])
    table_type = type(Store(db, dim=8)._documents)
    real = table_type.create_index

    def refuse_documents(self, column, *args, **kwargs):
        if self.name == DOCUMENT_TABLE:
            raise RuntimeError("Retryable commit conflict for version 9")
        return real(self, column, *args, **kwargs)

    monkeypatch.setattr(table_type, "create_index", refuse_documents)
    with pytest.warns(RuntimeWarning, match="document"):
        s2 = Store(db, dim=8, fts_language="Russian")
    assert s2.fts_index_language == "English"
    assert (db / FTS_LANGUAGE_FILENAME).read_text(encoding="utf-8").strip() == "English"

    monkeypatch.undo()
    s3 = Store(db, dim=8, fts_language="Russian")
    assert doc_hits(s3, "договоры") == ["/c.md"]


def test_the_bonus_follows_the_formula(tmp_path):
    s = Store(tmp_path / "db", dim=8)
    s.replace_source("/a.md", [rec("/a.md", 0, "alpha", V(0.0))])
    got = s.search("alpha", V(0.0), top_k=1, hybrid_weight=0.6, document_weight=0.5)
    assert got[0].score == pytest.approx((0.4 + 0.6 + 0.5) / 61)


def test_ranks_count_from_one_on_every_side(tmp_path):
    """Keyword side off, so the vector rank and the document rank are all there is: /b
    is nearer, /a holds the term three times in fewer words and ranks first as a
    document."""
    s = Store(tmp_path / "db", dim=8)
    s.replace_source("/a.md", [rec("/a.md", 0, "alpha alpha alpha", V(1.0))])
    s.replace_source("/b.md", [rec("/b.md", 0, "alpha zeta", V(0.0))])
    got = s.search("alpha", V(0.0), top_k=2, hybrid_weight=0.0, document_weight=0.5)
    assert ranked(got) == [("/b.md", 0), ("/a.md", 0)]
    assert got[0].score == pytest.approx(1 / 61 + 0.5 / 62)
    assert got[1].score == pytest.approx(1 / 62 + 0.5 / 61)


def lifting_store(path):
    s = Store(path, dim=8)
    s.replace_source("/b.md", [rec("/b.md", 0, "gamma delta", V(1.0))])
    s.replace_source(
        "/a.md", [rec("/a.md", 0, "alpha one", V(0.0)), rec("/a.md", 1, "beta two", V(0.1))]
    )
    return s


def test_a_document_matching_as_a_whole_lifts_its_chunk(tmp_path):
    s = lifting_store(tmp_path / "db")
    assert s.search("alpha beta", V(1.0), top_k=3, hybrid_weight=0.0)[0].source == "/b.md"
    lifted = s.search("alpha beta", V(1.0), top_k=3, hybrid_weight=0.0, document_weight=0.5)
    assert lifted[0].source == "/a.md"


@pytest.fixture
def mixed(tmp_path):
    s = Store(tmp_path / "mixed", dim=8)
    s.replace_source(
        "/a.md", [rec("/a.md", 0, "alpha one", V(0.0)), rec("/a.md", 1, "beta two", V(0.1))]
    )
    s.replace_source("/b.md", [rec("/b.md", 0, "gamma delta", V(1.0))])
    s.replace_source(
        "/c.md",
        [rec("/c.md", 0, "alpha gamma notes", V(0.6)), rec("/c.md", 1, "epsilon beta", V(0.5))],
    )
    s.update_indices()
    return s


# What `search` returned for `mixed` before the document bonus existed, by hybrid weight:
# (source, chunk_index, score, distance).
BEFORE = {
    0.0: [
        ("/b.md", 0, 1.0, 0.0),
        ("/c.md", 0, 0.7575757616807607, 0.3199999928474426),
        ("/c.md", 1, 0.6666666666666666, 0.5),
        ("/a.md", 1, 0.3816794059846342, 1.619999885559082),
        ("/a.md", 0, 0.3333333333333333, 2.0),
    ],
    0.6: [
        ("/a.md", 0, 0.015989911727616643, 2.0),
        ("/a.md", 1, 0.015927419354838712, 1.619999885559082),
        ("/c.md", 1, 0.015873015873015872, 0.5),
        ("/c.md", 0, 0.015826612903225807, 0.3199999928474426),
        ("/b.md", 0, 0.006557377049180328, 0.0),
    ],
    1.0: [
        ("/a.md", 0, 0.01639344262295082, 2.0),
        ("/a.md", 1, 0.016129032258064516, 1.619999885559082),
        ("/c.md", 1, 0.015873015873015872, 0.5),
        ("/c.md", 0, 0.015625, 0.3199999928474426),
        ("/b.md", 0, 0.0, 0.0),
    ],
}


@pytest.mark.parametrize("weight", sorted(BEFORE))
def test_weight_zero_ranks_as_before(mixed, monkeypatch, weight):
    documents = mixed._documents
    asked = []
    real = documents.search
    monkeypatch.setattr(documents, "search", lambda *a, **k: asked.append(a) or real(*a, **k))
    got = mixed.search("alpha beta", V(1.0), top_k=5, hybrid_weight=weight, document_weight=0.0)
    assert [(r.source, r.chunk_index) for r in got] == [row[:2] for row in BEFORE[weight]]
    assert [r.score for r in got] == pytest.approx([row[2] for row in BEFORE[weight]], rel=1e-12)
    assert [r.distance for r in got] == pytest.approx([row[3] for row in BEFORE[weight]])
    assert asked == []


def test_the_fixed_corpus_is_one_the_bonus_reorders(mixed):
    """Guard for the test above: a corpus the bonus leaves alone would prove nothing."""
    lifted = mixed.search("alpha beta", V(1.0), top_k=5, hybrid_weight=0.0, document_weight=0.5)
    assert ranked(lifted) != [row[:2] for row in BEFORE[0.0]]


def scoped_store(path):
    """/out/y.md outranks /in/x.md as a document and as a chunk on the keyword side."""
    s = Store(path, dim=8)
    s.replace_source("/in/x.md", [rec("/in/x.md", 0, "alpha zeta", V(0.0))])
    s.replace_source("/out/y.md", [rec("/out/y.md", 0, "alpha alpha alpha", V(1.0))])
    return s


def test_the_bonus_respects_scope(tmp_path):
    s = scoped_store(tmp_path / "db")
    everywhere = s.search("alpha", V(0.0), top_k=2, hybrid_weight=0.6, document_weight=0.5)
    x = next(r for r in everywhere if r.source == "/in/x.md")
    assert x.score == pytest.approx(0.4 / 61 + 0.6 / 62 + 0.5 / 62)

    scoped = s.search(
        "alpha", V(0.0), top_k=2, hybrid_weight=0.6, document_weight=0.5, scopes=("/in",)
    )
    assert ranked(scoped) == [("/in/x.md", 0)]
    assert scoped[0].score == pytest.approx((0.4 + 0.6 + 0.5) / 61)


def test_the_bonus_never_adds_a_chunk_the_filter_excluded(tmp_path):
    s = scoped_store(tmp_path / "db")
    s.set_document_labels(
        "/in/x.md",
        labels_json="",
        label_keys="|type=spec|",
        label_hash="h",
        doc_date="",
        doc_date_source="",
    )
    got = s.search(
        "alpha",
        V(1.0),
        top_k=5,
        hybrid_weight=0.6,
        document_weight=1.0,
        filter_sql="label_keys LIKE '%|type=spec|%'",
    )
    assert ranked(got) == [("/in/x.md", 0)]
    # The label filter does not narrow the documents ranking: /out/y.md still ranks first.
    assert got[0].score == pytest.approx(0.4 / 61 + 0.6 / 61 + 1.0 / 62)


def test_backfill_fills_rows_for_chunks_written_without_them(tmp_path):
    s = Store(tmp_path / "db", dim=8)
    s._table.add([asdict(r) for r in SECTIONED])
    assert s.undocumented_sources() == ["/a.md"]

    events = []
    assert s.backfill_documents(on_event=events.append) == (1, [])
    assert events == ["documented /a.md"]
    assert s.undocumented_sources() == []
    assert rows(s)["/a.md"] == [s._document_row("/a.md", SECTIONED)]
    assert s.backfill_documents() == (0, [])


def test_a_row_of_other_chunks_is_rebuilt_by_the_backfill(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "current text")])
    stale = {**rows(store)["/a.md"][0], "ingested_at": LATER, "text": "text it no longer holds"}
    store._db.open_table(DOCUMENT_TABLE).merge_insert("source").when_matched_update_all().execute(
        [stale]
    )
    assert store.undocumented_sources() == ["/a.md"]
    assert store.backfill_documents() == (1, [])
    assert [r["text"] for r in rows(store)["/a.md"]] == ["current text"]


def test_a_backfill_stays_inside_its_scope(tmp_path):
    s = Store(tmp_path / "db", dim=8)
    s._table.add([asdict(rec("/a/x.md", 0, "x")), asdict(rec("/b/y.md", 0, "y"))])
    assert s.backfill_documents(scopes=("/a",)) == (1, [])
    assert s.undocumented_sources() == ["/b/y.md"]


def test_a_source_stored_twice_by_racing_first_ingests_is_left_to_the_sync(tmp_path):
    """Its chunks carry two stamps, so no text joined from them is the document's; the
    sync's diff re-reads it, and that ingest writes the row."""
    s = Store(tmp_path / "db", dim=8)
    first = [rec("/a.md", 0, "alpha")]
    second = [rec("/a.md", 0, "alpha", stamp=LATER)]
    s._table.add([asdict(r) for r in first + second])
    assert s.undocumented_sources() == ["/a.md"]
    assert s.backfill_documents() == (0, [])
    assert rows(s) == {}
    s.replace_source("/a.md", [rec("/a.md", 0, "alpha", stamp="2026-11-21T00:00:00+00:00")])
    assert s.undocumented_sources() == []


def test_two_racing_ingests_leave_the_row_of_the_chunks_that_won(store, monkeypatch):
    """Ingest A writes its chunks; ingest B writes chunks and row; A's row lands last.
    Its chunks are gone, so it must not replace B's."""
    a = [rec("/x.md", 0, "minutes of the meeting")]
    b = [rec("/x.md", 0, "the regulation", stamp=LATER)]
    real = store._put_documents

    def b_lands_in_between(entries):
        monkeypatch.setattr(store, "_put_documents", real)
        store.replace_source("/x.md", b)
        return real(entries)

    monkeypatch.setattr(store, "_put_documents", b_lands_in_between)
    store.replace_source("/x.md", a)

    assert [r["text"] for r in rows(store)["/x.md"]] == ["the regulation"]
    assert store.undocumented_sources() == []


def test_update_indices_folds_new_document_rows(store):
    """The documents indexes are built on the empty table, as the chunk ones are."""
    store.replace_source("/a.md", [rec("/a.md", 0, "alpha")])
    store.replace_source("/b.md", [rec("/b.md", 0, "beta")])
    documents = store._db.open_table(DOCUMENT_TABLE)
    assert set(unindexed(documents)) == {"text_idx", "title_idx"}
    assert set(unindexed(documents).values()) != {0}
    assert store.update_indices() is None
    assert set(unindexed(documents).values()) == {0}


def test_update_indices_folds_document_rows_when_the_chunks_are_current(tmp_path):
    s = Store(tmp_path / "db", dim=8)
    s._table.add([asdict(rec("/a.md", 0, "alpha"))])
    assert s.update_indices() is None
    assert set(unindexed(s._table).values()) == {0}
    assert s.backfill_documents() == (1, [])
    documents = s._db.open_table(DOCUMENT_TABLE)
    assert set(unindexed(documents).values()) != {0}
    assert s.update_indices() is None
    assert set(unindexed(documents).values()) == {0}


def test_a_document_index_update_failure_is_reported_and_the_chunks_still_fold(store, monkeypatch):
    store.replace_source("/a.md", [rec("/a.md", 0, "alpha")])

    def broken(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(store._documents, "optimize", broken)
    problem = store.update_indices()
    assert problem and "disk full" in problem and "document" in problem
    assert set(unindexed(store._table).values()) == {0}


def test_a_failed_document_write_never_fails_the_chunk_write(store, monkeypatch):
    def conflict(*args, **kwargs):
        raise RuntimeError("Retryable commit conflict for version 7")

    monkeypatch.setattr(store._documents, "merge_insert", conflict)
    with pytest.warns(RuntimeWarning, match="document"):
        store.replace_source("/a.md", [rec("/a.md", 0, "alpha")])
    assert store.get_source("/a.md").chunk_count == 1
    assert store.undocumented_sources() == ["/a.md"]


def test_a_failed_document_delete_never_fails_the_chunk_delete(store, monkeypatch):
    store.replace_source("/a.md", [rec("/a.md", 0, "alpha")])

    def conflict(*args, **kwargs):
        raise RuntimeError("Retryable commit conflict for version 7")

    monkeypatch.setattr(store._documents, "delete", conflict)
    with pytest.warns(RuntimeWarning, match="document"):
        assert store.delete_source("/a.md") == 1
    assert store.get_source("/a.md") is None


def test_a_failing_document_query_degrades_to_no_bonus(store, monkeypatch):
    store.replace_source("/b.md", [rec("/b.md", 0, "gamma delta", V(1.0))])
    store.replace_source(
        "/a.md", [rec("/a.md", 0, "alpha one", V(0.0)), rec("/a.md", 1, "beta two", V(0.1))]
    )
    calls = []

    def failing(*args, **kwargs):
        calls.append(kwargs.get("query_type"))
        raise RuntimeError("FTS index unavailable")

    without = store.search("alpha beta", V(1.0), top_k=3, hybrid_weight=0.6)
    monkeypatch.setattr(store._documents, "search", failing)
    with pytest.warns(RuntimeWarning, match="FTS index unavailable") as warned:
        got = store.search("alpha beta", V(1.0), top_k=3, hybrid_weight=0.6, document_weight=0.5)
        store.search("alpha beta", V(1.0), top_k=3, hybrid_weight=0.6, document_weight=0.5)
    assert calls == ["fts", "fts"]
    assert len([w for w in warned if "FTS index unavailable" in str(w.message)]) == 1
    assert ranked(got) == ranked(without)
    assert [r.score for r in got] == pytest.approx([r.score for r in without])


def test_an_index_without_the_table_gains_it_on_open(tmp_path):
    db = tmp_path / "db"
    s1 = Store(db, dim=8)
    s1.replace_source("/a.md", [rec("/a.md", 0, "alpha")])
    s1._db.drop_table(DOCUMENT_TABLE)

    s2 = Store(db, dim=8)
    documents = s2._db.open_table(DOCUMENT_TABLE)
    assert {c for i in documents.list_indices() for c in i.columns} == {"text", "title"}
    assert s2.undocumented_sources() == ["/a.md"]
    assert s2.backfill_documents() == (1, [])


def test_a_table_the_open_could_not_create_is_created_with_its_indexes_by_a_write(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Store, "_open_documents", lambda self, db_path: None)
    s = Store(tmp_path / "db", dim=8, fts_language="Russian")
    monkeypatch.undo()
    with pytest.raises(ValueError, match="was not found"):
        s._db.open_table(DOCUMENT_TABLE)
    s.replace_source("/c.md", [rec("/c.md", 0, "договор поставки")])
    documents = s._db.open_table(DOCUMENT_TABLE)
    assert {c for i in documents.list_indices() for c in i.columns} == {"text", "title"}
    assert doc_hits(s, "договоры") == ["/c.md"]


def test_a_row_whose_chunks_are_gone_is_deleted_by_the_backfill(store):
    """What a failed `_delete_documents` leaves: a row that would take a document rank."""
    store.replace_source("/a.md", [rec("/a.md", 0, "alpha")])
    store.replace_source("/b.md", [rec("/b.md", 0, "beta")])
    store._table.delete("source = '/a.md'")
    events = []
    assert store.backfill_documents(on_event=events.append) == (0, [])
    assert list(rows(store)) == ["/b.md"]
    assert events == ["removed the document row of /a.md"]


def test_an_orphan_row_outside_the_scope_is_kept(store):
    store.replace_source("/a/x.md", [rec("/a/x.md", 0, "x")])
    store.replace_source("/b/y.md", [rec("/b/y.md", 0, "y")])
    store._table.delete("source IN ('/a/x.md', '/b/y.md')")
    assert store.backfill_documents(scopes=("/a",)) == (0, [])
    assert list(rows(store)) == ["/b/y.md"]


def test_a_row_written_after_the_orphan_was_seen_is_kept(store, monkeypatch):
    """A re-ingest of the path between the backfill's reads and its delete."""
    store.replace_source("/a.md", [rec("/a.md", 0, "old text")])
    store._table.delete("source = '/a.md'")
    real = store._document_gaps

    def reingest_after_reading():
        gaps = real()
        store.replace_source("/a.md", [rec("/a.md", 0, "new text", stamp=LATER)])
        return gaps

    monkeypatch.setattr(store, "_document_gaps", reingest_after_reading)
    store.backfill_documents()
    assert [(r["text"], r["ingested_at"]) for r in rows(store)["/a.md"]] == [("new text", LATER)]


def test_an_orphan_delete_failure_is_reported_and_the_rows_still_written(store, monkeypatch):
    store.replace_source("/a.md", [rec("/a.md", 0, "alpha")])
    store._table.delete("source = '/a.md'")
    store._table.add([asdict(rec("/b.md", 0, "beta"))])

    def conflict(*args, **kwargs):
        raise RuntimeError("Retryable commit conflict for version 7")

    monkeypatch.setattr(store._documents, "delete", conflict)
    written, errors = store.backfill_documents()
    assert written == 1 and sorted(rows(store)) == ["/a.md", "/b.md"]
    [error] = errors
    assert error["error"].startswith("documents: ") and "commit conflict" in error["error"]


def _refusing_document_indexes(monkeypatch, columns):
    real = LanceTable.create_index

    def refuse(self, column, *args, **kwargs):
        if self.name == DOCUMENT_TABLE and column in columns:
            raise RuntimeError("Permission denied")
        return real(self, column, *args, **kwargs)

    monkeypatch.setattr(LanceTable, "create_index", refuse)


def test_a_document_index_that_could_not_be_created_is_in_the_status(tmp_path, monkeypatch):
    _refusing_document_indexes(monkeypatch, {"title"})
    with pytest.warns(RuntimeWarning, match="documents table"):
        s = Store(tmp_path / "db", dim=8)
    assert (s.missing_document_fts_indices, s.missing_fts_indices) == (("title",), ())
    assert "'title'" in fts_status(s)["documentFtsWarning"]

    monkeypatch.undo()
    reopened = Store(tmp_path / "db", dim=8)
    assert reopened.missing_document_fts_indices == ()
    assert "documentFtsWarning" not in fts_status(reopened)


def test_a_document_index_the_first_write_could_not_create_is_in_the_status(tmp_path, monkeypatch):
    monkeypatch.setattr(Store, "_open_documents", lambda self, db_path: None)
    s = Store(tmp_path / "db", dim=8)
    assert "documentFtsWarning" not in fts_status(s)
    _refusing_document_indexes(monkeypatch, {"text", "title"})
    with pytest.warns(RuntimeWarning, match="documents table"):
        s.replace_source("/a.md", [rec("/a.md", 0, "alpha")])
    assert list(rows(s)) == ["/a.md"]
    assert s.missing_document_fts_indices == ("text", "title")
    assert "'text', 'title'" in fts_status(s)["documentFtsWarning"]


def _chmod_tree(root, dir_mode: int, file_mode: int) -> None:
    for path, dirs, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(os.path.join(path, name), file_mode)
        for name in dirs:
            os.chmod(os.path.join(path, name), dir_mode)
    os.chmod(root, dir_mode)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anyway")
def test_a_read_only_index_without_the_table_opens_and_searches(tmp_path):
    db = tmp_path / "db"
    s1 = Store(db, dim=8)
    s1.replace_source("/a.md", [rec("/a.md", 0, "alpha")])
    s1._db.drop_table(DOCUMENT_TABLE)
    _chmod_tree(db, 0o555, 0o444)
    try:
        with pytest.warns(RuntimeWarning, match="document"):
            s2 = Store(db, dim=8)
        got = s2.search("alpha", [0.1] * 8, top_k=3, hybrid_weight=0.6, document_weight=0.5)
        assert ranked(got) == [("/a.md", 0)]
        assert got[0].score == pytest.approx((0.4 + 0.6) / 61)
        assert s2.undocumented_sources() == ["/a.md"]
    finally:
        _chmod_tree(db, 0o755, 0o644)


def test_add_sources_writes_chunks_that_the_backfill_documents(store):
    store.add_sources([("a.md", [rec("a.md", 0, "alpha"), rec("a.md", 1, "beta")])])
    assert store.chunk_count() == 2
    assert store.undocumented_sources() == ["a.md"]
    assert store.backfill_documents() == (1, [])
    assert len(rows(store)["a.md"]) == 1


def test_add_sources_is_refused_by_the_write_check(store):
    def refuse():
        raise RuntimeError("replaced")

    store.write_check = refuse
    with pytest.raises(RuntimeError, match="replaced"):
        store.add_sources([("a.md", [rec("a.md", 0, "alpha")])])
    assert store.chunk_count() == 0
