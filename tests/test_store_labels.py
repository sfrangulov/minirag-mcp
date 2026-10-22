from dataclasses import replace

import pytest

from minirag_mcp.store import LABEL_COLUMNS, ChunkRecord, Store
from store_helpers import _older_table, _row


def V(x):  # 8-dim vector pointing "x of the way" between axis 0 and axis 1
    return [1.0 - x, x] + [0.0] * 6


def rec(source, i, text, vec=None, **labels):
    return ChunkRecord(
        id=f"{source}#{i}",
        source=source,
        source_type="file",
        title="T",
        chunk_index=i,
        text=text,
        vector=vec or V(0.5),
        file_hash="h",
        mtime=1.0,
        ingested_at="2026-08-07T00:00:00+00:00",
        **labels,
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db", dim=8)


def test_a_new_index_has_the_label_columns(store):
    assert LABEL_COLUMNS == (
        "labels_json",
        "label_keys",
        "label_hash",
        "doc_date",
        "doc_date_source",
    )
    assert set(LABEL_COLUMNS) <= set(store._table.schema.names)
    assert store.missing_columns == ()


def test_an_older_index_gains_the_label_columns_on_open(tmp_path):
    db_path = tmp_path / "db"
    _older_table(db_path, set(LABEL_COLUMNS), [_row("/a.md", 0, "x")])

    store = Store(db_path, dim=8)

    assert set(LABEL_COLUMNS) <= set(store._table.schema.names)
    assert store.missing_columns == ()
    info = store.list_sources()[0]
    assert info.labels_json == ""
    assert info.doc_date == ""
    assert store.search("x", V(0.5), top_k=1)[0].labels_json == ""


def test_set_document_labels_touches_every_chunk_of_one_source_only(store):
    store.replace_source("/a.md", [rec("/a.md", i, f"a {i}") for i in range(3)])
    store.replace_source("/b.md", [rec("/b.md", 0, "b")])

    updated = store.set_document_labels(
        "/a.md",
        labels_json='{"type": "minutes"}',
        label_keys="|type=minutes|",
        label_hash="h1",
        doc_date="2026-10-12",
        doc_date_source="header",
    )

    assert updated == 3
    rows = store._table.search().limit(100).to_list()
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    assert len(by_source["/a.md"]) == 3
    for r in by_source["/a.md"]:
        assert (r["labels_json"], r["label_keys"], r["label_hash"]) == (
            '{"type": "minutes"}',
            "|type=minutes|",
            "h1",
        )
        assert (r["doc_date"], r["doc_date_source"]) == ("2026-10-12", "header")
    for r in by_source["/b.md"]:
        assert all(r[c] == "" for c in LABEL_COLUMNS)
    assert store.set_document_labels("/nope.md", **dict.fromkeys(LABEL_COLUMNS, "")) == 0


def test_a_label_filter_reaches_the_keyword_side_before_its_window(store):
    # 117 unlabelled documents say "alpha" many times; the 3 labelled ones say it once
    # amid filler, so BM25 ranks them below the 50-row FTS window.
    heavy = "alpha " * 8
    light = "alpha " + "filler words pad this chunk out " * 6
    for n in range(117):
        store.replace_source(f"/heavy{n}.md", [rec(f"/heavy{n}.md", 0, heavy, V(0.0))])
    for n in range(3):
        src = f"/light{n}.md"
        store.replace_source(src, [rec(src, 0, light, V(1.0))])
        store.set_document_labels(
            src,
            labels_json="{}",
            label_keys="|type=minutes|",
            label_hash="h",
            doc_date="",
            doc_date_source="",
        )

    got = store.search(
        "alpha",
        V(0.0),
        top_k=3,
        hybrid_weight=1.0,
        include_parents=False,
        filter_sql="contains(label_keys, '|type=minutes|')",
    )

    assert sorted(r.source for r in got) == ["/light0.md", "/light1.md", "/light2.md"]
    # With hybrid_weight=1.0 a vector-only hit scores 0.0: only the keyword side
    # finding these rows lifts the score, so a post-filtered keyword side fails here.
    assert all(r.score > 0 for r in got)


def test_scope_and_filter_are_both_applied(store):
    for src in ("/x/a.md", "/x/b.md", "/y/c.md"):
        store.replace_source(src, [rec(src, 0, "alpha doc")])
    for src in ("/x/a.md", "/y/c.md"):
        store.set_document_labels(
            src,
            labels_json="{}",
            label_keys="|type=minutes|",
            label_hash="h",
            doc_date="",
            doc_date_source="",
        )
    flt = "contains(label_keys, '|type=minutes|')"

    found = store.search(
        "alpha", V(0.5), scopes=("/x", "/z"), filter_sql=flt, include_parents=False
    )
    assert {r.source for r in found} == {"/x/a.md"}
    listed = store.list_sources(scopes=("/x", "/z"), filter_sql=flt)
    assert [s.source for s in listed] == ["/x/a.md"]
    assert [s.source for s in store.list_sources(filter_sql=flt)] == ["/x/a.md", "/y/c.md"]


def test_results_and_sources_carry_labels_and_date(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "alpha")])
    store.set_document_labels(
        "/a.md",
        labels_json='{"type": "minutes"}',
        label_keys="|type=minutes|",
        label_hash="h1",
        doc_date="2026-10-12",
        doc_date_source="name",
    )

    hit = store.search("alpha", V(0.5), include_parents=False)[0]
    assert (hit.labels_json, hit.doc_date, hit.doc_date_source) == (
        '{"type": "minutes"}',
        "2026-10-12",
        "name",
    )
    for info in (store.list_sources()[0], store.get_source("/a.md")):
        assert (info.labels_json, info.label_hash, info.doc_date, info.doc_date_source) == (
            '{"type": "minutes"}',
            "h1",
            "2026-10-12",
            "name",
        )


def test_a_moved_source_keeps_its_vectors_and_loses_its_label_hash(store):
    store.replace_source("/old.md", [replace(rec("/old.md", 0, "alpha", V(0.3)))])
    store.set_document_labels(
        "/old.md",
        labels_json='{"type": "minutes"}',
        label_keys="|type=minutes|",
        label_hash="h1",
        doc_date="2026-10-12",
        doc_date_source="name",
    )

    assert store.move_source("/old.md", "/new.md", mtime=2.0) == 1

    info = store.get_source("/new.md")
    assert info.label_hash == ""
    assert info.labels_json == '{"type": "minutes"}'
    row = store._table.search().where("source = '/new.md'").to_list()[0]
    assert row["vector"] == pytest.approx(V(0.3))
