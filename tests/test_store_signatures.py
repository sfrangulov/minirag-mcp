"""Look-alike signatures as the store keeps them: written, moved, deleted and backfilled
alongside the chunks they were computed from."""

import os
from dataclasses import asdict, replace

import numpy as np
import pytest

from lookalike_texts import MINUTES, REGULATION_V1, REGULATION_V2, SPEC_WAREHOUSE
from minirag_mcp.chunker import join_document
from minirag_mcp.lookalike import signature
from minirag_mcp.store import SIGNATURE_TABLE, ChunkRecord, Store


def doc(source, text, *, stamp="2026-10-23T00:00:00+00:00", doc_date="", per_chunk=60):
    """`text` cut into chunks the way the chunker leaves them: one section, its heading
    repeated at the head of every unit."""
    lines = text.strip().split("\n")
    head, body = lines[0], " ".join(lines[1:]).split()
    pieces = [" ".join(body[i : i + per_chunk]) for i in range(0, len(body), per_chunk)]
    return [
        ChunkRecord(
            id=f"{source}#{i}",
            source=source,
            source_type="file",
            title="T",
            chunk_index=i,
            text=f"{head}\n\n{piece}",
            vector=[0.1 * (i + 1)] * 8,
            file_hash="h",
            mtime=1.0,
            ingested_at=stamp,
            parent_id=f"{source}#p0",
            doc_date=doc_date,
            doc_date_source="header" if doc_date else "",
        )
        for i, piece in enumerate(pieces)
    ]


def stored_rows(store, source):
    rows = store._signatures.search().where(f"source = '{source}'").limit(100).to_list()
    return rows


def near(store, source):
    return [(n.source, n.overlap) for n in store.look_alikes([source]).get(source, [])]


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db", dim=8)


def test_a_written_source_is_signed_from_its_reconstructed_text(store):
    """The text `read_file` returns, with each unit's repeated heading emitted once."""
    records = doc("/a.md", REGULATION_V1)
    assert len(records) > 2
    store.replace_source("/a.md", records)
    [row] = stored_rows(store, "/a.md")
    expected = signature(join_document(records))
    assert np.array_equal(np.asarray(row["signature"], dtype=np.uint32), expected.values)
    assert row["shingles"] == expected.shingles
    verbatim = signature("\n".join(r.text for r in records))
    assert not np.array_equal(expected.values, verbatim.values)


def test_two_versions_name_each_other_and_an_unrelated_document_names_neither(store):
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/v2.md", doc("/v2.md", REGULATION_V2, doc_date="2026-03-01"))
    store.replace_source("/minutes.md", doc("/minutes.md", MINUTES))
    found = store.look_alikes(["/v1.md", "/v2.md", "/minutes.md"])
    [v2] = found["/v1.md"]
    [v1] = found["/v2.md"]
    assert (v2.source, v1.source) == ("/v2.md", "/v1.md")
    assert v1.overlap == v2.overlap and v1.overlap >= 0.5
    assert (v2.doc_date, v2.doc_date_source) == ("2026-03-01", "header")
    assert (v1.doc_date, v1.doc_date_source) == ("", "")
    assert "/minutes.md" not in found


def test_deleting_a_source_deletes_its_signature(store):
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/v2.md", doc("/v2.md", REGULATION_V2))
    assert store.delete_source("/v2.md") > 0
    assert stored_rows(store, "/v2.md") == []
    assert near(store, "/v1.md") == []


def test_replacing_a_source_with_nothing_deletes_its_signature(store):
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/v1.md", [])
    assert stored_rows(store, "/v1.md") == []


def test_moving_a_source_moves_its_signature(store):
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/old/v2.md", doc("/old/v2.md", REGULATION_V2))
    assert store.move_source("/old/v2.md", "/new/v2.md", mtime=2.0) > 0
    assert stored_rows(store, "/old/v2.md") == []
    assert [s for s, _ in near(store, "/v1.md")] == ["/new/v2.md"]
    assert store.unsigned_sources() == []


def test_a_re_ingest_replaces_the_signature(store):
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/x.md", doc("/x.md", REGULATION_V2))
    store.replace_source("/x.md", doc("/x.md", MINUTES, stamp="2026-10-24T00:00:00+00:00"))
    assert len(stored_rows(store, "/x.md")) == 1
    assert near(store, "/v1.md") == []


def test_two_racing_ingests_leave_the_signature_of_the_chunks_that_won(store, monkeypatch):
    """Ingest A writes its chunks; ingest B writes chunks and signature; A's signature
    lands last. Its chunks are gone, so it must not replace B's."""
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    a = doc("/x.md", MINUTES, stamp="2026-10-23T10:00:00+00:00")
    b = doc("/x.md", REGULATION_V2, stamp="2026-10-23T10:00:01+00:00")
    real = store._put_signatures

    def b_lands_in_between(entries):
        monkeypatch.setattr(store, "_put_signatures", real)
        store.replace_source("/x.md", b)
        return real(entries)

    monkeypatch.setattr(store, "_put_signatures", b_lands_in_between)
    store.replace_source("/x.md", a)

    assert {c.text for c in store.all_chunks("/x.md")} == {r.text for r in b}
    assert [s for s, _ in near(store, "/v1.md")] == ["/x.md"]
    assert store.unsigned_sources() == []


def test_a_signature_of_other_chunks_is_never_used(store):
    """What a race can still leave behind: a signature computed from text the source no
    longer holds. It must read as missing, not as a look-alike."""
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/x.md", doc("/x.md", MINUTES))
    stale = signature(REGULATION_V2)
    store._signatures.merge_insert("source").when_matched_update_all().execute(
        [
            {
                "source": "/x.md",
                "ingested_at": "2020-01-01T00:00:00+00:00",
                "signature": stale.values.tolist(),
                "shingles": stale.shingles,
            }
        ]
    )
    assert near(store, "/v1.md") == []
    assert store.unsigned_sources() == ["/x.md"]

    assert store.backfill_signatures() == (1, [])
    assert store.unsigned_sources() == []
    assert near(store, "/v1.md") == []


def test_a_signature_written_twice_for_one_source_is_read_once(store):
    """Two first ingests of one file can both insert, as they can with chunks."""
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/v2.md", doc("/v2.md", REGULATION_V2))
    [row] = stored_rows(store, "/v2.md")
    store._signatures.add([row])
    assert len(stored_rows(store, "/v2.md")) == 2
    assert [s for s, _ in near(store, "/v1.md")] == ["/v2.md"]


def test_documents_without_words_are_signed_but_never_look_alike(store):
    """Two empty signatures are equal everywhere, which is not shared text."""
    for source in ("/rule-a.md", "/rule-b.md"):
        store.replace_source(source, [replace(doc(source, MINUTES)[0], text="---\n\n| — |")])
    assert store.unsigned_sources() == []
    assert store.look_alikes(["/rule-a.md", "/rule-b.md"]) == {}


def test_a_source_stored_twice_by_racing_first_ingests_is_left_unsigned(store):
    """Its text is doubled until the sync's diff re-reads it, so no signature of it is
    current; the re-read signs it."""
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    first = doc("/v2.md", REGULATION_V2, stamp="2026-10-23T10:00:00+00:00")
    second = doc("/v2.md", REGULATION_V2, stamp="2026-10-23T10:00:01+00:00")
    store._table.add([asdict(r) for r in first + second])
    assert store.get_source("/v2.md").duplicated

    assert store.backfill_signatures() == (0, [])
    assert store.unsigned_sources() == ["/v2.md"]
    assert near(store, "/v1.md") == []

    store.replace_source("/v2.md", doc("/v2.md", REGULATION_V2, stamp="2026-10-24T00:00:00+00:00"))
    assert store.unsigned_sources() == []
    assert [s for s, _ in near(store, "/v1.md")] == ["/v2.md"]


def _drop_signatures(db_path):
    """An index as a version before look-alikes wrote it."""
    s = Store(db_path, dim=8)
    s.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    s.replace_source("/v2.md", doc("/v2.md", REGULATION_V2))
    s.replace_source("/spec.md", doc("/spec.md", SPEC_WAREHOUSE))
    s._db.drop_table(SIGNATURE_TABLE)


def test_an_index_from_before_signatures_is_backfilled_from_its_stored_text(tmp_path):
    db = tmp_path / "db"
    _drop_signatures(db)
    store = Store(db, dim=8)
    assert store.unsigned_sources() == ["/spec.md", "/v1.md", "/v2.md"]
    assert near(store, "/v1.md") == []

    events = []
    assert store.backfill_signatures(on_event=events.append) == (3, [])

    assert sorted(events) == ["signed /spec.md", "signed /v1.md", "signed /v2.md"]
    assert store.unsigned_sources() == []
    assert [s for s, _ in near(store, "/v1.md")] == ["/v2.md"]
    [row] = stored_rows(store, "/v1.md")
    expected = signature(join_document(store.all_chunks("/v1.md")))
    assert np.array_equal(np.asarray(row["signature"], dtype=np.uint32), expected.values)
    assert store.backfill_signatures() == (0, [])


def test_a_backfill_stays_inside_its_scope(tmp_path):
    store = Store(tmp_path / "db", dim=8)
    store.replace_source("/a/v1.md", doc("/a/v1.md", REGULATION_V1))
    store.replace_source("/b/v2.md", doc("/b/v2.md", REGULATION_V2))
    store._db.drop_table(SIGNATURE_TABLE)
    store = Store(tmp_path / "db", dim=8)
    assert store.backfill_signatures(scopes=("/a",)) == (1, [])
    assert store.unsigned_sources() == ["/b/v2.md"]
    assert store.unsigned_sources(scopes=("/a",)) == []


def test_a_backfill_that_cannot_write_says_so(tmp_path, monkeypatch):
    db = tmp_path / "db"
    _drop_signatures(db)
    store = Store(db, dim=8)

    def conflict(*args, **kwargs):
        raise RuntimeError("Retryable commit conflict for version 7")

    monkeypatch.setattr(store, "_write_signatures", conflict)
    signed, errors = store.backfill_signatures()
    assert signed == 0
    assert errors and all("commit conflict" in e["error"] for e in errors)
    assert errors[0]["error"].startswith("lookAlikes: ")
    assert len(store.unsigned_sources()) == 3


def test_an_ingest_whose_signature_cannot_be_written_still_stores_its_chunks(store, monkeypatch):
    def conflict(*args, **kwargs):
        raise RuntimeError("Retryable commit conflict for version 7")

    monkeypatch.setattr(store, "_write_signatures", conflict)
    with pytest.warns(RuntimeWarning, match="look-alike signature"):
        store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    assert store.get_source("/v1.md").chunk_count > 0
    assert store.unsigned_sources() == ["/v1.md"]


def _chmod_tree(root, dir_mode: int, file_mode: int) -> None:
    for path, dirs, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(os.path.join(path, name), file_mode)
        for name in dirs:
            os.chmod(os.path.join(path, name), dir_mode)
    os.chmod(root, dir_mode)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anyway")
def test_a_read_only_index_from_before_signatures_opens_and_searches(tmp_path):
    db = tmp_path / "db"
    _drop_signatures(db)
    _chmod_tree(db, 0o555, 0o444)
    try:
        with pytest.warns(RuntimeWarning, match="look-alike"):
            store = Store(db, dim=8)
        assert store.chunk_count() > 0
        assert [r.source for r in store.search("закупку", [0.1] * 8, top_k=3)]
        assert store.look_alikes(["/v1.md"]) == {}
        assert store.unsigned_sources() == ["/spec.md", "/v1.md", "/v2.md"]
    finally:
        _chmod_tree(db, 0o755, 0o644)


def test_the_signature_matrix_is_read_once_per_table_version(store, monkeypatch):
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/v2.md", doc("/v2.md", REGULATION_V2))
    reads = []
    real = store._read_signature_index
    monkeypatch.setattr(store, "_read_signature_index", lambda: reads.append(1) or real())

    store.look_alikes(["/v1.md"])
    store.look_alikes(["/v2.md"])
    store.unsigned_sources()
    assert len(reads) == 1

    store.replace_source("/minutes.md", doc("/minutes.md", MINUTES))
    store.look_alikes(["/v1.md"])
    assert len(reads) == 2

    other = Store(store.db_path, dim=8)
    other.delete_source("/v2.md")
    assert store.look_alikes(["/v1.md"]) == {}
    assert len(reads) == 3


def test_relabelling_a_look_alike_shows_its_new_date(store):
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/v2.md", doc("/v2.md", REGULATION_V2))
    assert store.look_alikes(["/v1.md"])["/v1.md"][0].doc_date == ""
    store.set_document_labels(
        "/v2.md",
        labels_json="",
        label_keys="",
        label_hash="h",
        doc_date="2026-05-05",
        doc_date_source="text",
    )
    [v2] = store.look_alikes(["/v1.md"])["/v1.md"]
    assert (v2.doc_date, v2.doc_date_source) == ("2026-05-05", "text")


def test_a_stamp_reused_by_a_rewrite_keeps_the_signature_current(store):
    """A caller that reuses a stored stamp updates the chunks in place."""
    records = doc("/x.md", REGULATION_V2)
    store.replace_source("/v1.md", doc("/v1.md", REGULATION_V1))
    store.replace_source("/x.md", [replace(r, text="unrelated words here") for r in records])
    store.replace_source("/x.md", records)
    assert [s for s, _ in near(store, "/v1.md")] == ["/x.md"]
