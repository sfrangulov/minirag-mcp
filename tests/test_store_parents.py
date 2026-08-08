"""Parent sections and chunking-scheme migration at the storage layer."""

import lancedb
import pyarrow as pa
import pytest

from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.store import ChunkRecord, Store


def rec(source, i, text, parent, scheme=SCHEME_VERSION):
    return ChunkRecord(
        id=f"{source}#{i}",
        source=source,
        source_type="file",
        title="T",
        chunk_index=i,
        text=text,
        vector=[0.1] * 8,
        file_hash="h",
        mtime=1.0,
        ingested_at="2026-08-07T00:00:00+00:00",
        parent_id=parent,
        scheme_version=scheme,
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db", dim=8)


def test_parent_text_joins_the_siblings_in_chunk_order(store):
    """Stored out of order on purpose: the section is defined by chunk_index, not by
    whatever order the rows come back from the scan."""
    store.replace_source(
        "/a.md",
        [
            rec("/a.md", 2, "Раздел\n\nтретий кусок.", "/a.md#p0"),
            rec("/a.md", 0, "Раздел\n\nпервый кусок.", "/a.md#p0"),
            rec("/a.md", 1, "Раздел\n\nвторой кусок.", "/a.md#p0"),
        ],
    )
    assert store.parent_text("/a.md#p0") == (
        "Раздел\n\nпервый кусок.\n\nвторой кусок.\n\nтретий кусок."
    )


def test_parent_text_does_not_reach_across_sections_or_sources(store):
    store.replace_source(
        "/a.md", [rec("/a.md", 0, "первый", "/a.md#p0"), rec("/a.md", 1, "второй", "/a.md#p1")]
    )
    store.replace_source("/b.md", [rec("/b.md", 0, "чужой", "/b.md#p0")])
    assert store.parent_text("/a.md#p0") == "первый"
    assert store.parent_text("/b.md#p0") == "чужой"
    assert store.parent_text("/missing#p9") is None


def test_search_results_carry_the_enclosing_section(store):
    store.replace_source(
        "/a.md",
        [
            rec("/a.md", 0, "Учет запасов\n\nпервая часть.", "/a.md#p0"),
            rec("/a.md", 1, "Учет запасов\n\nвторая часть.", "/a.md#p0"),
        ],
    )
    hits = store.search("запасов", [0.1] * 8, top_k=1)
    assert len(hits) == 1
    # `text` stays the matched chunk — the thing that was ranked
    assert hits[0].text in ("Учет запасов\n\nпервая часть.", "Учет запасов\n\nвторая часть.")
    assert hits[0].parent_text == "Учет запасов\n\nпервая часть.\n\nвторая часть."
    assert hits[0].parent_id == "/a.md#p0"


def test_parent_lookup_can_be_switched_off(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "запасы", "/a.md#p0")])
    assert store.search("запасы", [0.1] * 8, include_parents=False)[0].parent_text is None


def test_a_fresh_index_reports_no_stale_chunks(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "свежий", "/a.md#p0")])
    assert store.stale_chunk_count() == 0


def test_chunks_from_an_older_scheme_are_counted(store):
    store.replace_source(
        "/a.md",
        [rec("/a.md", 0, "старый", "", scheme=0), rec("/a.md", 1, "новый", "/a.md#p1")],
    )
    assert store.stale_chunk_count() == 1


def _legacy_table(path, rows):
    """An index in the pre-parent schema: no parent_id, no scheme_version."""
    db = lancedb.connect(str(path))
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("title", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 8)),
            pa.field("file_hash", pa.string()),
            pa.field("mtime", pa.float64()),
            pa.field("ingested_at", pa.string()),
        ]
    )
    table = db.create_table("chunks", schema=schema)
    table.add(rows)
    return table


def test_an_index_predating_the_scheme_is_detected_not_assumed(tmp_path):
    """Detected by counting the rows, not by inspecting a version the tool is running:
    an index part-way through a re-sync holds chunks from both schemes."""
    path = tmp_path / "legacy"
    _legacy_table(
        path,
        [
            {
                "id": f"/a.md#{i}",
                "source": "/a.md",
                "source_type": "file",
                "title": "T",
                "chunk_index": i,
                "text": f"старый кусок {i}",
                "vector": [0.1] * 8,
                "file_hash": "h",
                "mtime": 1.0,
                "ingested_at": "2026-01-01T00:00:00+00:00",
            }
            for i in range(3)
        ],
    )
    store = Store(path, dim=8)
    assert store.schema_migrated, "the missing columns are added in place on open"
    assert store.stale_chunk_count() == 3
    assert store.search("кусок", [0.1] * 8)[0].parent_text is None

    # and a re-sync of that source clears it, without touching anything else
    store.replace_source("/a.md", [rec("/a.md", 0, "новый", "/a.md#p0")])
    assert store.stale_chunk_count() == 0
    assert store.parent_text("/a.md#p0") == "новый"


def test_a_legacy_index_does_not_fuse_every_chunk_into_one_section(tmp_path):
    """Migrated rows all carry parent_id '' — grouping on it would return the whole
    index as one 'section'."""
    path = tmp_path / "legacy2"
    _legacy_table(
        path,
        [
            {
                "id": f"/{name}#0",
                "source": f"/{name}",
                "source_type": "file",
                "title": "T",
                "chunk_index": 0,
                "text": f"кусок {name}",
                "vector": [0.1] * 8,
                "file_hash": "h",
                "mtime": 1.0,
                "ingested_at": "2026-01-01T00:00:00+00:00",
            }
            for name in ("a.md", "b.md")
        ],
    )
    store = Store(path, dim=8)
    assert store.parent_texts(["", ""]) == {}
    assert all(hit.parent_text is None for hit in store.search("кусок", [0.1] * 8))
