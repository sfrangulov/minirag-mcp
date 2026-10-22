"""Helpers for building index tables as older versions wrote them."""

import lancedb
import pyarrow as pa

from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.store import TABLE, Store


def _current_schema(tmp_path) -> pa.Schema:
    """The create-table schema, read off a real Store rather than copied.

    A hand-written copy of it goes on satisfying these tests after the real schema
    gains a column, which is exactly when a migration test has something new to say.
    """
    return Store(tmp_path / "schema_probe", dim=8)._table.schema


def _row(source, i, text, parent_id=""):
    return {
        "id": f"{source}#{i}",
        "source": source,
        "source_type": "file",
        "title": "T",
        "chunk_index": i,
        "text": text,
        "vector": [0.1] * 8,
        "file_hash": "h",
        "mtime": 1.0,
        "ingested_at": "2026-08-07T00:00:00+00:00",
        "parent_id": parent_id,
        "ocr_engine": "",
        "scheme_version": SCHEME_VERSION,
        "labels_json": "",
        "label_keys": "",
        "label_hash": "",
        "doc_date": "",
        "doc_date_source": "",
    }


def _older_table(db_path, missing: set[str], rows):
    """A `chunks` table as written by a version that predates the `missing` columns."""
    current = _current_schema(db_path.parent / "probe")
    assert all(set(row) == set(current.names) for row in rows), (
        "`_row` no longer writes exactly the columns the store creates"
    )
    db = lancedb.connect(str(db_path))
    schema = pa.schema([f for f in current if f.name not in missing])
    table = db.create_table(TABLE, schema=schema)
    table.add([{k: v for k, v in row.items() if k not in missing} for row in rows])
    return table


def _raise_commit_conflict(*args, **kwargs):
    raise RuntimeError("Retryable commit conflict for version 65")
