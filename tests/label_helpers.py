"""Build index records that carry labels in the real stored format."""

from minirag_mcp.labels.dates import DocDate
from minirag_mcp.labels.resolve import DocLabels, Label
from minirag_mcp.store import ChunkRecord


def L(value, *, second=None, uncertain=False, source="model"):
    return Label(value, 0.5 if uncertain else 1.0, source, second=second, uncertain=uncertain)


def record(
    source,
    labels: dict[str, Label],
    doc_date="",
    doc_date_source="",
    *,
    text="body",
    vector=None,
    mtime=1.0,
):
    doc = DocLabels(labels, DocDate(doc_date, doc_date_source), "h")
    return ChunkRecord(
        id=f"{source}#0",
        source=source,
        source_type="file",
        title="T",
        chunk_index=0,
        text=text,
        vector=vector or [1.0] + [0.0] * 7,
        file_hash="h",
        mtime=mtime,
        ingested_at="2026-10-01T00:00:00+00:00",
        labels_json=doc.labels_json(),
        label_keys=doc.label_keys(),
        label_hash="h" if labels else "",
        doc_date=doc_date,
        doc_date_source=doc_date_source,
    )
