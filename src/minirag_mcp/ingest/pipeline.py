"""Ingestion pipeline: parse -> chunk -> embed -> store (replace by source)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from minirag_mcp.chunker.semantic import merge_blocks
from minirag_mcp.chunker.structural import split_markdown
from minirag_mcp.config import Config
from minirag_mcp.ingest import parser as _parser
from minirag_mcp.ingest.parser import (
    SUPPORTED_EXTENSIONS,
    parse_file,
    parse_html,
)
from minirag_mcp.security import check_url_scheme
from minirag_mcp.store import ChunkRecord, Store

MAX_CHUNK_CHARS = 1500
DATA_FORMATS = ("text", "markdown", "html")


class UnsupportedFormatError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


class EmptyDocumentError(Exception):
    pass


@dataclass(frozen=True)
class IngestResult:
    source: str
    chunk_count: int
    title: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# module-level alias so tests can monkeypatch minirag_mcp.ingest.pipeline.parse_url
def parse_url(url: str):
    return _parser.parse_url(url)


class Pipeline:
    def __init__(self, store: Store, embedder, config: Config):
        self.store = store
        self.embedder = embedder
        self.config = config

    def _chunk_and_store(
        self,
        markdown: str,
        *,
        source: str,
        source_type: str,
        title: str,
        file_hash: str = "",
        mtime: float = 0.0,
    ) -> IngestResult:
        blocks = split_markdown(markdown, max_chars=MAX_CHUNK_CHARS)
        texts = merge_blocks(
            blocks,
            self.embedder.embed_documents,
            max_chars=MAX_CHUNK_CHARS,
            min_length=self.config.chunk_min_length,
        )
        if not texts:
            raise EmptyDocumentError(f"No text content extracted from {source}")
        vectors = self.embedder.embed_documents(texts)
        now = datetime.now(UTC).isoformat()
        records = [
            ChunkRecord(
                id=f"{source}#{i}",
                source=source,
                source_type=source_type,
                title=title,
                chunk_index=i,
                text=text,
                vector=vec,
                file_hash=file_hash,
                mtime=mtime,
                ingested_at=now,
            )
            for i, (text, vec) in enumerate(zip(texts, vectors, strict=True))
        ]
        self.store.replace_source(source, records)
        return IngestResult(source=source, chunk_count=len(records), title=title)

    def ingest_file(self, path: Path) -> IngestResult:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file extension {path.suffix!r}; supported: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS))
            )
        size = path.stat().st_size
        if size > self.config.max_file_size:
            raise FileTooLargeError(
                f"{path} is {size} bytes; MAX_FILE_SIZE is {self.config.max_file_size}"
            )
        doc = parse_file(path)
        return self._chunk_and_store(
            doc.markdown,
            source=str(path),
            source_type="file",
            title=doc.title,
            file_hash=file_sha256(path),
            mtime=path.stat().st_mtime,
        )

    def ingest_data(
        self, data: str, source: str, fmt: str = "text", title: str | None = None
    ) -> IngestResult:
        if fmt not in DATA_FORMATS:
            raise UnsupportedFormatError(f"format must be one of {DATA_FORMATS}, got {fmt!r}")
        if fmt == "html":
            doc = parse_html(data, title=title)
            markdown, final_title = doc.markdown, doc.title
        else:
            markdown = data
            from minirag_mcp.ingest.parser import extract_title

            final_title = extract_title(markdown, title, source)
        return self._chunk_and_store(markdown, source=source, source_type="data", title=final_title)

    def ingest_url(
        self, url: str, source: str | None = None, title: str | None = None
    ) -> IngestResult:
        check_url_scheme(url)
        doc = parse_url(url)
        final_title = title.strip() if title and title.strip() else doc.title
        return self._chunk_and_store(
            doc.markdown, source=source or url, source_type="url", title=final_title
        )
