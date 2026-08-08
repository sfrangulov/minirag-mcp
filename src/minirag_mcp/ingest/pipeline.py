"""Ingestion pipeline: parse -> chunk -> embed -> store (replace by source)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from minirag_mcp.chunker import (
    SCHEME_VERSION,
    Chunk,
    chunk_markdown,
    estimate_tokens,
)
from minirag_mcp.config import Config
from minirag_mcp.ingest import parser as _parser
from minirag_mcp.ingest.parser import (
    SUPPORTED_EXTENSIONS,
    find_title,
    parse_file,
    parse_html,
)
from minirag_mcp.security import check_url
from minirag_mcp.store import ChunkRecord, Store

DATA_FORMATS = ("text", "markdown", "html")


def _seed_title(text: str, title: str) -> str:
    """Put the title in front of the first chunk's text, once.

    Keyword search reads the title column directly, but the vector side only ever
    sees chunk text — so without this a filename-derived title would be invisible to
    semantic search. Applied after chunking (it can never move a chunk boundary) and
    before embedding (the vector reflects it). A chunk that already carries the title
    is left alone: that keeps re-ingest idempotent, and it keeps chunk 0 looking like
    its siblings, which is what lets the parent section be rebuilt from them.
    """
    if not title.strip() or title in text:
        return text
    return f"# {title}\n\n{text}"


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
def parse_url(url: str, *, allow_private: bool = False):
    return _parser.parse_url(url, allow_private=allow_private)


class Pipeline:
    def __init__(self, store: Store, embedder, config: Config):
        self.store = store
        self.embedder = embedder
        self.config = config

    @property
    def _count_tokens(self):
        """The embedding model's own tokenizer, or the character estimate without one.

        The budget is in tokens because the model's ceiling is in tokens; an embedder
        that cannot count them (a stub, an older wrapper) still has to be usable.
        """
        return getattr(self.embedder, "count_tokens", None) or estimate_tokens

    def _chunk_and_store(
        self,
        markdown: str,
        *,
        source: str,
        source_type: str,
        title: str,
        seed_title: bool,
        file_hash: str = "",
        mtime: float = 0.0,
    ) -> IngestResult:
        """Chunk, embed and store, replacing whatever `source` held before.

        `seed_title` says whether `title` is a real title and may therefore be written
        into chunk 0's text. Only the caller knows: a data source with no title of its
        own is titled after its source id, and a titleless page after its URL — neither
        belongs in the embedded text.
        """
        chunks = chunk_markdown(
            markdown,
            count_tokens=self._count_tokens,
            title=title if seed_title else "",
            budget=self.config.token_budget,
        )
        if not chunks:
            raise EmptyDocumentError(f"No text content extracted from {source}")
        if seed_title:
            seeded = _seed_title(chunks[0].text, title)
            if seeded != chunks[0].text:
                chunks[0] = Chunk(text=seeded, parent_index=chunks[0].parent_index)
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_documents(texts)
        now = datetime.now(UTC).isoformat()
        records = [
            ChunkRecord(
                id=f"{source}#{i}",
                source=source,
                source_type=source_type,
                title=title,
                chunk_index=i,
                text=chunk.text,
                vector=vec,
                file_hash=file_hash,
                mtime=mtime,
                ingested_at=now,
                # Scoped to the source so a parent is addressable on its own, and so
                # two documents can never share one.
                parent_id=f"{source}#p{chunk.parent_index}",
                scheme_version=SCHEME_VERSION,
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self.store.replace_source(source, records)
        return IngestResult(source=source, chunk_count=len(records), title=title)

    def ingest_file(self, path: Path) -> IngestResult:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file extension {path.suffix!r}; supported: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS))
            )
        # One stat for both checks: the recorded mtime must describe the same file
        # state whose size was accepted.
        info = path.stat()
        if info.st_size > self.config.max_file_size:
            raise FileTooLargeError(
                f"{path} is {info.st_size} bytes; MAX_FILE_SIZE is {self.config.max_file_size}"
            )
        doc = parse_file(path)
        return self._chunk_and_store(
            doc.markdown,
            source=str(path),
            source_type="file",
            title=doc.title,
            # a file's title is always a real one: metadata, a heading, or its name
            seed_title=True,
            file_hash=file_sha256(path),
            mtime=info.st_mtime,
        )

    def ingest_data(
        self, data: str, source: str, fmt: str = "text", title: str | None = None
    ) -> IngestResult:
        if fmt not in DATA_FORMATS:
            raise UnsupportedFormatError(f"format must be one of {DATA_FORMATS}, got {fmt!r}")
        if fmt == "html":
            doc = parse_html(data, title=title)
            markdown, real_title = doc.markdown, doc.title if doc.has_title else None
        else:
            markdown = data
            real_title = find_title(markdown, title)
        # Without a title of its own the source id stands in, and an id is not a title.
        return self._chunk_and_store(
            markdown,
            source=source,
            source_type="data",
            title=real_title or source,
            seed_title=real_title is not None,
        )

    def ingest_url(
        self, url: str, source: str | None = None, title: str | None = None
    ) -> IngestResult:
        # Checked here so a URL the caller supplied is refused before any request is
        # made; the same rule is re-applied per redirect hop inside parse_url, which
        # is the only place that sees where a fetch is redirected to.
        check_url(url, allow_private=self.config.allow_private_urls)
        doc = parse_url(url, allow_private=self.config.allow_private_urls)
        explicit = title.strip() if title and title.strip() else None
        # A titleless page is titled after its URL, which is an address, not a title.
        real_title = explicit or (doc.title if doc.has_title else None)
        return self._chunk_and_store(
            doc.markdown,
            source=source or url,
            source_type="url",
            title=real_title or doc.title,
            seed_title=real_title is not None,
        )
