"""Ingestion pipeline: parse -> chunk -> embed -> store (replace by source)."""

from __future__ import annotations

import contextvars
import hashlib
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from minirag_mcp import ocr
from minirag_mcp.chunker import SCHEME_VERSION, chunk_markdown, estimate_tokens
from minirag_mcp.config import Config
from minirag_mcp.ingest import ignore, pages
from minirag_mcp.ingest import parser as _parser
from minirag_mcp.ingest.ignore import IgnoredFileError
from minirag_mcp.ingest.parser import (
    IMAGE_EXTENSIONS,
    find_title,
    parse_file,
    parse_html,
    supported_extensions,
)
from minirag_mcp.labels import JobReport, Labeler, LabelerSource
from minirag_mcp.labels.filters import LabelFilterError, facet_applies
from minirag_mcp.labels.resolve import DocLabels, root_relative
from minirag_mcp.security import check_url
from minirag_mcp.store import LABEL_COLUMNS, ChunkRecord, SourceInfo, Store

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
    # Why the document was stored without some or all of its labels; None when nothing
    # is missing or the corpus has no classify.toml.
    label_problem: str | None = None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# module-level alias so tests can monkeypatch minirag_mcp.ingest.pipeline.parse_url
def parse_url(url: str, *, allow_private: bool = False):
    return _parser.parse_url(url, allow_private=allow_private)


def _file_time(info: os.stat_result) -> float:
    return getattr(info, "st_birthtime", info.st_mtime)


def _reason(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def _label_columns(doc: DocLabels) -> dict[str, str]:
    return {
        "labels_json": doc.labels_json(),
        "label_keys": doc.label_keys(),
        "label_hash": doc.taxonomy_hash,
        "doc_date": doc.date.value,
        "doc_date_source": doc.date.source,
    }


def _already_stored(info: SourceInfo, columns: dict[str, str]) -> bool:
    """`label_keys` is derived from `labels_json`, so these four decide equality."""
    return (info.labels_json, info.label_hash, info.doc_date, info.doc_date_source) == (
        columns["labels_json"],
        columns["label_hash"],
        columns["doc_date"],
        columns["doc_date_source"],
    )


@dataclass(frozen=True)
class LabelPreview:
    """One file labelled as an ingest would label it; nothing embedded or stored."""

    labels: DocLabels
    distribution: dict[str, dict[str, float]]
    # Every facet of the taxonomy, in its order.
    facets: tuple[str, ...]
    # Why each facet without a label has none.
    unset: dict[str, str]
    # The job's labelling problems, read once it ended.
    problems: list[str]


@dataclass(frozen=True)
class _Job:
    pipeline: Pipeline
    labeler: Labeler | None
    error: str | None
    report: JobReport


# The job this thread is inside. A job opened inside it on the same pipeline joins it
# instead of asking the source again, so one sync labels with one taxonomy even when
# classify.toml is edited underneath it, and its re-label pass compares against the
# hash its own documents were written with.
_ACTIVE: contextvars.ContextVar[_Job | None] = contextvars.ContextVar(
    "minirag_label_job", default=None
)


@contextmanager
def _activated(job: _Job) -> Iterator[None]:
    token = _ACTIVE.set(job)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _named(facets: Sequence[str]) -> str:
    return f"{facets[0]} label" if len(facets) == 1 else f"{', '.join(facets)} labels"


def _label_problem(doc: DocLabels) -> str | None:
    """What a model problem left out of a document's labels, in words.

    A facet with a path rule the model may override keeps its path value meanwhile:
    that label is there, only unconfirmed, and saying it is missing would be wrong.
    """
    if not doc.unanswered:
        return None
    missing = sorted(f for f in doc.unanswered if f not in doc.labels)
    unconfirmed = sorted(f for f in doc.unanswered if f in doc.labels)
    parts = []
    if missing:
        parts.append(
            f"without a {_named(missing)}" if len(missing) == 1 else f"without {_named(missing)}"
        )
    if unconfirmed:
        parts.append(f"with its {_named(unconfirmed)} from the path, unconfirmed by the model")
    return f"stored {' and '.join(parts)}: {doc.reason}. A full sync asks again."


class Pipeline:
    def __init__(self, store: Store, embedder, config: Config, labels: LabelerSource | None = None):
        self.store = store
        self.embedder = embedder
        self.config = config
        self.labels = labels

    @contextmanager
    def job(self) -> Iterator[JobReport]:
        """One unit of work for labelling: the model is loaded at most once inside it.

        The report collects labelling problems for the whole job, which another thread's
        ingest may join; it is the job's, not any one document's.
        """
        current = self._active()
        if current is not None:
            yield current.report
            return
        labeler, error = self.labels.get() if self.labels is not None else (None, None)
        if labeler is not None:
            with labeler.job() as report, _activated(_Job(self, labeler, None, report)):
                yield report
            return
        report = JobReport()
        if error is not None:
            report.add(f"{self.labels.path} is invalid: {error}")
        with _activated(_Job(self, None, error, report)):
            yield report

    def _active(self) -> _Job | None:
        current = _ACTIVE.get()
        return current if current is not None and current.pipeline is self else None

    @property
    def _count_tokens(self):
        """The embedding model's own tokenizer, or the character estimate without one.

        The budget is in tokens because the model's ceiling is in tokens; an embedder
        that cannot count them (a stub, an older wrapper) still has to be usable.
        """
        return getattr(self.embedder, "count_tokens", None) or estimate_tokens

    def _chunk(self, markdown: str, *, source: str, title: str, seed_title: bool):
        """Cut `markdown` into chunks.

        `seed_title` says whether `title` is a real title and may therefore reach the
        chunk text — as chunk 0's opening line, and as part of a transcript window's
        label. Only the caller knows: a data source with no title of its own is titled
        after its source id, and a titleless page after its URL, and neither an id nor
        an address describes a document well enough to embed.
        """
        chunks = chunk_markdown(
            markdown,
            count_tokens=self._count_tokens,
            title=title if seed_title else "",
            budget=self.config.token_budget,
        )
        if not chunks:
            raise EmptyDocumentError(f"No text content extracted from {source}")
        return chunks

    def _label(
        self,
        *,
        source: str,
        source_type: str,
        title: str,
        chunk_texts: Sequence[str],
        file_time: float | None,
    ) -> tuple[dict[str, str], str | None]:
        """The label columns for a document, and why any of them are missing.

        A labelling failure never fails the ingest: the document is stored unlabelled,
        its empty hash brings it back to the next re-label pass, and the reason goes to
        the job's report. The report names a failure once per job, so the document's
        own outcome is returned too: a second document of a job whose model could not
        be made would otherwise look labelled.
        """
        job = self._active()
        if job is None:
            return {}, None
        if job.labeler is None:
            problem = None
            if job.error is not None:
                problem = f"stored without labels: {self.labels.path} is invalid: {job.error}"
            return {}, problem
        try:
            doc = job.labeler.label(
                source=source,
                source_type=source_type,
                title=title,
                chunk_texts=chunk_texts,
                file_time=file_time,
            )
        except Exception as e:
            job.report.add(f"{source}: {_reason(e)}")
            return {}, f"stored without labels: {_reason(e)}"
        return _label_columns(doc), _label_problem(doc)

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
        ocr_engine: str = "",
        file_time: float | None = None,
        page_starts: pages.PageStarts = (),
    ) -> IngestResult:
        """Chunk, embed, label and store, replacing whatever `source` held before."""
        chunks = self._chunk(markdown, source=source, title=title, seed_title=seed_title)
        texts = [c.text for c in chunks]
        chunk_pages = pages.chunk_pages(markdown, page_starts, chunks)
        vectors = self.embedder.embed_documents(texts)
        labels, label_problem = self._label(
            source=source,
            source_type=source_type,
            title=title,
            chunk_texts=texts,
            file_time=file_time,
        )
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
                ocr_engine=ocr_engine,
                scheme_version=SCHEME_VERSION,
                pages=chunk_pages[i],
                **labels,
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self.store.replace_source(source, records)
        return IngestResult(
            source=source, chunk_count=len(records), title=title, label_problem=label_problem
        )

    def _checked_stat(self, path: Path) -> os.stat_result:
        if path.is_dir():
            raise UnsupportedFormatError(
                f"{path} is a folder; this takes one file. To index a folder, sync it: "
                "sync_start with this path, or `minirag-mcp sync <folder>`."
            )
        exts = supported_extensions()
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.suffix.lower() not in exts:
            raise UnsupportedFormatError(f"{path} is an image, which needs OCR; {ocr.INSTALL_HINT}")
        if path.suffix.lower() not in exts:
            raise UnsupportedFormatError(
                f"Unsupported file extension {path.suffix!r}; supported: " + ", ".join(sorted(exts))
            )
        # One stat for both checks: the recorded mtime must describe the same file
        # state whose size was accepted.
        info = path.stat()
        if info.st_size > self.config.max_file_size:
            raise FileTooLargeError(
                f"{path} is {info.st_size} bytes; MAX_FILE_SIZE is {self.config.max_file_size}"
            )
        return info

    def ingest_file(self, path: Path, *, rules_checked: bool = False) -> IngestResult:
        """`rules_checked` is a sync's: its scan already held the file to the rules, and
        a rule file edited mid-sync must not turn files it scanned into failures."""
        # Refused rather than indexed: the next sync would take it out again, silently.
        if not rules_checked and (by := ignore.excluded_by(path, self.config.roots)) is not None:
            raise IgnoredFileError(f"{path} is excluded by {by}; remove its rule to index it")
        info = self._checked_stat(path)
        doc = parse_file(path, self.config)
        with self.job():
            return self._chunk_and_store(
                doc.markdown,
                source=str(path),
                source_type="file",
                title=doc.title,
                # a file's title is always a real one: metadata, a heading, or its name
                seed_title=True,
                file_hash=file_sha256(path),
                mtime=info.st_mtime,
                ocr_engine=doc.ocr_engine,
                file_time=_file_time(info),
                page_starts=doc.page_starts,
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
        with self.job():
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
        with self.job():
            return self._chunk_and_store(
                doc.markdown,
                source=source or url,
                source_type="url",
                title=real_title or doc.title,
                seed_title=real_title is not None,
            )

    def relabel_stale(
        self,
        scopes: tuple[str, ...] = (),
        on_event: Callable[[str], None] | None = None,
    ) -> tuple[int, list[dict]]:
        """Re-label every document in `scopes` whose labels predate the current taxonomy.

        Reads the stored chunk text and writes label columns only: nothing is parsed or
        embedded again. A document's date does not depend on the taxonomy, so a stored
        one is kept; only a document never dated (an empty date source) gets one here.
        """
        with self.job():
            job = self._active()
            if job is None or job.labeler is None:
                return 0, []
            # Checked once: every document would fail the same way, one error each.
            missing = [c for c in LABEL_COLUMNS if c in self.store.missing_columns]
            if missing:
                return 0, [
                    {
                        "source": str(self.store.db_path),
                        "error": f"labels: this index has no {', '.join(missing)} column(s), "
                        "so no document was labelled; reopen it with write access and no "
                        "concurrent writer so they can be added",
                    }
                ]
            labeler = job.labeler
            target = labeler.taxonomy.hash
            relabelled = 0
            errors: list[dict] = []
            for info in self.store.list_sources(scopes):
                if info.label_hash == target:
                    continue
                try:
                    file_time = None
                    if info.source_type == "file":
                        try:
                            file_time = _file_time(os.stat(info.source))
                        except OSError:
                            pass
                    doc = labeler.label(
                        source=info.source,
                        source_type=info.source_type,
                        title=info.title,
                        chunk_texts=[c.text for c in self.store.all_chunks(info.source)],
                        file_time=file_time,
                    )
                    columns = _label_columns(doc)
                    if info.doc_date_source:
                        columns["doc_date"] = info.doc_date
                        columns["doc_date_source"] = info.doc_date_source
                    # A document a model problem keeps unlabelled is back on every sync
                    # until the model answers; rewriting the same values each time would
                    # count it as re-labelled and add a table version per document.
                    if _already_stored(info, columns):
                        continue
                    self.store.set_document_labels(info.source, **columns)
                except Exception as e:  # one document never stops the pass
                    errors.append({"source": info.source, "error": f"labels: {_reason(e)}"})
                    continue
                relabelled += 1
                if on_event:
                    on_event(f"relabelled {info.source}")
            return relabelled, errors

    def preview_labels(self, path: Path) -> LabelPreview:
        """Label one file as an ingest would, with each model answer's full distribution.

        Nothing is embedded or stored.
        """
        with self.job() as report:
            job = self._active()
            if job is None or job.labeler is None:
                if self.labels is None:
                    reason = "this index was opened without a classify.toml"
                elif job is not None and job.error is not None:
                    reason = f"{self.labels.path} is invalid: {job.error}"
                else:
                    reason = f"there is no {self.labels.path}"
                raise LabelFilterError(f"document labels are off: {reason}")
            info = self._checked_stat(path)
            doc = parse_file(path, self.config)
            chunks = self._chunk(doc.markdown, source=str(path), title=doc.title, seed_title=True)
            trace: dict[str, dict[str, float]] = {}
            labeler = job.labeler
            labels = labeler.label(
                source=str(path),
                source_type="file",
                title=doc.title,
                chunk_texts=[c.text for c in chunks],
                file_time=_file_time(info),
                trace=trace,
            )
        rel = root_relative(str(path), labeler.roots)
        unset: dict[str, str] = {}
        for facet in labeler.taxonomy.facets:
            if facet.name in labels.labels:
                continue
            if facet.name in labels.unanswered:
                unset[facet.name] = "the model did not answer"
            elif not facet_applies(facet, rel):
                unset[facet.name] = f"outside within: {', '.join(facet.within)}"
            else:
                unset[facet.name] = "no path rule, default or model answer gave it a value"
        return LabelPreview(
            labels=labels,
            distribution=trace,
            facets=tuple(f.name for f in labeler.taxonomy.facets),
            unset=unset,
            problems=list(report.problems),
        )
