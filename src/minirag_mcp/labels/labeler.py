"""Label one document at a time; own the choice model for the length of a job."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from minirag_mcp.labels import model as _model
from minirag_mcp.labels.dates import document_date
from minirag_mcp.labels.model import (
    INSTALL_HINT,
    MAX_EVIDENCE_CHARS,
    ChoiceModel,
    ModelError,
    default_model_factory,
)
from minirag_mcp.labels.resolve import DocLabels, Label, resolve_facet, root_relative
from minirag_mcp.labels.taxonomy import (
    Facet,
    ModelSpec,
    Taxonomy,
    TaxonomyError,
    load_taxonomy,
    taxonomy_path,
)

ModelFactory = Callable[[ModelSpec, Path], ChoiceModel]


@dataclass
class JobReport:
    problems: list[str] = field(default_factory=list)

    def add(self, problem: str) -> None:
        if problem not in self.problems:
            self.problems.append(problem)


def _reason(e: Exception) -> str:
    return str(e) if isinstance(e, ModelError) else f"{type(e).__name__}: {e}"


@dataclass
class _Document:
    source: str
    name: str
    chunk_texts: Sequence[str]
    trace: dict[str, dict[str, float]] | None
    failed: bool = False
    # Facets the model was asked for and could not answer: it could not be made, or this
    # document's call failed.
    unanswered: set[str] = field(default_factory=set)
    reason: str = ""
    _evidence: str | None = field(default=None, init=False, repr=False)

    def evidence(self, file_name_label: str) -> str:
        if self._evidence is None:
            text = " ".join(" ".join(self.chunk_texts).split())
            self._evidence = f"{file_name_label}: {self.name}\n" + text[:MAX_EVIDENCE_CHARS]
        return self._evidence


class Labeler:
    def __init__(
        self,
        taxonomy: Taxonomy,
        roots: Sequence[Path],
        cache_dir: Path,
        model_factory: ModelFactory = default_model_factory,
    ):
        self.taxonomy = taxonomy
        self._roots = tuple(roots)
        self._cache_dir = cache_dir
        self._model_factory = model_factory
        self._hash = taxonomy.hash
        spec = taxonomy.model
        self._uncertain_below = spec.uncertain_below if spec else ModelSpec.uncertain_below
        # Job bookkeeping only, so entering a job never waits behind a model call or a
        # model download in another thread. Taken before `_lock`, never after it.
        self._jobs = threading.Lock()
        self._depth = 0
        self._report = JobReport()
        # llama.cpp contexts are not thread-safe, and a sync and an MCP ingest_file in
        # another thread share one job and so one model: creating, calling and closing
        # it all happen under this lock.
        self._lock = threading.RLock()
        self._model: ChoiceModel | None = None
        self._model_failed = False
        # Why the last attempt to make the model failed. Kept past the job, which retries
        # afresh, so `status` can say what went wrong instead of "on demand".
        self._last_failure: str | None = None

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    @contextmanager
    def job(self) -> Iterator[JobReport]:
        with self._jobs:
            if self._depth == 0:
                self._report = JobReport()
            self._depth += 1
            report = self._report
        try:
            yield report
        finally:
            with self._jobs:
                self._depth -= 1
                if self._depth == 0:
                    with self._lock:
                        self._end_job(report)

    def _end_job(self, report: JobReport) -> None:
        model, self._model = self._model, None
        self._model_failed = False
        if model is None:
            return
        try:
            model.close()
        except Exception as e:
            report.add(f"could not release the classify model: {_reason(e)}")

    def label(
        self,
        *,
        source: str,
        source_type: str,
        title: str,
        chunk_texts: Sequence[str],
        file_time: float | None,
        trace: dict[str, dict[str, float]] | None = None,
    ) -> DocLabels:
        is_file = source_type == "file"
        with self.job() as report:
            doc = _Document(
                source=source,
                name=Path(source).name if is_file else title,
                chunk_texts=chunk_texts,
                trace=trace,
            )
            rel = root_relative(source, self._roots) if is_file else None
            labels: dict[str, Label] = {}
            for facet in self.taxonomy.facets:
                ask = partial(self._ask, report, doc, facet) if facet.needs_model else None
                label = resolve_facet(facet, rel, ask, self._uncertain_below)
                if label is not None:
                    labels[facet.name] = label
        # Line structure matters to the header rules, so this text is not collapsed.
        date = document_date(
            Path(source).stem if is_file else None, "\n\n".join(chunk_texts), file_time
        )
        # A model problem left a facet unanswered: no hash, so the next re-label pass asks
        # again. That includes a path value the model could have overridden, which is
        # stored meanwhile but is not yet the label the taxonomy asks for.
        return DocLabels(
            labels=labels,
            date=date,
            taxonomy_hash="" if doc.unanswered else self._hash,
            unanswered=frozenset(doc.unanswered),
            reason=doc.reason if doc.unanswered else "",
        )

    def _ask(self, report: JobReport, doc: _Document, facet: Facet) -> Mapping[str, float]:
        spec = self.taxonomy.model
        if doc.failed or spec is None:
            doc.unanswered.add(facet.name)
            doc.reason = doc.reason or "classify.toml has no [model]"
            return {}
        evidence = doc.evidence(spec.file_name_label)
        options = [(v.id, v.description) for v in facet.values]
        with self._lock:
            if self._model is None and not self._model_failed:
                try:
                    self._model = self._model_factory(spec, self._cache_dir)
                    self._last_failure = None
                except Exception as e:
                    self._model_failed = True
                    self._last_failure = _reason(e)
                    report.add(self._last_failure)
            if self._model is None:
                doc.unanswered.add(facet.name)
                doc.reason = doc.reason or self._last_failure or ""
                return {}
            try:
                probs = self._model.probabilities(evidence, facet.question, options)
            except Exception as e:
                doc.failed = True
                doc.unanswered.add(facet.name)
                doc.reason = _reason(e)
                report.add(f"{doc.source}: {doc.reason}")
                return {}
        if doc.trace is not None:
            doc.trace[facet.name] = dict(probs)
        return probs

    def model_state(self) -> str:
        if self._model is not None:
            return "loaded"
        return "unavailable" if self.model_problem() else "on demand"

    def model_problem(self) -> str | None:
        """Why the model cannot answer here, or None when it can or is not needed.

        Checked without loading it: the extra, then a local model file, then the last
        load this process tried. A download not yet made is not a problem until it fails.
        """
        if self._model is not None or not any(f.needs_model for f in self.taxonomy.facets):
            return None
        if not _model.available():
            return INSTALL_HINT
        spec = self.taxonomy.model
        if spec is not None and spec.path is not None and not spec.path.is_file():
            return f"the classify model file {spec.path} does not exist"
        return self._last_failure


class LabelerSource:
    """The labeler for a corpus's classify.toml, rebuilt when the file changes.

    A long-lived server must pick up an edit without a restart.
    """

    def __init__(
        self,
        roots: Sequence[Path],
        cache_dir: Path,
        model_factory: ModelFactory = default_model_factory,
    ):
        self.path = taxonomy_path(roots)
        self._roots = tuple(roots)
        self._cache_dir = cache_dir
        self._model_factory = model_factory
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._current: tuple[Labeler | None, str | None] = (None, None)

    def get(self) -> tuple[Labeler | None, str | None]:
        try:
            st = self.path.stat()
        except (FileNotFoundError, NotADirectoryError):
            return None, None
        except OSError as e:
            return None, f"{self.path}: {e}"
        # Stat before reading: an edit landing in between leaves a stale stamp, which
        # only costs one more reload.
        stamp = (st.st_mtime_ns, st.st_size)
        with self._lock:
            if stamp != self._stamp:
                try:
                    taxonomy = load_taxonomy(self.path)
                except TaxonomyError as e:
                    self._current = (None, str(e))
                else:
                    labeler = Labeler(taxonomy, self._roots, self._cache_dir, self._model_factory)
                    self._current = (labeler, None)
                self._stamp = stamp
            return self._current
