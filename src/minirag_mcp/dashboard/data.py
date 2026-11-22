"""What the dashboard shows of each index, computed from the store's source metadata.

Never from the disk: a folder is a prefix of indexed source paths, so a path the page is
handed is compared with strings and never opened. The statistics of an index are computed
once per version of its table and kept until the table, its last-sync report or its
configuration changes.
"""

from __future__ import annotations

import os
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from minirag_mcp.identity import IndexState
from minirag_mcp.labels.resolve import decode_labels
from minirag_mcp.registry import IndexEntry, IndexRegistry
from minirag_mcp.results import labels_status
from minirag_mcp.scope import SEP, is_under
from minirag_mcp.store import SourceInfo
from minirag_mcp.sync_report import REPORT_FILE, read_report

ROOT_FILES = "(files at the root)"
FOLDER_FILES = "(files in this folder)"


@dataclass(frozen=True)
class FolderStats:
    name: str
    path: str
    documents: int
    chunks: int
    children: tuple[FolderStats, ...] = ()


@dataclass(frozen=True)
class RootStats:
    path: str
    documents: int
    chunks: int
    folders: tuple[FolderStats, ...]


@dataclass(frozen=True)
class IndexView:
    name: str
    description: str
    state: str
    why: str
    documents: int
    chunks: int
    roots: tuple[RootStats, ...]
    facets: dict
    latest: dict | None
    coverage: dict | None
    report_problem: str | None
    exclude: tuple[str, ...]
    max_table_rows: int | None
    model: str
    fts_language: str | None
    reranker: str | None


@dataclass(frozen=True)
class FileRow:
    source: str
    title: str
    chunks: int
    labels: dict
    ingested: str | None


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


@dataclass
class _Tally:
    documents: int = 0
    chunks: int = 0
    children: dict[str, _Tally] = field(default_factory=dict)

    def add(self, chunks: int) -> None:
        self.documents += 1
        self.chunks += chunks


def _ranked(folders: list[FolderStats]) -> tuple[FolderStats, ...]:
    return tuple(sorted(folders, key=lambda f: (-f.chunks, f.name)))


def _folders(root: str, sources: list[SourceInfo]) -> tuple[FolderStats, ...]:
    """The two folder levels under `root` that hold `sources`, largest first."""
    base = root.rstrip(SEP)
    top: dict[str, _Tally] = {}
    for info in sources:
        parts = _nfc(info.source)[len(base) :].strip(SEP).split(SEP)[:-1]
        tally = top.setdefault(parts[0] if parts else ROOT_FILES, _Tally())
        tally.add(info.chunk_count)
        if parts:
            child = tally.children.setdefault(
                parts[1] if len(parts) > 1 else FOLDER_FILES, _Tally()
            )
            child.add(info.chunk_count)
    out = []
    for name, tally in top.items():
        path = root if name == ROOT_FILES else f"{base}{SEP}{name}"
        children = _ranked(
            [
                FolderStats(
                    n, path if n == FOLDER_FILES else f"{path}{SEP}{n}", t.documents, t.chunks
                )
                for n, t in tally.children.items()
            ]
        )
        out.append(FolderStats(name, path, tally.documents, tally.chunks, children))
    return _ranked(out)


@dataclass(frozen=True)
class _Stats:
    """Everything of an index that depends only on its table, report and configuration."""

    key: tuple
    sources: tuple[SourceInfo, ...]
    documents: int
    chunks: int
    roots: tuple[RootStats, ...]
    facets: dict
    latest: dict | None
    coverage: dict | None
    report_problem: str | None
    # Keeps the configuration the key's identity names alive, so its id cannot be reused.
    config: object


def _report_mtime(db_path: Path) -> int | None:
    try:
        return os.stat(Path(db_path) / REPORT_FILE).st_mtime_ns
    except OSError:
        return None


class DashboardData:
    def __init__(self, registry: IndexRegistry) -> None:
        self._registry = registry
        self._lock = threading.Lock()
        self._stats: dict[str, _Stats] = {}

    def overview(self) -> list[IndexView]:
        return [self._view(entry) for entry in self._registry.entries]

    def index(self, name: str) -> IndexView | None:
        entry = self._entry(name)
        return None if entry is None else self._view(entry)

    def folder(self, name: str, path: str) -> list[FileRow] | None:
        """The indexed files under `path`; None unless `path` is a root of the index or a
        folder under one that holds at least one of its sources."""
        entry = self._entry(name)
        if entry is None:
            return None
        stats, _ = self._stats_of(entry)
        if stats is None:
            return None
        wanted = _nfc(path).rstrip(SEP)
        if not wanted or ".." in wanted.split(SEP):
            return None
        roots = [_nfc(str(r)).rstrip(SEP) for r in entry.config.roots]
        if not any(is_under(wanted, root) for root in roots):
            return None
        rows = [
            FileRow(
                source=info.source,
                title=info.title,
                chunks=info.chunk_count,
                labels={k: v.value for k, v in decode_labels(info.labels_json).items()},
                ingested=info.ingested_at or None,
            )
            for info in stats.sources
            if is_under(_nfc(info.source), wanted)
        ]
        if not rows and wanted not in roots:
            return None
        return sorted(rows, key=lambda r: r.source)

    def _entry(self, name: str) -> IndexEntry | None:
        return next((e for e in self._registry.entries if e.name == name), None)

    def _view(self, entry: IndexEntry) -> IndexView:
        stats, problem = self._stats_of(entry)
        state, why = entry.state()
        if stats is None:
            state, why = IndexState.UNAVAILABLE, problem or why
        config = entry.config
        endpoint = config.rerank_endpoint
        return IndexView(
            name=entry.name,
            description=entry.description,
            state=state.value,
            why=why,
            documents=0 if stats is None else stats.documents,
            chunks=0 if stats is None else stats.chunks,
            roots=() if stats is None else stats.roots,
            facets={} if stats is None else stats.facets,
            latest=None if stats is None else stats.latest,
            coverage=None if stats is None else stats.coverage,
            report_problem=None if stats is None else stats.report_problem,
            exclude=tuple(config.exclude),
            max_table_rows=config.max_table_rows,
            model=config.model_name,
            fts_language=config.fts_language,
            reranker=None if endpoint is None else endpoint.name,
        )

    def _stats_of(self, entry: IndexEntry) -> tuple[_Stats | None, str]:
        """The statistics of `entry`, or None with why it has none."""
        state, why = entry.state()
        if state is IndexState.UNAVAILABLE:
            return None, why
        try:
            store = entry.runtime().store
            key = (store.table_version(), _report_mtime(entry.config.db_path), id(entry.config))
            with self._lock:
                cached = self._stats.get(entry.name)
                if cached is not None and cached.key == key:
                    return cached, ""
                stats = self._compute(entry, key)
                self._stats[entry.name] = stats
                return stats, ""
        except Exception as e:
            return None, str(e)

    @staticmethod
    def _compute(entry: IndexEntry, key: tuple) -> _Stats:
        opened = entry.runtime()
        sources = opened.store.list_sources()
        roots = [_nfc(str(r)).rstrip(SEP) for r in entry.config.roots]
        under: dict[str, list[SourceInfo]] = {root: [] for root in roots}
        for info in sources:
            owner = max((r for r in roots if is_under(_nfc(info.source), r)), key=len, default=None)
            if owner is not None:
                under[owner].append(info)
        report = read_report(entry.config.db_path)
        problem = report["unreadable"] if report is not None and "unreadable" in report else None
        ok = report if report is not None and problem is None else {}
        return _Stats(
            key=key,
            sources=tuple(sources),
            documents=len(sources),
            chunks=sum(i.chunk_count for i in sources),
            roots=tuple(
                RootStats(
                    root,
                    len(infos),
                    sum(i.chunk_count for i in infos),
                    _folders(root, infos),
                )
                for root, infos in under.items()
            ),
            facets=labels_status(opened.labels, opened.store).get("labels", {}).get("facets", {}),
            latest=ok.get("latest"),
            coverage=ok.get("coverage"),
            report_problem=problem,
            config=entry.config,
        )
