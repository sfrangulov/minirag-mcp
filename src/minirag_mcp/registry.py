"""The indexes one process serves, which of them a connection may use, and every refusal.

Each index is one `IndexHandle`, which opens it on first use and follows its recorded
scheme and a swapped generation; the registry builds nothing of an index itself and keeps
no `OpenIndex`, so every call reaches the index through its handle. The environment mode
is a registry of one index named `default`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from minirag_mcp.config import Config
from minirag_mcp.embedder import UnknownModelError
from minirag_mcp.handle import IndexHandle, IndexUnavailableError, OpenIndex, RebuildPendingError
from minirag_mcp.identity import EmbeddingScheme, IndexState
from minirag_mcp.indexes import ServerConfig, load_server_config
from minirag_mcp.labels.labeler import ModelFactory

SINGLE_INDEX_NAME = "default"
# Where `http_server` puts a request's `Connection` in its ASGI scope.
CONNECTION_SCOPE_KEY = "minirag.connection"

RegistryEmbedderFactory = Callable[[Config, EmbeddingScheme], object]


class Use(StrEnum):
    """What a call does with the index it opens. Deleting a document is a READ: like a
    search, it is answered while the index waits for a rebuild."""

    READ = "read"
    INGEST = "ingest"
    SYNC = "sync"


@dataclass(frozen=True)
class Connection:
    """What one client may use: its default index, and the others it may name. `allow`
    None reaches every index."""

    default: str | None
    allow: frozenset[str] | None

    def reachable(self, name: str) -> bool:
        return self.allow is None or name == self.default or name in self.allow


UNRESTRICTED = Connection(default=None, allow=None)


class IndexRefused(Exception):
    """A call no index answers; the message is what the user is told."""


def _also(others: tuple[str, ...]) -> str:
    """The reachable indexes a refused call could use instead, as a closing sentence."""
    return f" Other reachable indexes: {', '.join(others)}." if others else ""


class IndexUnavailable(IndexRefused):
    """A configured, reachable index that cannot be opened now; `reason` says why."""

    def __init__(self, name: str, reason: str, others: tuple[str, ...] = ()) -> None:
        self.name = name
        self.reason = reason
        # A reason may be a whole sentence already, such as the refusal of a 1.x index.
        super().__init__(f"Index {name!r} is unavailable: {reason.rstrip('.')}.{_also(others)}")


def _reason(error: Exception) -> str:
    if isinstance(error, IndexUnavailable):
        return error.reason
    if isinstance(error, UnknownModelError):
        return f"{error} — see fastembed's supported models"
    return str(error)


@dataclass(frozen=True, eq=False)
class IndexEntry:
    """One configured index, and the handle that opens it."""

    name: str
    config: Config
    description: str
    handle: IndexHandle

    def runtime(self) -> OpenIndex:
        """The open index; IndexUnavailable while it cannot be opened, tried again on the
        next call."""
        try:
            return self.handle.get()
        except Exception as e:
            raise IndexUnavailable(self.name, _reason(e)) from e

    def state(self) -> tuple[IndexState, str]:
        return self.handle.state()


def _admitted(entry: IndexEntry) -> OpenIndex | None:
    """The open index whose pipeline the handle admitted an ingest into; None when another
    generation was opened between the admission and the look at the open index, twice."""
    for _ in range(2):
        pipeline = entry.handle.ingest_pipeline()
        opened = entry.handle.get()
        if opened.pipeline is pipeline:
            return opened
    return None


def _entry_factory(
    config: Config, factory: RegistryEmbedderFactory
) -> Callable[[EmbeddingScheme], object]:
    def make(scheme: EmbeddingScheme) -> object:
        embedder = factory(config, scheme)
        # The handle checks the width of a recorded scheme only, and this one may be the
        # configured scheme a fresh index is about to be stamped with.
        if embedder.model_name == scheme.model and embedder.dim != scheme.dim:
            raise IndexUnavailableError(
                f"{scheme.model} makes {scheme.dim}-dimensional vectors, but the embedder "
                f"supplied for it makes {embedder.dim}-dimensional ones"
            )
        return embedder

    return make


def _entry(
    name: str,
    config: Config,
    description: str,
    embedder_factory: RegistryEmbedderFactory | None,
    model_factory: ModelFactory | None,
) -> IndexEntry:
    # Without a factory each handle builds and owns its embedders, so each index checks its
    # own recorded probe; indexes of one model still share its loaded model (embedder.py).
    factory = None if embedder_factory is None else _entry_factory(config, embedder_factory)
    handle = IndexHandle(config, embedder_factory=factory, model_factory=model_factory)
    return IndexEntry(name, config, description, handle)


class IndexRegistry:
    """Every configured index by name, opened lazily, and the refusals of a call that names
    one it cannot use."""

    def __init__(
        self,
        entries: Mapping[str, IndexEntry],
        *,
        path: Path | None = None,
        server: ServerConfig | None = None,
        config_error: str | None = None,
    ) -> None:
        self.path = path
        self.server = server
        self.config_error = config_error
        # Guards the entries only. Opening an index (seconds when a model loads) never holds
        # it: a reload swaps the entries on the event loop's thread.
        self._lock = threading.Lock()
        self._entries: dict[str, IndexEntry] = dict(entries)

    @classmethod
    def from_file(
        cls,
        path: Path,
        env: Mapping[str, str],
        *,
        embedder_factory: RegistryEmbedderFactory | None = None,
        model_factory: ModelFactory | None = None,
    ) -> IndexRegistry:
        """The indexes `path` configures; ConfigError when it does not load."""
        server = load_server_config(path, env)
        entries = {
            name: _entry(name, config, server.descriptions[name], embedder_factory, model_factory)
            for name, config in server.indexes.items()
        }
        return cls(entries, path=path, server=server)

    @classmethod
    def single(
        cls,
        config: Config | None,
        *,
        error: str | None = None,
        embedder_factory: RegistryEmbedderFactory | None = None,
        model_factory: ModelFactory | None = None,
    ) -> IndexRegistry:
        """The environment mode: one index named `default`, or none when the configuration
        did not load, with `error` saying why."""
        entries: dict[str, IndexEntry] = {}
        if config is not None and error is None:
            entries[SINGLE_INDEX_NAME] = _entry(
                SINGLE_INDEX_NAME, config, "", embedder_factory, model_factory
            )
        return cls(entries, config_error=error)

    def _snapshot(self) -> dict[str, IndexEntry]:
        with self._lock:
            return self._entries

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._snapshot())

    def reachable(self, connection: Connection) -> tuple[str, ...]:
        return tuple(name for name in self._snapshot() if connection.reachable(name))

    def resolve(self, connection: Connection, requested: str | None) -> IndexEntry:
        """The entry a call names, or its connection's default; IndexRefused naming the
        indexes it may use instead. Nothing is opened."""
        return self._resolve(self._snapshot(), connection, requested)

    def _resolve(
        self, entries: Mapping[str, IndexEntry], connection: Connection, requested: str | None
    ) -> IndexEntry:
        if not entries:
            raise IndexRefused(f"Configuration error: {self.config_error or 'no configuration'}")
        reachable = ", ".join(n for n in entries if connection.reachable(n)) or "none"
        name = requested if requested is not None else connection.default
        if name is None:
            raise IndexRefused(
                "This connection has no default index, so name one with `index`. "
                f"Reachable indexes: {reachable}."
            )
        entry = entries.get(name)
        if entry is None:
            whose = " (this connection's default)" if requested is None else ""
            raise IndexRefused(f"Unknown index {name!r}{whose}. Reachable indexes: {reachable}.")
        if not connection.reachable(name):
            allowed = ",".join(sorted(connection.allow or ()))
            raise IndexRefused(
                f"Index {name!r} is not allowed on this connection (allow={allowed}). "
                f"Reachable indexes: {reachable}."
            )
        return entry

    def open(
        self, connection: Connection, requested: str | None, use: Use
    ) -> tuple[IndexEntry, OpenIndex]:
        """The entry and its open index, for `use`. An ingest is admitted by the handle,
        and the open index returned for it carries the pipeline the handle admitted."""
        entries = self._snapshot()
        entry = self._resolve(entries, connection, requested)
        others = tuple(n for n in entries if n != entry.name and connection.reachable(n))
        try:
            opened = _admitted(entry) if use is Use.INGEST else entry.handle.get()
        except RebuildPendingError as e:
            raise IndexRefused(f"Index {entry.name!r}: {e}{_also(others)}") from e
        except Exception as e:
            raise IndexUnavailable(entry.name, _reason(e), others) from e
        if opened is None:
            raise IndexRefused(
                f"Index {entry.name!r} changed twice while this ingest was being admitted "
                "(another process rebuilt it, or recorded its probe), so nothing was written. "
                f"Retry it.{_also(others)}"
            )
        return entry, opened
