"""The indexes one process serves, which of them a connection may use, and every refusal.

Each index is one `IndexHandle`, which opens it on first use and follows its recorded
scheme and a swapped generation; the registry builds nothing of an index itself and keeps
no `OpenIndex`, so every call reaches the index through its handle. The environment mode
is a registry of one index named `default`.

A file registry follows its file. A reload keeps the handle, and with it the open index, of
every index whose configuration is unchanged; a file that does not load leaves the previous
indexes serving. The handle of an index a reload drops or changes is closed by a later
`refresh` once no sync of it runs, so the reload itself, run on the event loop by SIGHUP,
never waits on an index.
"""

from __future__ import annotations

import hashlib
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from minirag_mcp.config import Config, ConfigError
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


@dataclass(frozen=True)
class _Source:
    """The file a registry loads, and what each of its indexes is built with."""

    path: Path
    env: Mapping[str, str]
    embedder_factory: RegistryEmbedderFactory | None
    model_factory: ModelFactory | None

    def entries(
        self, server: ServerConfig, previous: Mapping[str, IndexEntry]
    ) -> dict[str, IndexEntry]:
        """The entries `server` configures. An index whose Config is unchanged keeps its
        handle, and its entry too unless its description changed."""
        entries = {}
        for name, config in server.indexes.items():
            description = server.descriptions[name]
            kept = previous.get(name)
            if kept is None or kept.config != config:
                entries[name] = _entry(
                    name, config, description, self.embedder_factory, self.model_factory
                )
            elif kept.description != description:
                entries[name] = replace(kept, description=description)
            else:
                entries[name] = kept
        return entries


def _digest(path: Path) -> bytes | None:
    """The sha256 of the file's bytes; None while it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).digest()
    except OSError:
        return None


def _restart_only(server: ServerConfig) -> tuple[str, int, Path]:
    """What a running server took at start, which a reload cannot change."""
    return server.host, server.port, server.token_file


class IndexRegistry:
    """Every configured index by name, opened lazily, and the refusals of a call that names
    one it cannot use."""

    def __init__(
        self,
        entries: Mapping[str, IndexEntry],
        *,
        server: ServerConfig | None = None,
        config_error: str | None = None,
        source: _Source | None = None,
        digest: bytes | None = None,
    ) -> None:
        self.path = source.path if source is not None else None
        self.server = server
        self.config_error = config_error
        # Guards the entries and the retired handles only. Opening an index (seconds when a
        # model loads) never holds it: a reload swaps the entries on the event loop's thread.
        # Readers iterate the dict `_snapshot` returns after letting go of it, so a reload
        # replaces the dict and never changes one in place.
        self._lock = threading.Lock()
        self._entries: dict[str, IndexEntry] = dict(entries)
        self._source = source
        # Serializes reloads; held while the file is parsed, never while an index opens.
        self._reload_lock = threading.Lock()
        self._digest = digest
        # The last error a refresh told stderr, so a file that stays broken is told once.
        self._told: str | None = None
        # Handles a reload let go; a refresh closes each once no sync of it runs.
        self._retired: list[IndexHandle] = []
        self._at_start = _restart_only(server) if server is not None else None

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
        source = _Source(path, env, embedder_factory, model_factory)
        # Before the parse: an edit landing in between is then loaded by the next refresh.
        digest = _digest(path)
        server = load_server_config(path, env)
        return cls(source.entries(server, {}), server=server, source=source, digest=digest)

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

    def refresh(self) -> None:
        """Reload the file when its bytes changed since the last load, an unreadable file
        counting as changed, and close the retired handles whose syncs have ended. Never
        raises; a no-op in the environment mode."""
        if self._source is None:
            return
        self._close_retired()
        digest = _digest(self._source.path)
        if digest is not None and digest == self._digest:
            return
        with self._reload_lock:
            # Another refresh may have loaded these bytes while this one waited.
            digest = _digest(self._source.path)
            if digest is not None and digest == self._digest:
                return
            self._load(digest, asked=False)

    def reload(self) -> None:
        """Read the file again, changed or not: what SIGHUP asks for. Never raises, and
        closes nothing; a no-op in the environment mode."""
        if self._source is None:
            return
        with self._reload_lock:
            self._load(_digest(self._source.path), asked=True)

    def _load(self, digest: bytes | None, *, asked: bool) -> None:
        """Load the file whose bytes hash to `digest`. When it does not load, the entries
        are kept and stderr is told why: on every reload `asked` for, and on a refresh only
        when the error differs from the last one told. The caller holds the reload lock.

        `digest` is stored last: a refresh that finds it stored skips the reload lock, so
        it must not find it before this load has taken effect."""
        source = self._source
        try:
            server = load_server_config(source.path, source.env)
            entries = source.entries(server, self._entries)
        except Exception as e:
            error = str(e) if isinstance(e, (ConfigError, OSError)) else f"{type(e).__name__}: {e}"
            self.config_error = error
            self._digest = digest
            if asked or error != self._told:
                self._told = error
                print(
                    f"minirag-mcp: could not reload {source.path}, still serving "
                    f"{', '.join(self._entries)}: {error}",
                    file=sys.stderr,
                )
            return
        retired = [
            old.handle
            for name, old in self._entries.items()
            if name not in entries or entries[name].handle is not old.handle
        ]
        with self._lock:
            self._entries = entries
            self._retired.extend(retired)
        self.server = server
        self.config_error = None
        self._told = None
        self._digest = digest
        restart = (
            "; host/port/token_file changes apply after a restart"
            if _restart_only(server) != self._at_start
            else ""
        )
        print(
            f"minirag-mcp: reloaded {source.path}: {', '.join(entries)}{restart}", file=sys.stderr
        )

    def _close_retired(self) -> None:
        """Close each retired handle whose sync has ended. Closing during a sync would make
        the job open the index again, and a close waits for an open in progress."""
        with self._lock:
            retired, self._retired = self._retired, []
            ended = []
            for handle in retired:
                (self._retired if handle.sync_running else ended).append(handle)
        for handle in ended:
            handle.close()

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
