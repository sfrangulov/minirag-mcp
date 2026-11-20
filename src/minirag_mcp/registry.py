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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from minirag_mcp.config import Config, ConfigError
from minirag_mcp.embedder import UnknownModelError
from minirag_mcp.handle import IndexHandle, IndexUnavailableError, OpenIndex, RebuildPendingError
from minirag_mcp.identity import EmbeddingScheme, IdentityError, IndexState, read_identity
from minirag_mcp.indexes import ServerConfig, load_server_config
from minirag_mcp.labels.labeler import ModelFactory
from minirag_mcp.lock import SyncLockBusy
from minirag_mcp.security import OutsideRootsError, resolve_in_roots, within
from minirag_mcp.store import IndexBusyError, IndexOpenError
from minirag_mcp.sync import SyncBusyError, SyncJob

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
    # Set by `only`: a server of its default index alone, which points a call naming another
    # index, or a path under one, at the server of that index.
    alone: bool = False

    @classmethod
    def only(cls, name: str) -> Connection:
        """The connection of a server started for index `name` alone: one entry of an MCP
        client's configuration per index (`minirag-mcp --index NAME`)."""
        return cls(name, frozenset(), alone=True)

    def reachable(self, name: str) -> bool:
        return self.allow is None or name == self.default or name in self.allow


UNRESTRICTED = Connection(default=None, allow=None)


class IndexRefused(Exception):
    """A call no index answers; the message is what the user is told."""


def _also(others: tuple[str, ...]) -> str:
    """The reachable indexes a refused call could use instead, as a closing sentence."""
    return f" Other reachable indexes: {', '.join(others)}." if others else ""


def _own_entry(name: str) -> str:
    """Where a server of another index alone sends a call for index `name`."""
    return (
        f"its own entry in the client's MCP configuration (`minirag-mcp --index {name}`): use "
        "that server's tools, or tell the user to add that entry"
    )


def _refused(name: str, why: str, others: tuple[str, ...]) -> IndexRefused:
    return IndexRefused(f"Index {name!r}: {why.rstrip('.')}.{_also(others)}")


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
    # The latest sync job of the handle a reload replaced with this one. While it runs the
    # index is syncing; it is polled through this entry, its outcome too.
    predecessor_job: SyncJob | None = None

    def runtime(self) -> OpenIndex:
        """The open index; IndexUnavailable while it cannot be opened, tried again on the
        next call."""
        try:
            return self.handle.get()
        except Exception as e:
            raise IndexUnavailable(self.name, _reason(e)) from e

    def state(self) -> tuple[IndexState, str]:
        job = self.predecessor_job
        if job is not None and job.running:
            return IndexState.SYNCING, ""
        return self.handle.state()

    def sync_status(self, job_id: str) -> SyncJob:
        """The sync job `job_id` of this index: the handle's latest, or the predecessor's.
        KeyError for any other."""
        job = self.predecessor_job
        if job is not None and job.job_id == job_id:
            return job
        return self.handle.sync.status(job_id)


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
    predecessor_job: SyncJob | None = None,
) -> IndexEntry:
    # Without a factory each handle builds and owns its embedders, so each index checks its
    # own recorded probe; indexes of one model still share its loaded model (embedder.py).
    factory = None if embedder_factory is None else _entry_factory(config, embedder_factory)
    handle = IndexHandle(config, embedder_factory=factory, model_factory=model_factory)
    return IndexEntry(name, config, description, handle, predecessor_job)


def _serves(old: Config, new: Config) -> bool:
    """Whether a handle built for `old` serves an index configured as `new` as one built
    for `new` would. Every Config carries the endpoint of every [embedding] table, and only
    those of the models the index embeds with count: its own, and the one its vectors were
    made with, which embeds its queries until a rebuild."""
    if old == new:
        return True
    try:
        recorded = read_identity(new.db_path)
    except (IdentityError, IndexOpenError):
        return False
    models = {new.model_name} | ({recorded.embedding.model} if recorded is not None else set())

    def narrowed(config: Config) -> Config:
        endpoints = {m: e for m, e in config.embed_endpoints.items() if m in models}
        return replace(config, embed_endpoints=endpoints)

    return narrowed(old) == narrowed(new)


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
        """The entries `server` configures. An index its previous handle still serves keeps
        that handle, and its entry too unless its description changed. A replaced handle's
        latest sync job, or else the one it inherited, passes to its replacement."""
        entries = {}
        for name, config in server.indexes.items():
            description = server.descriptions[name]
            kept = previous.get(name)
            if kept is None or not _serves(kept.config, config):
                # A handle's own job is the later one: the sync lock refuses it while the
                # job it inherited runs.
                job = None if kept is None else kept.handle.sync_job or kept.predecessor_job
                entries[name] = _entry(
                    name, config, description, self.embedder_factory, self.model_factory, job
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
        it must not find it before this load has taken effect. A failed load stores none,
        so the next refresh loads again: the cause may lie outside the file, such as a key
        file it names."""
        source = self._source
        try:
            server = load_server_config(source.path, source.env)
            entries = source.entries(server, self._entries)
        except Exception as e:
            error = str(e) if isinstance(e, (ConfigError, OSError)) else f"{type(e).__name__}: {e}"
            self.config_error = error
            self._digest = None
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
            try:
                handle.close()
            except Exception as e:
                print(
                    "minirag-mcp: could not close the previous handle of index "
                    f"{handle.config.index_name!r}: {e}",
                    file=sys.stderr,
                )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._snapshot())

    @property
    def entries(self) -> tuple[IndexEntry, ...]:
        """Every configured index, in the file's order."""
        return tuple(self._snapshot().values())

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
            if connection.alone:
                raise IndexRefused(
                    f"This server serves only index {connection.default!r}. Index {name!r} is "
                    f"served by {_own_entry(name)}."
                )
            allowed = ",".join(sorted(connection.allow or ()))
            raise IndexRefused(
                f"Index {name!r} is not allowed on this connection (allow={allowed}). "
                f"Reachable indexes: {reachable}."
            )
        return entry

    def _others(
        self, entries: Mapping[str, IndexEntry], connection: Connection, name: str
    ) -> tuple[str, ...]:
        return tuple(n for n in entries if n != name and connection.reachable(n))

    def open(
        self, connection: Connection, requested: str | None, use: Use
    ) -> tuple[IndexEntry, OpenIndex]:
        """The entry and its open index, for `use`. An ingest is admitted by the handle,
        and the open index returned for it carries the pipeline the handle admitted."""
        entries = self._snapshot()
        entry = self._resolve(entries, connection, requested)
        others = self._others(entries, connection, entry.name)
        try:
            opened = _admitted(entry) if use is Use.INGEST else entry.handle.get()
        except RebuildPendingError as e:
            raise _refused(entry.name, str(e), others) from e
        except Exception as e:
            raise IndexUnavailable(entry.name, _reason(e), others) from e
        if opened is None:
            raise IndexRefused(
                f"Index {entry.name!r} changed twice while this ingest was being admitted "
                "(another process rebuilt it, or recorded its probe), so nothing was written. "
                f"Retry it.{_also(others)}"
            )
        return entry, opened

    @contextmanager
    def refusals(self, connection: Connection, entry: IndexEntry) -> Iterator[None]:
        """Refusals raised in the block, named as `open` names its own: a sync the handle of
        `entry` does not start, a write its store turns away, an index it cannot serve as
        it stands, and, in file mode, a path outside its roots. A block holds only what
        precedes work: a failure of the work itself is not a refusal."""
        try:
            yield
        except (RebuildPendingError, SyncBusyError, SyncLockBusy, IndexBusyError) as e:
            others = self._others(self._snapshot(), connection, entry.name)
            why = str(e)
            if isinstance(e, SyncLockBusy) and self.path is not None:
                why = e.naming(f"db in {self.path} [index.{entry.name}]")
            raise _refused(entry.name, why, others) from e
        except IndexUnavailableError as e:
            others = self._others(self._snapshot(), connection, entry.name)
            raise IndexUnavailable(entry.name, _reason(e), others) from e
        except OutsideRootsError as e:
            # The environment's text names BASE_DIRS, which configures that one index.
            if self.path is None:
                raise
            raise self._outside(connection, entry, e) from e

    def _outside(
        self, connection: Connection, entry: IndexEntry, error: OutsideRootsError
    ) -> IndexRefused:
        """A path outside the roots of `entry`, and the reachable index whose roots hold it:
        the one a call that forgot `index` meant. A server of `entry` alone names the index
        holding it, which only that index's own server reaches."""
        entries = self._snapshot()
        others = self._others(entries, connection, entry.name)
        roots = ", ".join(str(root) for root in entry.config.roots)
        why = f"{error.requested} is outside its roots ({roots})"
        candidates = [n for n in entries if n != entry.name] if connection.alone else others
        holder = next((n for n in candidates if within(error.path, entries[n].config.roots)), None)
        if holder is not None and connection.alone:
            why += f" but under those of index {holder!r}, which is served by {_own_entry(holder)}"
        elif holder is not None:
            why += f' but under those of index {holder!r}: pass index="{holder}" to reach it'
        else:
            why += (
                ", and no other index this connection may use holds it; a folder is reached "
                f"by adding it to an index's roots in {self.path}"
            )
        return _refused(entry.name, why, others)

    def in_roots(
        self,
        connection: Connection,
        entry: IndexEntry,
        path: str,
        *,
        require_absolute: bool = True,
    ) -> Path:
        """`path` resolved inside the roots of `entry`, or refused as `refusals` words it."""
        with self.refusals(connection, entry):
            return resolve_in_roots(path, entry.config.roots, require_absolute=require_absolute)
