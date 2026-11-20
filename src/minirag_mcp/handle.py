"""One place opens an index: its identity, embedder, store, labels, pipeline and sync manager.

Both shells ask an `IndexHandle` for the open index and build nothing of their own, so
what an index is opened with cannot differ between them. `make_embedder` is the one
constructor of an embedder for a scheme.

An index is opened by the schemes its INDEX_FILE records, never by the configured ones:
its vectors are comparable only with queries embedded the same way. A configuration that
differs makes it `rebuild pending`, and it keeps answering from what it holds until an
unscoped sync rebuilds it. A rebuild swapped in by any process is followed: the handle
opens the new generation before the next call is answered, and so it does when another
process records the probe of an index this one opened before it had one.
"""

from __future__ import annotations

import contextlib
import os
import threading
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import Config
from minirag_mcp.embedder import (
    QUERY_TIMEOUT,
    Embedder,
    EmbedderUnavailableError,
    ProbeMismatchError,
    UnknownModelError,
    prefix_rule,
)
from minirag_mcp.endpoints import EmbeddingEndpoint, RerankEndpoint
from minirag_mcp.identity import (
    INDEX_FILE,
    ChunkingScheme,
    EmbeddingScheme,
    IdentityError,
    IndexIdentity,
    IndexState,
    RebuildInfo,
    acquire_sync_lock,
    create_identity,
    index_state,
    is_legacy_index,
    legacy_error,
    read_identity,
    rebuild_running,
    record_probe,
    recover_swap,
    update_identity,
)
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.labels import LabelerSource
from minirag_mcp.labels.endpoint_model import endpoint_model_factory
from minirag_mcp.labels.labeler import ModelFactory
from minirag_mcp.labels.model import default_model_factory
from minirag_mcp.results import embedder_notes
from minirag_mcp.store import IndexBusyError, IndexOpenError, Store
from minirag_mcp.sync import OnRebuild, SyncManager, _run_sync_unlocked

EmbedderFactory = Callable[[EmbeddingScheme], object]


class IndexUnavailableError(Exception):
    """The index cannot be opened as it stands; the message says why and what to do."""


class RebuildPendingError(Exception):
    """An ingest into an index whose configuration changed since it was built."""


def configured_scheme(config: Config) -> EmbeddingScheme:
    """The embedding scheme `config` asks for; UnknownModelError for a model fastembed
    does not know."""
    model = config.model_name
    return EmbeddingScheme(model, Embedder(model, config.cache_dir).dim, prefix_rule(model))


def make_embedder(
    config: Config, scheme: EmbeddingScheme, *, threads: int | None = None
) -> Embedder:
    """An embedder for `scheme`, through the endpoint configured for its model if any."""
    endpoint = config.embed_endpoints.get(scheme.model)
    return Embedder(
        scheme.model,
        config.cache_dir,
        endpoint=EmbeddingEndpoint(endpoint) if endpoint else None,
        prefixes=scheme.prefixes,
        threads=threads,
    )


def _scheme_of(embedder) -> EmbeddingScheme:
    return EmbeddingScheme(embedder.model_name, embedder.dim, embedder.prefixes)


@dataclass(frozen=True)
class OpenIndex:
    config: Config
    store: Store
    # Built for the recorded scheme, whatever the configuration asks for.
    embedder: object
    pipeline: Pipeline
    labels: LabelerSource
    identity: IndexIdentity
    reranker: RerankEndpoint | None = None


def _stat_key(db: Path) -> tuple[int, int] | None:
    """What tells one INDEX_FILE of `db` from another without reading it; None while there
    is none to read."""
    try:
        st = os.stat(db / INDEX_FILE)
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns


class _Watch:
    """Tells whether INDEX_FILE at `db` holds another generation than the one opened, or a
    probe the open one was not armed with."""

    def __init__(
        self, db: Path, generation: int, probe: tuple[float, ...], key: tuple[int, int] | None
    ) -> None:
        self.db = db
        self.generation = generation
        self.probe = probe
        self._key = key

    def _changed(self) -> IndexIdentity | None:
        """The identity at `db` when it differs from the one watched. Reads INDEX_FILE only
        when it changed since the last look. While there is none to read (a swap between
        its renames), the open generation stays."""
        key = _stat_key(self.db)
        if key is None:
            # A swap whose process died between its renames is finished here; a live one
            # holds the identity lock, which this waits for.
            with contextlib.suppress(IdentityError, IndexOpenError, OSError):
                recover_swap(self.db)
            key = _stat_key(self.db)
        if key is None or key == self._key:
            return None
        try:
            found = read_identity(self.db)
        except (IdentityError, IndexOpenError):
            return None
        if found is None:
            return None
        if found.generation == self.generation and found.probe == self.probe:
            self._key = key
            return None
        return found

    def replaced(self) -> bool:
        """Whether another generation is in place."""
        found = self._changed()
        return found is not None and found.generation != self.generation

    def stale(self) -> bool:
        """Whether another generation is in place, or a probe was recorded since."""
        return self._changed() is not None

    def armed(self, probe: tuple[float, ...]) -> None:
        """The open generation's embedder is armed with `probe` now."""
        self.probe = probe


def _replaced(db: Path) -> IndexBusyError:
    return IndexBusyError(
        f"{db} was replaced by a rebuilt index while this write was being prepared, so it "
        "was not written: it was made the way the replaced index was. Retry it."
    )


def _same_generation(db: Path, generation: int) -> Callable[[], None]:
    """A store's `write_check`: refuses a write once another generation is in place."""

    def check() -> None:
        # Never recover_swap here: it takes the identity lock exclusively, which this
        # thread already holds shared through another descriptor (`writing`).
        found = read_identity(db)
        if found is None or found.generation != generation:
            raise _replaced(db)

    return check


class _BeforeStore:
    """Called before every write of one generation's pipeline.

    Refuses a write prepared for a generation another has replaced since, which would put
    chunks made the old way into the new index. Records the probe at the first write into
    a fresh index, so that write fixes which model it was built with, and arms the
    embedder with it; a probe another writer recorded first arms it instead, and since this
    write's vectors were made before that, the served model is checked against it now.
    Refuses the write while the served model is not the recorded one.
    """

    def __init__(self, embedder, watch: _Watch) -> None:
        self._embedder = embedder
        self._watch = watch
        self._lock = threading.Lock()
        self._recorded = False

    def __call__(self) -> None:
        db = self._watch.db
        if self._watch.replaced():
            raise _replaced(db)
        IndexHandle._refuse_mismatch(self._embedder)
        if self._recorded:
            return
        embedder = self._embedder
        with self._lock:
            if self._recorded:
                return
            current = read_identity(db)
            if current is None or current.generation != self._watch.generation:
                raise _replaced(db)
            measured = None
            if not current.probe:
                measured = tuple(embedder.probe_vector())
                if why := embedder.reference_error(measured):
                    raise IndexUnavailableError(why)
                warn_unverifiable(embedder, "this index")
                current = record_probe(db, current.generation, measured)
                if current is None:
                    raise _replaced(db)
            embedder.expect_probe(current.probe)
            self._watch.armed(current.probe)
            if current.probe != measured:
                IndexHandle._refuse_mismatch(embedder, now=True)
            self._recorded = True


def warn_unverifiable(embedder, what: str) -> None:
    """Warn when an endpoint serves a model no reference probe ships for."""
    if getattr(embedder, "endpoint", None) is not None and not embedder.has_reference():
        warnings.warn(
            f"No reference probe ships for {embedder.model_name}, so the model the endpoint "
            f"serves cannot be verified; {what} records whatever it serves as that model.",
            RuntimeWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class _Opened:
    index: OpenIndex
    # Built for the configuration: `index.embedder` itself unless the index was built
    # with another scheme.
    configured: object
    # Every embedder this generation uses, by the scheme it was asked for with.
    made: Mapping[EmbeddingScheme, object]
    watch: _Watch
    before_store: _BeforeStore


def _distinct(embedders: Iterable[object]) -> list[object]:
    out: list[object] = []
    for e in embedders:
        if all(e is not o for o in out):
            out.append(e)
    return out


class IndexHandle:
    """Opens one index on first use and hands every caller the same open index, until
    another generation replaces it.

    A failed open raises and leaves nothing open, so the next call tries again. The
    endpoint clients of embedders the handle built itself are closed when what they
    served is let go; embedders from an injected factory belong to whoever supplied it,
    and may serve other indexes too.
    """

    def __init__(
        self,
        config: Config,
        *,
        embedder_factory: EmbedderFactory | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.config = config
        self._owns_embedders = embedder_factory is None
        self._make = embedder_factory or (lambda scheme: make_embedder(config, scheme))
        served = model_factory is None and config.classify_endpoint is not None
        if model_factory is None:
            model_factory = (
                endpoint_model_factory(config.classify_endpoint)
                if served
                else default_model_factory
            )
        self._labels = LabelerSource(
            config.roots, config.cache_dir, model_factory, needs_extra=not served
        )
        self._lock = threading.Lock()
        self._current: _Opened | None = None
        # Kept across generations, so a job started before a swap is still the one polled.
        self._sync: SyncManager | None = None

    def get(self) -> OpenIndex:
        return self._opened().index

    @property
    def sync(self) -> SyncManager:
        self._opened()
        return self._sync

    @property
    def sync_running(self) -> bool:
        """Whether a sync job of this handle is pending or running; never opens the index."""
        sync = self._sync
        return sync is not None and sync.running

    def state(self) -> tuple[IndexState, str]:
        """The index's state and why; never raises. An index that cannot be opened is
        unavailable, with the open's error as the reason. It is syncing while a sync of this
        handle runs, or a rebuild of the index by any process."""
        try:
            opened = self._opened()
            if (self._sync is not None and self._sync.running) or rebuild_running(
                self.config.db_path
            ):
                return IndexState.SYNCING, ""
            return index_state(self.config, opened.configured)
        except Exception as e:
            return IndexState.UNAVAILABLE, str(e)

    def _opened(self, *, write: bool = False) -> _Opened:
        """The open generation. A known mismatch is checked again within QUERY_TIMEOUT for a
        read, and within the endpoint's own timeout for a `write`."""
        current = self._current
        if current is None or current.watch.stale():
            with self._lock:
                if self._current is current:
                    self._current = self._reopen(current, {})
                current = self._current
        self._refuse_mismatch(current.index.embedder, timeout=None if write else QUERY_TIMEOUT)
        return current

    def _reopen(self, current: _Opened | None, keep: Mapping[EmbeddingScheme, object]) -> _Opened:
        """Open the generation in place now, with the embedders of `current` and `keep`
        where they fit; those it does not use are let go. The caller holds `self._lock`."""
        held = current.made if current else {}
        # From both, not from the merged pool: an embedder of `current` that `keep` replaces
        # under its scheme is not in the pool, and is let go too.
        candidates = _distinct([*held.values(), *keep.values()])
        reranker = current.index.reranker if current else None
        try:
            opened = self._build({**held, **keep}, reranker)
        except BaseException:
            self._current = None
            for e in candidates:
                self._release(e)
            if reranker is not None:
                reranker.close()
            raise
        used = list(opened.made.values())
        for e in candidates:
            if all(e is not u for u in used):
                self._release(e)
        return opened

    @staticmethod
    def _refuse_mismatch(embedder, *, now: bool = False, timeout: float | None = None) -> None:
        """An endpoint that stopped serving the index's model keeps the index unavailable
        until a check passes again. `now` checks it even when no mismatch is known; a check
        that outlasts `timeout` passes nothing."""
        if not now and embedder.probe_error is None:
            return
        unchecked = False
        try:
            unchecked = embedder.verify(timeout=timeout)
        except EmbedderUnavailableError:
            pass
        if (why := embedder.probe_error) is not None:
            if unchecked:
                why += (
                    f" That was found by an earlier check: the endpoint did not answer within "
                    f"{timeout:g} s, so it could not be re-checked now."
                )
            raise IndexUnavailableError(why)

    def _build(
        self, pool: Mapping[EmbeddingScheme, object], reranker: RerankEndpoint | None
    ) -> _Opened:
        config = self.config
        made: dict[EmbeddingScheme, object] = {}
        built: list[object] = []

        def make(scheme: EmbeddingScheme) -> object:
            if scheme not in made:
                embedder = pool.get(scheme)
                if embedder is None:
                    embedder = self._make(scheme)
                    built.append(embedder)
                made[scheme] = embedder
            return made[scheme]

        db = config.db_path
        try:
            configured = make(configured_scheme(config))
            # Before the identity is read: a swap landing in between changes the file again.
            key = _stat_key(db)
            identity = self._identity(
                _scheme_of(configured), ChunkingScheme(SCHEME_VERSION, config.token_budget)
            )
            embedder = self._embedder_for(identity, configured, make)
            store = Store(db, dim=identity.embedding.dim, fts_language=config.fts_language)
            identity = self._record_fts_language(identity, store)
            store.write_check = _same_generation(db, identity.generation)
            chunking = replace(config, token_budget=identity.chunking.token_budget)
            if identity.probe:
                embedder.expect_probe(identity.probe)
            if identity.probe and getattr(embedder, "endpoint", None) is not None:
                self._verify_at_open(embedder)
            watch = _Watch(db, identity.generation, identity.probe, key)
            before_store = _BeforeStore(embedder, watch)
            pipeline = Pipeline(
                store, embedder, chunking, labels=self._labels, before_store=before_store
            )
            if reranker is None and config.rerank_endpoint:
                reranker = RerankEndpoint(config.rerank_endpoint)
        except BaseException:
            for e in built:
                self._release(e)
            raise
        index = OpenIndex(config, store, embedder, pipeline, self._labels, identity, reranker)
        if self._sync is None:
            self._sync = SyncManager(db, self._run_held)
        return _Opened(index, configured, made, watch, before_store)

    @staticmethod
    def _verify_at_open(embedder) -> None:
        """An endpoint serving another model than the index records is unavailable at open."""
        try:
            embedder.verify(timeout=QUERY_TIMEOUT)
        except ProbeMismatchError as e:
            raise IndexUnavailableError(str(e)) from e

    def ensure_probe(self) -> None:
        """What the open generation's pipeline calls before every write (`_BeforeStore`)."""
        self._opened(write=True).before_store()

    def _identity(self, embedding: EmbeddingScheme, chunking: ChunkingScheme) -> IndexIdentity:
        """The recorded identity; a folder without one is stamped with these schemes before
        anything else is written to it, so it never reads as a 1.x index."""
        db = self.config.db_path
        try:
            recover_swap(db)
            recorded = read_identity(db)
            if recorded is not None:
                return recorded
            if is_legacy_index(db):
                raise IndexUnavailableError(str(legacy_error(db, self.config.roots)))
            fresh = IndexIdentity.fresh(embedding, chunking, self.config.fts_language)
            return create_identity(db, fresh)
        except IdentityError as e:
            raise IndexUnavailableError(str(e)) from e

    def _embedder_for(
        self, identity: IndexIdentity, configured, make: Callable[[EmbeddingScheme], object]
    ) -> object:
        """The embedder every query and ingest of this index goes through: one for the
        scheme `identity` says its vectors were made with."""
        recorded = identity.embedding
        if _scheme_of(configured) == recorded:
            return configured

        def unavailable(why: str) -> IndexUnavailableError:
            return IndexUnavailableError(
                f"The index at {self.config.db_path} was built with {recorded.model}, {why}. "
                "An index is searched only with the model it was built with: make that model "
                "available here, or point DB_PATH at another folder to build a new index."
            )

        try:
            embedder = make(recorded)
            width = embedder.dim
        except UnknownModelError as e:
            raise unavailable(f"which cannot be loaded here: {str(e).rstrip('.')}") from e
        if width != recorded.dim:
            raise unavailable(
                f"at {recorded.dim} dimensions, but it makes {width}-dimensional vectors here"
            )
        return embedder

    def _record_fts_language(self, identity: IndexIdentity, store: Store) -> IndexIdentity:
        """`identity` with the stemmer the keyword indexes now use, recorded best effort."""
        language = store.fts_index_language
        if identity.fts_language == language:
            return identity
        db = self.config.db_path
        try:
            updated = update_identity(db, identity.generation, fts_language=language)
        except (OSError, IdentityError) as e:
            warnings.warn(
                f"The keyword indexes of {db} were rebuilt with the {language} stemmer, but "
                f"recording that in {db / INDEX_FILE} failed: {e}. Search is unaffected; the "
                f"file names {identity.fts_language} until an open records it.",
                RuntimeWarning,
                stacklevel=6,
            )
            return identity
        # None: a swap put another generation in place since it was read.
        return updated or identity

    def _release(self, embedder: object) -> None:
        if self._owns_embedders and (endpoint := getattr(embedder, "endpoint", None)):
            endpoint.close()

    def _pending(self, why: str, then: str) -> RebuildPendingError:
        return RebuildPendingError(
            f"The index at {self.config.db_path} differs from its configuration ({why}), "
            "so this index needs a rebuild: run an unscoped sync (sync_start without a "
            f"path, or `minirag-mcp sync`). {then}"
        )

    def ingest_pipeline(self) -> Pipeline:
        """The pipeline the ingest tools and commands write through; RebuildPendingError
        while the index needs a rebuild, which leaves it to searches and deletes until an
        unscoped sync rebuilds it. The served model is checked before it is handed out."""
        opened = self._opened(write=True)
        state, why = index_state(self.config, opened.configured)
        if state is IndexState.REBUILD_PENDING:
            raise self._pending(
                why, "Ingests wait for it; searches are answered from the existing index meanwhile."
            )
        if state is IndexState.UNAVAILABLE:
            raise IndexUnavailableError(why)
        self._refuse_mismatch(opened.index.embedder, now=True)
        return opened.index.pipeline

    def run_sync(
        self, scope: Path | None, on_event: Callable[[str], None] | None = None
    ) -> tuple[dict[str, int], list[dict], RebuildInfo | None]:
        """Sync the index with the roots, or only `scope` inside them, holding the sync lock.

        On a `rebuild pending` index an unscoped sync rebuilds it, and the handle opens the
        rebuilt generation; a scoped one is refused with RebuildPendingError, nothing
        written. SyncLockBusy while another sync of the index runs.
        """
        # Opened first: a 1.x index is refused before the lock file is made inside it.
        self._opened(write=True)
        lock = acquire_sync_lock(self.config.db_path)
        try:
            return self._sync_held(scope, on_event, None)
        finally:
            lock.release()

    def start_sync(self, scope: Path | None = None) -> str:
        """`run_sync` as a background job of `sync`; its job id. The refusal of a scoped
        sync on a `rebuild pending` index is raised here, before any job starts."""
        opened = self._opened(write=True)
        if scope is not None:
            state, why = index_state(self.config, opened.configured)
            if state is IndexState.REBUILD_PENDING:
                raise self._scoped_refusal(why, scope)
        return self.sync.start(scope)

    def _scoped_refusal(self, why: str, scope: Path) -> RebuildPendingError:
        return self._pending(
            why,
            f"A sync of {scope} alone would add chunks made the old way, so nothing was synced; "
            "searches are answered from the existing index meanwhile.",
        )

    def _run_held(
        self, scope: Path | None, on_rebuild: OnRebuild
    ) -> tuple[dict[str, int], list[dict], RebuildInfo | None]:
        return self._sync_held(scope, None, on_rebuild)

    def _sync_held(
        self,
        scope: Path | None,
        on_event: Callable[[str], None] | None,
        on_rebuild: OnRebuild | None,
    ) -> tuple[dict[str, int], list[dict], RebuildInfo | None]:
        """`run_sync` for a caller holding the sync lock (`acquire_sync_lock`). A rebuild
        is announced to `on_event`, and to `on_rebuild` with its kind and reasons."""
        opened = self._opened(write=True)
        state, why = index_state(self.config, opened.configured)
        if state is IndexState.UNAVAILABLE:
            raise IndexUnavailableError(why)
        if state is IndexState.REBUILD_PENDING:
            if scope is not None:
                raise self._scoped_refusal(why, scope)
            return self._rebuild(opened, on_event, on_rebuild)
        index = opened.index
        # A rebuild checks the embedder it builds with itself.
        self._refuse_mismatch(index.embedder, now=True)
        counts, errors = _run_sync_unlocked(
            index.pipeline,
            index.store,
            self.config.roots,
            self.config.max_file_size,
            scope,
            on_event,
            exclude=self.config.exclude,
            exclude_source=self.config.exclude_source,
        )
        return counts, errors, None

    def _rebuild(
        self,
        opened: _Opened,
        on_event: Callable[[str], None] | None,
        on_rebuild: OnRebuild | None,
    ) -> tuple[dict[str, int], list[dict], RebuildInfo]:
        from minirag_mcp.rebuild import rebuild  # rebuild.py imports this module

        # Only what the rebuild builds itself: an embedder the open index already holds for
        # a scheme (the configured one is built at open) is not loaded a second time.
        made: dict[EmbeddingScheme, object] = {}

        def make(scheme: EmbeddingScheme) -> object:
            if scheme in opened.made:
                return opened.made[scheme]
            if scheme not in made:
                made[scheme] = self._make(scheme)
            return made[scheme]

        try:
            counts, errors, info = rebuild(opened.index, make, on_event, on_rebuild)
        except BaseException:
            for e in _distinct(made.values()):
                self._release(e)
            raise
        with self._lock:
            current = self._current
            if current is opened:
                try:
                    self._current = self._reopen(current, made)
                except Exception as e:
                    # The rebuilt generation is in place either way; the next call opens it.
                    errors.append(
                        {
                            "source": str(self.config.db_path),
                            "error": f"rebuilt as generation {info.generation}, but opening "
                            f"it failed: {e}",
                        }
                    )
            else:
                # Another call opened the rebuilt generation first, or the handle was closed.
                used = list(current.made.values()) if current is not None else []
                for e in _distinct(made.values()):
                    if all(e is not u for u in used):
                        self._release(e)
        return counts, errors, info

    def notes(self) -> list[str]:
        """What a result's `warnings` carries about the open index; empty while all is well."""
        current = self._current
        return embedder_notes(current.index.embedder) if current is not None else []

    def close(self) -> None:
        """Let the open index go and close the endpoint clients the handle built for it.

        The next `get()` opens the index again, with a new sync manager. A sync job already
        running resolves the index through the handle, and so opens it again.
        """
        with self._lock:
            current, self._current = self._current, None
            self._sync = None
        if current is not None:
            for e in _distinct(current.made.values()):
                self._release(e)
            if current.index.reranker is not None:
                current.index.reranker.close()
