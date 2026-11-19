"""One place opens an index: its identity, embedder, store, labels, pipeline and sync manager.

Both shells ask an `IndexHandle` for the open index and build nothing of their own, so
what an index is opened with cannot differ between them. `make_embedder` is the one
constructor of an embedder for a scheme.

An index is opened by the schemes its INDEX_FILE records, never by the configured ones:
its vectors are comparable only with queries embedded the same way. A configuration that
differs makes it `rebuild pending`, and it keeps answering from what it holds.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import Config
from minirag_mcp.embedder import Embedder, UnknownModelError, prefix_rule
from minirag_mcp.endpoints import EmbeddingEndpoint, RerankEndpoint
from minirag_mcp.identity import (
    INDEX_FILE,
    ChunkingScheme,
    EmbeddingScheme,
    IdentityError,
    IndexIdentity,
    IndexState,
    create_identity,
    index_state,
    is_legacy_index,
    legacy_error,
    read_identity,
    recover_swap,
    update_identity,
)
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.labels import LabelerSource
from minirag_mcp.labels.endpoint_model import endpoint_model_factory
from minirag_mcp.labels.labeler import ModelFactory
from minirag_mcp.labels.model import default_model_factory
from minirag_mcp.results import embedder_notes
from minirag_mcp.store import Store
from minirag_mcp.sync import SyncManager

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


@dataclass(frozen=True)
class _Opened:
    index: OpenIndex
    sync: SyncManager
    # Built for the configuration: `index.embedder` itself unless the index was built
    # with another scheme.
    configured: object


class IndexHandle:
    """Opens one index on first use and hands every caller the same open index.

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

    def get(self) -> OpenIndex:
        return self._opened().index

    @property
    def sync(self) -> SyncManager:
        return self._opened().sync

    def configured(self) -> tuple[EmbeddingScheme, ChunkingScheme]:
        """The schemes the configuration asks for, which a rebuild would apply."""
        return (
            _scheme_of(self._opened().configured),
            ChunkingScheme(SCHEME_VERSION, self.config.token_budget),
        )

    def state(self) -> tuple[IndexState, str]:
        """The index's state and why; never raises. An index that cannot be opened is
        unavailable, with the open's error as the reason."""
        try:
            opened = self._opened()
            if opened.sync.running:
                return IndexState.SYNCING, ""
            return index_state(self.config, opened.configured)
        except Exception as e:
            return IndexState.UNAVAILABLE, str(e)

    def _opened(self) -> _Opened:
        if (current := self._current) is not None:
            return current
        with self._lock:
            if self._current is None:
                self._current = self._build()
            return self._current

    def _build(self) -> _Opened:
        config = self.config
        configured = self._make(configured_scheme(config))
        built = [configured]
        try:
            identity = self._identity(
                _scheme_of(configured), ChunkingScheme(SCHEME_VERSION, config.token_budget)
            )
            embedder = self._embedder_for(identity, configured)
            if embedder is not configured:
                built.append(embedder)
            store = Store(
                config.db_path, dim=identity.embedding.dim, fts_language=config.fts_language
            )
            identity = self._record_fts_language(identity, store)
            pipeline = Pipeline(store, embedder, config, labels=self._labels)
            reranker = RerankEndpoint(config.rerank_endpoint) if config.rerank_endpoint else None
        except BaseException:
            for made in built:
                self._release(made)
            raise
        index = OpenIndex(config, store, embedder, pipeline, self._labels, identity, reranker)
        return _Opened(index, SyncManager(pipeline, store, config), configured)

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

    def _embedder_for(self, identity: IndexIdentity, configured) -> object:
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
            embedder = self._make(recorded)
        except UnknownModelError as e:
            raise unavailable(f"which cannot be loaded here: {str(e).rstrip('.')}") from e
        try:
            width = embedder.dim
        except UnknownModelError as e:
            self._release(embedder)
            raise unavailable(f"which cannot be loaded here: {str(e).rstrip('.')}") from e
        except BaseException:
            self._release(embedder)
            raise
        if width != recorded.dim:
            self._release(embedder)
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
                stacklevel=5,
            )
            return identity
        # None: a swap put another generation in place since it was read.
        return updated or identity

    def _release(self, embedder: object) -> None:
        if self._owns_embedders and (endpoint := getattr(embedder, "endpoint", None)):
            endpoint.close()

    def ingest_pipeline(self) -> Pipeline:
        """The pipeline the ingest tools and commands write through; RebuildPendingError
        while the index needs a rebuild, which leaves it to searches and deletes until an
        unscoped sync rebuilds it."""
        opened = self._opened()
        state, why = index_state(self.config, opened.configured)
        if state is IndexState.REBUILD_PENDING:
            raise RebuildPendingError(
                f"The index at {self.config.db_path} differs from its configuration ({why}), "
                "so this index needs a rebuild: run an unscoped sync (sync_start without a "
                "path, or `minirag-mcp sync`). Ingests wait for it; searches are answered "
                "from the existing index meanwhile."
            )
        if state is IndexState.UNAVAILABLE:
            raise IndexUnavailableError(why)
        return opened.index.pipeline

    def notes(self) -> list[str]:
        """What a result's `warnings` carries about the open index; empty while all is well."""
        current = self._current
        return embedder_notes(current.index.embedder) if current is not None else []

    def close(self) -> None:
        """Let the open index go and close the endpoint clients the handle built for it.

        The next `get()` opens the index again, with a new sync manager. A sync job already
        running finishes with what it was given.
        """
        with self._lock:
            current, self._current = self._current, None
        if current is not None:
            self._release(current.index.embedder)
            if current.configured is not current.index.embedder:
                self._release(current.configured)
            if current.index.reranker is not None:
                current.index.reranker.close()
