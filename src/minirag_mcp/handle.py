"""One place opens an index: its embedder, store, labels, pipeline and sync manager.

Both shells ask an `IndexHandle` for the open index and build nothing of their own, so
what an index is opened with cannot differ between them. `make_embedder` is the one
constructor of an embedder for a scheme.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from minirag_mcp.config import Config
from minirag_mcp.embedder import Embedder, prefix_rule
from minirag_mcp.endpoints import EmbeddingEndpoint
from minirag_mcp.identity import EmbeddingScheme
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.labels import LabelerSource
from minirag_mcp.labels.labeler import ModelFactory
from minirag_mcp.labels.model import default_model_factory
from minirag_mcp.results import embedder_notes
from minirag_mcp.store import Store
from minirag_mcp.sync import SyncManager

EmbedderFactory = Callable[[EmbeddingScheme], object]


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


@dataclass(frozen=True)
class OpenIndex:
    config: Config
    store: Store
    embedder: object
    pipeline: Pipeline
    labels: LabelerSource


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
        self._labels = LabelerSource(
            config.roots, config.cache_dir, model_factory or default_model_factory
        )
        self._lock = threading.Lock()
        self._current: tuple[OpenIndex, SyncManager] | None = None

    def get(self) -> OpenIndex:
        return self._opened()[0]

    @property
    def sync(self) -> SyncManager:
        return self._opened()[1]

    def _opened(self) -> tuple[OpenIndex, SyncManager]:
        if (current := self._current) is not None:
            return current
        with self._lock:
            if self._current is None:
                opened = self._build()
                self._current = (opened, SyncManager(opened.pipeline, opened.store, self.config))
            return self._current

    def _build(self) -> OpenIndex:
        config = self.config
        embedder = self._make(configured_scheme(config))
        try:
            store = Store(config.db_path, dim=embedder.dim, fts_language=config.fts_language)
        except BaseException:
            self._release(embedder)
            raise
        pipeline = Pipeline(store, embedder, config, labels=self._labels)
        return OpenIndex(config, store, embedder, pipeline, self._labels)

    def _release(self, embedder: object) -> None:
        if self._owns_embedders and (endpoint := getattr(embedder, "endpoint", None)):
            endpoint.close()

    def ingest_pipeline(self) -> Pipeline:
        """The pipeline every ingest tool and command writes through: the one place an
        ingest into this index is admitted."""
        return self.get().pipeline

    def notes(self) -> list[str]:
        """What a result's `warnings` carries about the open index; empty while all is well."""
        current = self._current
        return embedder_notes(current[0].embedder) if current is not None else []

    def close(self) -> None:
        """Let the open index go and close the endpoint clients the handle built for it.

        The next `get()` opens the index again, with a new sync manager. A sync job already
        running finishes with what it was given.
        """
        with self._lock:
            current, self._current = self._current, None
        if current is not None:
            self._release(current[0].embedder)
