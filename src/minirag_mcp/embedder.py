"""fastembed wrapper: lazy model load, registry-based dim lookup."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from fastembed import TextEmbedding


class UnknownModelError(Exception):
    pass


def _unit(vector: list[float]) -> list[float]:
    """Scale to unit length so LanceDB's L2 ranking matches cosine similarity.

    sentence-transformers paraphrase-* models are trained for cosine; for unit
    vectors ‖a-b‖² = 2 - 2·cos, so L2 order and cosine order coincide.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


class Embedder:
    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: TextEmbedding | None = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            for entry in TextEmbedding.list_supported_models():
                if entry["model"] == self.model_name:
                    self._dim = int(entry["dim"])
                    break
            else:
                raise UnknownModelError(
                    f"Model {self.model_name!r} is not in the fastembed registry. "
                    "See TextEmbedding.list_supported_models() for valid names."
                )
        return self._dim

    def _load(self) -> TextEmbedding:
        if self._model is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [_unit([float(x) for x in v]) for v in self._load().embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [_unit([float(x) for x in v]) for v in self._load().query_embed([text])][0]
