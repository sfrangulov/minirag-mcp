"""fastembed wrapper: lazy model load, registry-based dim lookup, token counting."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from pathlib import Path

from fastembed import TextEmbedding

from minirag_mcp.chunker.tokens import estimate_tokens


class UnknownModelError(Exception):
    pass


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, 0.0 when either side is the zero vector."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


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
        self._tokenizer_broken = False

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

    def count_tokens(self, text: str) -> int:
        """How many tokens this model would make of `text` — the chunker's budget unit.

        Counts saturate at the model's trained sequence length (128 for the shipped
        multilingual MiniLM), because fastembed's tokenizer has truncation configured.
        That cannot affect a packing decision: the chunker only ever asks whether a
        piece is within a budget set below that ceiling, and a saturated count is
        already over any such budget.

        `TextEmbedding.token_count` is public in fastembed 0.8, but it is one method
        on one backend away from the tokenizer, and a chunker that silently stops
        counting tokens would produce a corpus of truncated vectors with nothing to
        show for it. So a failure degrades to the character estimate, loudly and once.
        """
        if self._tokenizer_broken:
            return estimate_tokens(text)
        try:
            return int(self._load().token_count(text))
        except Exception as e:
            self._tokenizer_broken = True
            warnings.warn(
                f"Could not count tokens with the {self.model_name!r} tokenizer: {e}. "
                "Falling back to a character-based estimate for chunk sizing, which is "
                "less accurate — chunks will be smaller than they need to be, and text "
                "that tokenizes unusually badly may still be truncated by the encoder.",
                RuntimeWarning,
                stacklevel=2,
            )
            return estimate_tokens(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [_unit([float(x) for x in v]) for v in self._load().embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [_unit([float(x) for x in v]) for v in self._load().query_embed([text])][0]
