"""fastembed wrapper: lazy model load, registry-based dim lookup, token counting."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from fastembed import TextEmbedding

# fastembed already pulls this in — its tokenizers *are* `tokenizers.Tokenizer` objects —
# but we construct one directly, so it is declared as a dependency of ours too.
from tokenizers import Tokenizer

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


def _untruncated_counter(model: TextEmbedding) -> Callable[[str], int]:
    """A counter built on a *copy* of the model's tokenizer, with truncation switched off.

    fastembed configures the tokenizer it embeds with for truncation at the model's
    trained sequence length, so `TextEmbedding.token_count` cannot report a number above
    that length: every over-long text counts as exactly the ceiling. A budget compared
    against a saturating count stops meaning anything as it approaches that ceiling —
    at the ceiling itself `count <= budget` is true of a one-megabyte string.

    The copy matters. Turning truncation off on the model's own tokenizer would feed the
    encoder sequences longer than the ONNX graph accepts, so this serializes the
    configured tokenizer and rebuilds it, leaving the original untouched.
    """
    tokenizer = getattr(getattr(model, "model", None), "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("fastembed model exposes no tokenizer to count with")
    counting = Tokenizer.from_str(tokenizer.to_str())
    counting.no_truncation()
    # Padding would count towards `len(ids)`; the special tokens are wanted, the pad
    # tokens are not, and the encoder does not attend to them either.
    counting.no_padding()
    return lambda text: len(counting.encode(text).ids)


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
        self._count: Callable[[str], int] | None = None
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

        The count is **not** truncated at the model's trained sequence length. A budget
        is only a budget if the counter can report a number above it: with fastembed's
        own `token_count`, which saturates at that length, `count <= budget` is
        vacuously true for every text once the budget reaches the ceiling, and the
        packer silently stops enforcing anything. So this counts with a copy of the
        tokenizer that has truncation disabled (see `_untruncated_counter`), which is
        also the only way the packer can be told how far over the ceiling a piece is.

        The tokenizer is two attributes away from a third-party backend, and a chunker
        that silently stops counting tokens would produce a corpus of truncated vectors
        with nothing to show for it. So a failure degrades to the character estimate,
        loudly and once. It degrades *there* and never to `token_count`: the estimate is
        rough but monotonic, and a saturating counter is not a counter at all.
        """
        if self._tokenizer_broken:
            return estimate_tokens(text)
        try:
            if self._count is None:
                self._count = _untruncated_counter(self._load())
            return int(self._count(text))
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
