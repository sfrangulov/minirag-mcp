"""fastembed wrapper and llama.cpp endpoint: prefixes, fallback, window cut, token counting."""

from __future__ import annotations

import math
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from fastembed import TextEmbedding
from huggingface_hub import hf_hub_download

# fastembed already pulls this in — its tokenizers *are* `tokenizers.Tokenizer` objects —
# but we construct one directly, so it is declared as a dependency of ours too.
from tokenizers import Tokenizer

from minirag_mcp.chunker import DEFAULT_TOKEN_BUDGET, MODEL_MAX_TOKENS
from minirag_mcp.chunker.tokens import estimate_tokens
from minirag_mcp.endpoints import EmbeddingEndpoint, EndpointError

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_LARGE = "intfloat/multilingual-e5-large"
MODEL_WINDOWS: dict[str, int] = {E5_LARGE: 512, MINILM: 128}
DEFAULT_BUDGETS: dict[str, int] = {E5_LARGE: 110, MINILM: 110}
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
ENDPOINT_RETRY_SECONDS = 60.0


def known_model(model_name: str) -> bool:
    return any(entry["model"] == model_name for entry in TextEmbedding.list_supported_models())


def model_window(model_name: str) -> int:
    return MODEL_WINDOWS.get(model_name, MODEL_MAX_TOKENS)


def default_budget(model_name: str) -> int:
    return DEFAULT_BUDGETS.get(model_name, DEFAULT_TOKEN_BUDGET)


def prefix_rule(model_name: str) -> str:
    lowered = model_name.lower()
    if lowered.startswith("intfloat/") and "e5" in lowered.split("/", 1)[1]:
        return "e5"
    return "none"


class UnknownModelError(Exception):
    pass


class EmbedderUnavailableError(Exception):
    """The embedding model could not be loaded, so nothing can be searched or ingested."""


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
    """One embedding model, run by an endpoint when one is given, else in-process.

    Both backends see the same text: the model's prefixes, and for the endpoint a cut to
    the window where fastembed would truncate. An unreachable endpoint falls back to the
    in-process model, loudly, and is tried again every ENDPOINT_RETRY_SECONDS.
    """

    # What a pickled copy carries. A loaded model, the tokenizers, the locks, the HTTP
    # session and the fallback state belong to one process and are rebuilt lazily.
    _CONFIG = ("model_name", "cache_dir", "endpoint", "prefixes", "threads", "allow_fallback")

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        *,
        endpoint: EmbeddingEndpoint | None = None,
        prefixes: str | None = None,
        threads: int | None = None,
        allow_fallback: bool = True,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.endpoint = endpoint
        self.prefixes = prefixes if prefixes is not None else prefix_rule(model_name)
        self.threads = threads
        self.allow_fallback = allow_fallback
        self._reset()

    def _reset(self) -> None:
        self._model: TextEmbedding | None = None
        self._dim: int | None = None
        self._count: Callable[[str], int] | None = None
        self._tokenizer_broken = False
        self._file_tokenizers: tuple[Tokenizer, Tokenizer] | None = None
        self._prefix_tokens: int | None = None
        self._load_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._warning: str | None = None
        self._failed_at = 0.0

    def __getstate__(self) -> dict:
        return {name: getattr(self, name) for name in self._CONFIG}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._reset()

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self._registry_entry()["dim"])
        return self._dim

    def _registry_entry(self) -> dict:
        for entry in TextEmbedding.list_supported_models():
            if entry["model"] == self.model_name:
                return entry
        raise UnknownModelError(
            f"Model {self.model_name!r} is not in the fastembed registry. "
            "See TextEmbedding.list_supported_models() for valid names."
        )

    @property
    def max_tokens(self) -> int:
        return model_window(self.model_name)

    @property
    def backend(self) -> str:
        if self.endpoint is None:
            return "in-process"
        return "in-process (endpoint unreachable)" if self._warning else "endpoint"

    @property
    def warning(self) -> str | None:
        return self._warning

    @property
    def prefix_tokens(self) -> int:
        """How many tokens the passage prefix adds to a chunk; the packer leaves room."""
        if self.prefixes != "e5":
            return 0
        if self._prefix_tokens is None:
            self._prefix_tokens = self.count_tokens(PASSAGE_PREFIX + "x") - self.count_tokens("x")
        return self._prefix_tokens

    def _load(self) -> TextEmbedding:
        with self._load_lock:
            if self._model is None:
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self._model = TextEmbedding(
                        model_name=self.model_name,
                        cache_dir=str(self.cache_dir),
                        threads=self.threads,
                    )
                except Exception as e:
                    raise EmbedderUnavailableError(
                        f"The embedding model {self.model_name!r} could not be loaded ({e}). "
                        "Every search and ingest needs it, so each one fails this way until "
                        "it loads; this is not a problem with the query. It is downloaded "
                        f"once, into {self.cache_dir}: the server needs to reach huggingface.co "
                        "for that, or CACHE_DIR has to point at a folder that already holds it."
                    ) from e
            return self._model

    def _tokenizers(self) -> tuple[Tokenizer, Tokenizer]:
        """The model's tokenizer.json, read without the ONNX model: a counter that never
        truncates, and a cutter that truncates to the window as fastembed does."""
        with self._load_lock:
            if self._file_tokenizers is None:
                repo = self._registry_entry()["sources"].get("hf")
                try:
                    if not repo:
                        raise FileNotFoundError("the model has no Hugging Face source")
                    counting = Tokenizer.from_file(self._tokenizer_path(repo))
                except Exception as e:
                    raise EmbedderUnavailableError(
                        f"The tokenizer of {self.model_name!r} could not be read from "
                        f"{repo or 'Hugging Face'} ({e}). It is needed to size chunks and to "
                        "cut text to the model's window before the embedding endpoint sees "
                        f"it. It is downloaded once, into {self.cache_dir}: the server needs "
                        "to reach huggingface.co for that, or CACHE_DIR has to point at a "
                        "folder that already holds it."
                    ) from e
                counting.no_truncation()
                counting.no_padding()
                cutting = Tokenizer.from_str(counting.to_str())
                cutting.enable_truncation(max_length=self.max_tokens)
                self._file_tokenizers = (counting, cutting)
            return self._file_tokenizers

    def _tokenizer_path(self, repo: str) -> str:
        where = {"repo_id": repo, "filename": "tokenizer.json", "cache_dir": str(self.cache_dir)}
        try:
            return hf_hub_download(**where, local_files_only=True)
        except Exception:
            return hf_hub_download(**where)

    def count_tokens(self, text: str) -> int:
        """How many tokens this model would make of `text` — the chunker's budget unit.

        The count is **not** truncated at the model's trained sequence length. A budget
        is only a budget if the counter can report a number above it: with fastembed's
        own `token_count`, which saturates at that length, `count <= budget` is
        vacuously true for every text once the budget reaches the ceiling, and the
        packer silently stops enforcing anything. So this counts with a tokenizer that
        has truncation disabled — a copy of the loaded model's (see
        `_untruncated_counter`), or with an endpoint the model's tokenizer.json alone, so
        the ONNX model is never loaded just to count.

        The loaded model's tokenizer is two attributes away from a third-party backend,
        and a chunker that silently stops counting tokens would produce a corpus of
        truncated vectors with nothing to show for it. So a failure degrades to the
        character estimate, loudly and once. It degrades *there* and never to
        `token_count`: the estimate is rough but monotonic, and a saturating counter is
        not a counter at all. A model or tokenizer file that does not load at all is not
        that case: it raises EmbedderUnavailableError.
        """
        if self._tokenizer_broken:
            return estimate_tokens(text)
        model = None
        if self._count is None:
            # Outside the fallback: a model that cannot load cannot embed what the
            # estimate would size, and the fallback, once taken, is never undone.
            if self.endpoint is not None:
                counting, _ = self._tokenizers()
                self._count = lambda t: len(counting.encode(t).ids)
            else:
                model = self._load()
        try:
            if self._count is None:
                self._count = _untruncated_counter(model)
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

    def _fit_window(self, text: str) -> str:
        counting, cutting = self._tokenizers()
        encoding = cutting.encode(text)
        if not encoding.overflowing:
            return text
        ends = [
            end
            for (_, end), special in zip(
                encoding.offsets, encoding.special_tokens_mask, strict=True
            )
            if not special
        ]
        # A kept token's span can reach into the first dropped one (XLM-R's lone "▁" covers
        # the next word's first character), and that character re-tokenizes on its own.
        for end in reversed(ends):
            if len(counting.encode(text[:end]).ids) <= self.max_tokens:
                return text[:end]
        return ""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.prefixes == "e5":
            texts = [PASSAGE_PREFIX + t for t in texts]
        return self._embed(list(texts), query=False)

    def embed_query(self, text: str) -> list[float]:
        if self.prefixes == "e5":
            text = QUERY_PREFIX + text
        return self._embed([text], query=True)[0]

    def _embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        if self.endpoint is not None and self._endpoint_due():
            fitted = [self._fit_window(t) for t in texts]
            try:
                vectors = self.endpoint.embed(fitted)
            except EndpointError as e:
                if not self.allow_fallback:
                    raise
                self._fall_back(e)
            else:
                self._check_width(vectors)
                with self._state_lock:
                    self._warning = None
                return vectors
        try:
            model = self._load()
        except EmbedderUnavailableError as e:
            if self.endpoint is None:
                raise
            raise EmbedderUnavailableError(
                f"The embedding endpoint {self.endpoint.endpoint.url} is unreachable, so "
                f"{self.model_name} was to run in-process, and that failed too. {e}"
            ) from e
        raw = model.query_embed(texts) if query else model.embed(texts)
        return [_unit([float(x) for x in v]) for v in raw]

    def _endpoint_due(self) -> bool:
        with self._state_lock:
            return (
                self._warning is None
                or time.monotonic() - self._failed_at >= ENDPOINT_RETRY_SECONDS
            )

    def _fall_back(self, error: EndpointError) -> None:
        url = self.endpoint.endpoint.url
        message = (
            f"The embedding endpoint {url} is unreachable ({error}). {self.model_name} runs "
            "in-process meanwhile: same vectors, slower. The endpoint is tried again every "
            f"{ENDPOINT_RETRY_SECONDS:g} s."
        )
        with self._state_lock:
            first = self._warning is None
            self._warning = message
            self._failed_at = time.monotonic()
        if first:
            warnings.warn(message, RuntimeWarning, stacklevel=4)

    def _check_width(self, vectors: list[list[float]]) -> None:
        widths = sorted({len(v) for v in vectors} - {self.dim})
        if widths:
            endpoint = self.endpoint.endpoint
            raise EmbedderUnavailableError(
                f"The embedding endpoint {endpoint.url} (model {endpoint.name!r}) returned "
                f"{widths[0]}-dimensional vectors, but {self.model_name} makes "
                f"{self.dim}-dimensional ones: it serves another model. Nothing was "
                "embedded; point the endpoint at the same model, or remove it to embed "
                "in-process."
            )
