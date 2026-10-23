"""In-process choice model behind the [classify] extra: a GGUF read out as letter probabilities.

llama_cpp, numpy and huggingface_hub are imported lazily inside functions: the core
package must import and run without the extra. Nothing here may write to stdout, which
is the MCP stdio channel; llama.cpp logs go to stderr.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import math
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from minirag_mcp.labels.taxonomy import ModelSpec

INSTALL_HINT = (
    "install the classify extra: uv tool install 'minirag-mcp[classify]' "
    "(or pip install 'minirag-mcp[classify]')"
)

LETTERS = "ABCDEFGHIJKLMNOP"
N_CTX = 8192
MAX_EVIDENCE_CHARS = 6000

SYSTEM_PROMPT = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
# Qwen3.5 template with thinking switched off: the empty think block is part of the
# format the model was tuned on.
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


# ggml_log_level values.
_LOG_ERROR = 3
_LOG_CONT = 5

# llama.cpp keeps only a raw pointer to its log callback, so the ctypes object must stay
# referenced for as long as the library may log: a collected one crashes the next line.
_log_callback: tuple[object, object] | None = None  # (the llama_cpp module, its callback)


def _route_llama_logs(llama_cpp) -> None:
    """Pass llama.cpp's own log lines to stderr, errors only.

    The library is built with verbose=True because verbose=False points fds 1 and 2 at
    /dev/null around the load, process wide, losing whatever the MCP server writes to
    stdout meanwhile. verbose=True, left alone, prints every load and context line.
    """
    global _log_callback
    if _log_callback is None or _log_callback[0] is not llama_cpp:
        last = [0]

        @llama_cpp.llama_log_callback
        def forward(level: int, text: bytes, user_data) -> None:
            if level != _LOG_CONT:
                last[0] = level
            if last[0] == _LOG_ERROR and text:
                out = _stderr()
                out.write(text.decode("utf-8", "replace"))
                out.flush()

        _log_callback = (llama_cpp, forward)
    llama_cpp.llama_log_set(_log_callback[1], ctypes.c_void_p(0))


class ModelError(Exception):
    """The choice model is unavailable or failed. Always loud, never a silent no-op."""


class ChoiceModel(Protocol):
    def probabilities(
        self, evidence: str, question: str, options: Sequence[tuple[str, str]]
    ) -> dict[str, float]:
        """Probability per value id; `options` are (value_id, description)."""
        ...

    def close(self) -> None: ...


def available() -> bool:
    return all(importlib.util.find_spec(mod) is not None for mod in ("llama_cpp", "numpy"))


def build_prompt(evidence: str, question: str, options: Sequence[tuple[str, str]]) -> str:
    user = json.dumps(
        {
            "evidence": evidence,
            "criterion": question,
            "options": [
                {"letter": LETTERS[i], "description": f"{vid}: {desc}"}
                for i, (vid, desc) in enumerate(options)
            ],
        },
        ensure_ascii=False,
    )
    return CHAT_TEMPLATE.format(system=SYSTEM_PROMPT, user=user)


def calibrate(letter_logprobs: Sequence[float], temperature: float) -> list[float]:
    top = max(letter_logprobs)
    weights = [math.exp((z - top) / temperature) for z in letter_logprobs]
    total = sum(weights)
    return [w / total for w in weights]


# sha256 of each downloaded model file this process has read, keyed by what would change
# if the file did. Every MCP ingest_file is a job of its own, and each job resolves the
# file again: hashing 4.5 GB every time would cost more than labelling the document.
_digests: dict[tuple[str, int, int], str] = {}


def _sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _digests:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        _digests[key] = digest.hexdigest()
    return _digests[key]


def resolve_model_file(spec: ModelSpec, cache_dir: Path) -> Path:
    if spec.path is not None:
        if not spec.path.is_file():
            raise ModelError(f"the classify model file {spec.path} does not exist")
        return spec.path

    from huggingface_hub import hf_hub_download

    try:
        downloaded = Path(
            hf_hub_download(
                spec.repo,
                spec.file,
                revision=spec.revision,
                cache_dir=cache_dir / "classify",
            )
        )
    except Exception as e:
        raise ModelError(f"could not download {spec.file} from {spec.repo}: {e}") from e

    actual = _sha256(downloaded)
    if actual != spec.sha256:
        raise ModelError(
            f"{downloaded} has sha256 {actual}, expected {spec.sha256}; "
            "delete the file to download it again"
        )
    return downloaded


class _MutedFor:
    """sys.stderr with one thread's writes dropped and every other thread's passed on."""

    def __init__(self, target, thread: int):
        self.target = target
        self._thread = thread

    def write(self, s: str) -> int:
        if threading.get_ident() == self._thread:
            return len(s)
        return self.target.write(s)

    def __getattr__(self, name: str):
        return getattr(self.target, name)


def _stderr():
    """sys.stderr, past a load's muting: llama.cpp's errors must still reach it."""
    out = sys.stderr
    return out.target if isinstance(out, _MutedFor) else out


# A swap restored out of order would leave a muted stream installed, so loads take turns.
_QUIET = threading.Lock()


@contextmanager
def _quiet_load() -> Iterator[None]:
    """Drop what llama-cpp-python prints to sys.stderr while this thread loads a model.

    With verbose=True that is about 17 KB a load, the chat template twice, and every
    MCP ingest_file is a job of its own. Python-level only: no file descriptor is
    touched, since fd 1 is the MCP channel, and the server's other threads keep writing.
    """
    with _QUIET:
        target = sys.stderr
        sys.stderr = _MutedFor(target, threading.get_ident())
        try:
            yield
        finally:
            sys.stderr = target


class LlamaChoiceModel:
    def __init__(self, path: Path, temperature: float):
        try:
            import llama_cpp
            import numpy  # noqa: F401
            from llama_cpp import Llama
        except ImportError as e:
            raise ModelError(
                f"a classify dependency is present but failed to import: {e}; {INSTALL_HINT}"
            ) from e

        # After the import: importing Llama installs the library's own, noisy callback.
        _route_llama_logs(llama_cpp)
        self._temperature = temperature
        try:
            with _quiet_load():
                self._llm = Llama(
                    model_path=str(path),
                    n_ctx=N_CTX,
                    n_gpu_layers=-1,
                    n_batch=2048,
                    verbose=True,
                )
        except Exception as e:
            raise ModelError(f"could not load the classify model {path}: {e}") from e

        self._letter_tokens: list[int] = []
        for letter in LETTERS:
            ids = self._llm.tokenize(letter.encode(), add_bos=False, special=False)
            if len(ids) != 1:
                self.close()
                raise ModelError(
                    f"letter {letter} is {len(ids)} tokens in this model's vocabulary, not 1"
                )
            self._letter_tokens.append(ids[0])

    def probabilities(
        self, evidence: str, question: str, options: Sequence[tuple[str, str]]
    ) -> dict[str, float]:
        import llama_cpp
        import numpy as np

        if not 0 < len(options) <= len(LETTERS):
            raise ModelError(f"{len(options)} options; between 1 and {len(LETTERS)} are supported")
        prompt = build_prompt(evidence, question, options)
        tokens = self._llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) >= N_CTX:
            raise ModelError(f"the prompt is {len(tokens)} tokens, over the {N_CTX}-token context")

        self._llm.reset()
        self._llm.eval(tokens)
        # Llama.scores stays empty unless logits_all is set; read the last position directly.
        ptr = llama_cpp.llama_get_logits_ith(self._llm.ctx, -1)
        logits = np.ctypeslib.as_array(ptr, shape=(self._llm.n_vocab(),)).astype(np.float64)
        shifted = logits - logits.max()
        logprobs = shifted - np.log(np.exp(shifted).sum())

        picked = [float(logprobs[self._letter_tokens[i]]) for i in range(len(options))]
        probs = calibrate(picked, self._temperature)
        return {vid: p for (vid, _), p in zip(options, probs, strict=True)}

    def close(self) -> None:
        self._llm.close()


def default_model_factory(spec: ModelSpec, cache_dir: Path) -> ChoiceModel:
    if not available():
        raise ModelError(INSTALL_HINT)
    return LlamaChoiceModel(resolve_model_file(spec, cache_dir), spec.temperature)
