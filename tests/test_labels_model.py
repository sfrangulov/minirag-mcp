import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import types
import weakref
from pathlib import Path

import pytest

from minirag_mcp.labels import model
from minirag_mcp.labels.model import (
    ModelError,
    available,
    build_prompt,
    calibrate,
    default_model_factory,
    resolve_model_file,
)
from minirag_mcp.labels.taxonomy import ModelSpec


def test_prompt_is_the_reference_format():
    p = build_prompt(
        "File name: a.docx\nhello", "What kind?", [("mail", "Mail."), ("minutes", "Minutes.")]
    )
    assert p.startswith("<|im_start|>system\nApply the supplied criterion")
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    user = p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    assert json.loads(user) == {
        "evidence": "File name: a.docx\nhello",
        "criterion": "What kind?",
        "options": [
            {"letter": "A", "description": "mail: Mail."},
            {"letter": "B", "description": "minutes: Minutes."},
        ],
    }


def test_prompt_keeps_non_ascii_unescaped():
    p = build_prompt("café", "Q?", [("a", "x")])
    assert "café" in p


def test_calibration_matches_the_reference_readout():
    z, t = [-0.1, -2.3, -5.0], 2.07
    w = [math.exp((v + 0.1) / t) for v in z]
    assert calibrate(z, t) == pytest.approx([x / sum(w) for x in w])


def test_available_is_false_without_the_extra(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert available() is False


def test_available_needs_both_modules(monkeypatch):
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: object() if name == "numpy" else None
    )
    assert available() is False


def test_a_missing_local_model_file_is_named(tmp_path):
    missing = tmp_path / "nope.gguf"
    with pytest.raises(ModelError, match="nope.gguf"):
        resolve_model_file(ModelSpec(temperature=1.0, path=missing), tmp_path)


def test_a_local_model_file_is_used_as_is(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x")
    assert resolve_model_file(ModelSpec(temperature=1.0, path=f), tmp_path) == f


def _fake_download(calls, content=b"weights"):
    def fake(repo_id, filename, *, revision=None, cache_dir=None, **kw):
        calls.append({"repo": repo_id, "file": filename, "revision": revision, "cache": cache_dir})
        out = Path(cache_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)
        return str(out)

    return fake


def _spec(sha):
    return ModelSpec(temperature=1.0, repo="o/r", file="m.gguf", revision="abc123", sha256=sha)


def test_a_downloaded_file_with_the_wrong_sha256_is_refused(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download([]))
    with pytest.raises(ModelError) as e:
        resolve_model_file(_spec("0" * 64), tmp_path)
    assert "0" * 64 in str(e.value)
    assert hashlib.sha256(b"weights").hexdigest() in str(e.value)


def test_a_pinned_download_lands_in_the_cache_dir(tmp_path, monkeypatch):
    import huggingface_hub

    calls: list[dict] = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download(calls))
    got = resolve_model_file(_spec(hashlib.sha256(b"weights").hexdigest()), tmp_path)
    assert calls == [
        {"repo": "o/r", "file": "m.gguf", "revision": "abc123", "cache": tmp_path / "classify"}
    ]
    assert got.read_bytes() == b"weights"


def _warm_cache_download(content=b"weights"):
    """hf_hub_download on a warm cache: the file is written once, then returned untouched."""

    def fake(repo_id, filename, *, revision=None, cache_dir=None, **kw):
        out = Path(cache_dir) / filename
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(content)
        return str(out)

    return fake


class _CountingHashlib:
    def __init__(self):
        self.digests = 0

    def sha256(self, *args):
        self.digests += 1
        return hashlib.sha256(*args)


@pytest.fixture
def counted_download(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _warm_cache_download())
    counter = _CountingHashlib()
    monkeypatch.setattr(model, "hashlib", counter)
    return counter


def test_a_verified_download_is_hashed_once_per_process_until_it_changes(
    tmp_path, counted_download
):
    """Every MCP ingest_file is a job of its own, and re-hashing 4.5 GB each time is not."""
    spec = _spec(hashlib.sha256(b"weights").hexdigest())

    first = resolve_model_file(spec, tmp_path)
    assert resolve_model_file(spec, tmp_path) == first
    assert counted_download.digests == 1

    st = first.stat()
    os.utime(first, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert resolve_model_file(spec, tmp_path) == first
    assert counted_download.digests == 2


def test_a_file_that_failed_verification_fails_it_again(tmp_path, counted_download):
    spec = _spec("0" * 64)
    for _ in range(2):
        with pytest.raises(ModelError, match="0" * 64):
            resolve_model_file(spec, tmp_path)


def test_the_default_factory_without_the_extra_says_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "available", lambda: False)
    with pytest.raises(ModelError, match=r"minirag-mcp\[classify\]"):
        default_model_factory(ModelSpec(temperature=1.0, path=tmp_path / "x.gguf"), tmp_path)


class _FakeLlama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def tokenize(self, text: bytes, add_bos: bool, special: bool) -> list[int]:
        return [text[0]]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_llama_cpp(monkeypatch):
    """Stands in for the [classify] extra, which the base environment does not have."""
    mod = types.ModuleType("llama_cpp")
    mod.made = []
    mod.callbacks = []

    def make(**kwargs):
        llm = _FakeLlama(**kwargs)
        mod.made.append(llm)
        return llm

    mod.Llama = make
    # A ctypes CFUNCTYPE instance wraps a Python function; the fake keeps it callable as is.
    mod.llama_log_callback = lambda fn: fn
    mod.llama_log_set = lambda callback, user_data: mod.callbacks.append(callback)
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return mod


def test_the_model_is_built_so_that_the_library_never_redirects_stdout(
    fake_llama_cpp, tmp_path, capfd
):
    """verbose=False makes llama-cpp-python point fds 1 and 2 at /dev/null around the
    load, process wide, which loses whatever the MCP server writes to stdout meanwhile."""
    model.LlamaChoiceModel(tmp_path / "m.gguf", 2.07)

    [llm] = fake_llama_cpp.made
    assert llm.kwargs["verbose"] is True
    assert len(fake_llama_cpp.callbacks) == 1
    assert capfd.readouterr().out == ""


def test_the_installed_log_callback_forwards_errors_to_stderr_and_drops_the_rest(
    fake_llama_cpp, tmp_path, capfd
):
    model.LlamaChoiceModel(tmp_path / "m.gguf", 2.07)
    [callback] = fake_llama_cpp.callbacks

    for level, text in [
        (1, b"llama_model_loader: loaded meta data\n"),
        (5, b"more of the same\n"),
        (2, b"n_ctx_per_seq < n_ctx_train\n"),
        (4, b"debug detail\n"),
        (3, b"error loading model: "),
        (5, b"truncated file\n"),
    ]:
        callback(level, text, None)

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == "error loading model: truncated file\n"


def test_the_log_callback_stays_referenced_while_the_library_can_call_it(fake_llama_cpp, tmp_path):
    """llama.cpp keeps only a raw pointer: a collected ctypes callback crashes the next log."""
    model.LlamaChoiceModel(tmp_path / "a.gguf", 2.07)
    model.LlamaChoiceModel(tmp_path / "b.gguf", 2.07)

    first, second = fake_llama_cpp.callbacks
    assert first is second
    ref = weakref.ref(first)
    del first, second
    fake_llama_cpp.callbacks.clear()
    gc.collect()
    assert ref() is not None
