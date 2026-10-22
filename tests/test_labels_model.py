import hashlib
import importlib.util
import json
import math
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


def test_the_default_factory_without_the_extra_says_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "available", lambda: False)
    with pytest.raises(ModelError, match=r"minirag-mcp\[classify\]"):
        default_model_factory(ModelSpec(temperature=1.0, path=tmp_path / "x.gguf"), tmp_path)
