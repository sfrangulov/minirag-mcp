"""EndpointChoiceModel: the choice model served by llama-server, against the fake router."""

from __future__ import annotations

import hashlib
import math

import pytest

from minirag_mcp.endpoints import Endpoint
from minirag_mcp.labels.endpoint_model import EndpointChoiceModel, endpoint_model_factory
from minirag_mcp.labels.model import N_CTX, ModelError, build_prompt, calibrate
from minirag_mcp.labels.taxonomy import ModelSpec

OPTS = [("a", "A"), ("b", "B"), ("c", "C")]
TOP = [
    {"id": 32, "token": "A", "logprob": -0.1},
    {"id": 33, "token": "B", "logprob": -2.5},
    {"id": 7, "token": "x", "logprob": -3.0},
    {"id": 34, "token": "C", "logprob": -4.0},
]


@pytest.fixture
def pinned(fake_llama, tmp_path) -> ModelSpec:
    path = tmp_path / "plumb.gguf"
    path.write_bytes(b"gguf")
    fake_llama.props = {"model_path": str(path)}
    return ModelSpec(temperature=2.07, path=path)


def model(fake_llama, pinned, **kw) -> EndpointChoiceModel:
    return EndpointChoiceModel(
        Endpoint(fake_llama.url(), "plumb", "k"), 2.07, expected=pinned, **kw
    )


def test_probabilities_equal_the_in_process_readout(fake_llama, pinned):
    fake_llama.top_logprobs = TOP
    got = model(fake_llama, pinned).probabilities("ev", "q?", OPTS)
    assert got == pytest.approx(dict(zip("abc", calibrate([-0.1, -2.5, -4.0], 2.07), strict=True)))


def test_the_prompt_goes_as_token_ids_with_the_measured_settings(fake_llama, pinned):
    fake_llama.top_logprobs = TOP
    model(fake_llama, pinned).probabilities("ev", "q?", OPTS)
    tok = [r for r in fake_llama.requests if r.path == "/tokenize"][-1].body
    assert tok == {
        "content": build_prompt("ev", "q?", OPTS),
        "add_special": False,
        "parse_special": True,
        "model": "plumb",
    }
    done = next(r for r in fake_llama.requests if r.path == "/completion").body
    assert {k: v for k, v in done.items() if k != "prompt"} == {
        "n_predict": 1,
        "n_probs": 100,
        "post_sampling_probs": False,
        "cache_prompt": False,
        "temperature": 0,
        "samplers": [],
        "model": "plumb",
    }
    assert all(isinstance(t, int) for t in done["prompt"])


def test_letters_are_matched_by_id_not_by_text(fake_llama, pinned):
    fake_llama.top_logprobs = [{"id": 999, "token": "A", "logprob": -0.01}, *TOP]
    got = model(fake_llama, pinned).probabilities("ev", "q?", OPTS)
    assert got == pytest.approx(dict(zip("abc", calibrate([-0.1, -2.5, -4.0], 2.07), strict=True)))


def test_a_letter_outside_the_top_n_gets_zero(fake_llama, pinned):
    fake_llama.top_logprobs = TOP[:2]
    got = model(fake_llama, pinned).probabilities("ev", "q?", OPTS)
    assert got["c"] == 0.0
    assert math.isclose(sum(got.values()), 1.0)


def test_no_letter_in_the_top_n_is_a_model_error(fake_llama, pinned):
    fake_llama.top_logprobs = [{"id": 7, "token": "x", "logprob": -1.0}]
    with pytest.raises(ModelError, match="raise n_probs"):
        model(fake_llama, pinned).probabilities("ev", "q?", OPTS)


def test_a_prompt_over_the_context_is_refused_before_completion(fake_llama, pinned):
    m = model(fake_llama, pinned)
    fake_llama.token_count = N_CTX
    with pytest.raises(ModelError, match=str(N_CTX)):
        m.probabilities("ev", "q?", OPTS)
    assert "/completion" not in [r.path for r in fake_llama.requests]


def test_every_request_names_the_router_model_and_carries_the_key(fake_llama, pinned):
    fake_llama.top_logprobs = TOP
    model(fake_llama, pinned).probabilities("ev", "q?", OPTS)
    posts = [r for r in fake_llama.requests if r.method == "POST"]
    assert posts and all(r.body["model"] == "plumb" for r in posts)
    assert all(r.headers.get("Authorization") == "Bearer k" for r in fake_llama.requests)
    props = next(r for r in fake_llama.requests if r.path == "/props")
    assert props.query == "model=plumb"


def test_the_served_file_must_be_the_pinned_one(fake_llama, tmp_path):
    served = tmp_path / "other.gguf"
    served.write_bytes(b"other bytes")
    fake_llama.props = {"model_path": str(served)}
    pin = hashlib.sha256(b"gguf").hexdigest()
    with pytest.raises(ModelError) as err:
        EndpointChoiceModel(
            Endpoint(fake_llama.url(), "plumb"), 2.07, expected=ModelSpec(2.07, sha256=pin)
        )
    assert str(served) in str(err.value) and pin in str(err.value)

    served.write_bytes(b"gguf")
    EndpointChoiceModel(
        Endpoint(fake_llama.url(), "plumb"), 2.07, expected=ModelSpec(2.07, sha256=pin)
    )


def test_a_served_path_other_than_the_pinned_path_is_refused(fake_llama, pinned, tmp_path):
    other = tmp_path / "other.gguf"
    other.write_bytes(b"gguf")
    fake_llama.props = {"model_path": str(other)}
    with pytest.raises(ModelError, match="other.gguf"):
        model(fake_llama, pinned)


def test_props_without_a_model_path_is_refused(fake_llama, pinned):
    fake_llama.props = {}
    with pytest.raises(ModelError, match="model_path"):
        model(fake_llama, pinned)


def test_a_served_file_that_cannot_be_read_is_refused(fake_llama, tmp_path):
    fake_llama.props = {"model_path": str(tmp_path / "missing.gguf")}
    with pytest.raises(ModelError, match="missing.gguf"):
        EndpointChoiceModel(
            Endpoint(fake_llama.url(), "plumb"), 2.07, expected=ModelSpec(2.07, sha256="0" * 64)
        )


def test_an_unreachable_server_is_a_model_error(dead_url, pinned):
    with pytest.raises(ModelError, match=dead_url):
        EndpointChoiceModel(Endpoint(dead_url, "plumb"), 2.07, expected=pinned)


def test_a_reply_without_token_ids_is_a_model_error(fake_llama, pinned):
    fake_llama.replies["/tokenize"] = {"tokens": "nope"}
    with pytest.raises(ModelError, match="token ids"):
        model(fake_llama, pinned)


def test_a_letter_that_is_two_tokens_is_refused(fake_llama, pinned):
    fake_llama.split_letters = True
    with pytest.raises(ModelError, match="not 1"):
        model(fake_llama, pinned)


def test_the_factory_pins_the_spec(fake_llama, pinned):
    m = endpoint_model_factory(Endpoint(fake_llama.url(), "plumb"))(pinned, pinned.path.parent)
    assert isinstance(m, EndpointChoiceModel)
    m.close()
