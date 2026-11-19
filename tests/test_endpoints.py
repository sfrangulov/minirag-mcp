import re

import pytest

from conftest import FakeEmbedder
from minirag_mcp.endpoints import (
    EmbeddingEndpoint,
    Endpoint,
    EndpointError,
    RerankEndpoint,
    read_api_key,
)


def ep(fake_llama, path="/v1/embeddings", key=None):
    return Endpoint(fake_llama.url(path), "e5", key)


def test_embed_returns_unit_vectors_in_input_order(fake_llama):
    got = EmbeddingEndpoint(ep(fake_llama)).embed(["a", "b", "c"])
    assert got == [pytest.approx(FakeEmbedder()._vec(t)) for t in "abc"]


def test_embed_batches_and_names_the_model(fake_llama):
    EmbeddingEndpoint(ep(fake_llama)).embed([f"t{i}" for i in range(70)])
    assert [len(r.body["input"]) for r in fake_llama.requests] == [32, 32, 6]
    assert {r.body["model"] for r in fake_llama.requests} == {"e5"}


def test_a_bearer_header_is_sent_only_with_a_key(fake_llama):
    EmbeddingEndpoint(ep(fake_llama, key="k")).embed(["a"])
    EmbeddingEndpoint(ep(fake_llama)).embed(["a"])
    with_key, without = fake_llama.requests
    assert with_key.headers["Authorization"] == "Bearer k"
    assert "Authorization" not in without.headers


def test_no_texts_make_no_request(fake_llama):
    assert EmbeddingEndpoint(ep(fake_llama)).embed([]) == []
    assert fake_llama.requests == []


def test_an_unreachable_endpoint_names_the_url(dead_url):
    with pytest.raises(EndpointError, match=re.escape(dead_url)):
        EmbeddingEndpoint(Endpoint(dead_url + "/v1/embeddings", "e5")).embed(["a"])


def test_an_http_error_carries_status_and_body(fake_llama):
    fake_llama.fail["/v1/embeddings"] = (500, "boom")
    with pytest.raises(EndpointError, match=r"500.*boom"):
        EmbeddingEndpoint(ep(fake_llama)).embed(["a"])


def test_a_reply_that_is_not_json_is_an_endpoint_error(fake_llama):
    fake_llama.html.add("/v1/embeddings")
    with pytest.raises(EndpointError, match="JSON"):
        EmbeddingEndpoint(ep(fake_llama)).embed(["a"])


def test_a_short_reply_is_an_endpoint_error(fake_llama):
    fake_llama.drop_last = True
    with pytest.raises(EndpointError):
        EmbeddingEndpoint(ep(fake_llama)).embed(["a", "b"])


def test_proxy_variables_are_ignored(fake_llama, monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert len(EmbeddingEndpoint(ep(fake_llama)).embed(["a"])) == 1


def test_rerank_scores_come_back_in_input_order_unnormalized(fake_llama):
    rr = RerankEndpoint(ep(fake_llama, "/v1/rerank"))
    assert rr.scores("x y", ["x y z", "q", "x"]) == [0.0, -2.0, -1.0]
    assert fake_llama.requests[0].body["top_n"] == 3
    assert rr.last_error is None


def test_a_rerank_failure_is_remembered_until_a_success(fake_llama):
    fake_llama.fail["/v1/rerank"] = (503, "loading")
    rr = RerankEndpoint(ep(fake_llama, "/v1/rerank"))
    with pytest.raises(EndpointError):
        rr.scores("q", ["a"])
    assert "503" in rr.last_error
    fake_llama.fail.clear()
    rr.scores("q", ["a"])
    assert rr.last_error is None


def test_read_api_key(tmp_path):
    (tmp_path / "k").write_text("\n  secret \nother\n")
    assert read_api_key(tmp_path / "k") == "secret"
    assert read_api_key(None) is None
    (tmp_path / "blank").write_text(" \n")
    for path in (tmp_path / "missing", tmp_path / "blank"):
        with pytest.raises(EndpointError, match=re.escape(str(path))):
            read_api_key(path)
