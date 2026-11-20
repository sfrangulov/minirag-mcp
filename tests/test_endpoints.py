import pickle
import re
import socket

import pytest

import minirag_mcp.endpoints as endpoints
from conftest import FakeEmbedder
from minirag_mcp.endpoints import (
    EmbeddingEndpoint,
    Endpoint,
    EndpointError,
    RerankEndpoint,
    read_api_key,
    request_json,
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
    with pytest.raises(EndpointError, match=re.escape(dead_url)) as raised:
        EmbeddingEndpoint(Endpoint(dead_url + "/v1/embeddings", "e5")).embed(["a"])
    assert raised.value.timed_out is False


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
    with pytest.raises(EndpointError, match="returned 1 embeddings for 2 texts"):
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


def test_a_rerank_failure_is_remembered_until_a_success(fake_llama, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(endpoints.time, "monotonic", lambda: now[0])
    fake_llama.fail["/v1/rerank"] = (503, "loading")
    rr = RerankEndpoint(ep(fake_llama, "/v1/rerank"))
    with pytest.raises(EndpointError):
        rr.scores("q", ["a"])
    assert "503" in rr.last_error
    fake_llama.fail.clear()
    with pytest.raises(EndpointError, match="not retried"):
        rr.scores("q", ["a"])
    now[0] += endpoints.RERANK_RETRY_SECONDS
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


def test_request_json_gets_when_there_is_no_body(fake_llama):
    fake_llama.props = {"n_ctx": 512}
    got = request_json(ep(fake_llama), fake_llama.url("/props"))
    assert got == {"n_ctx": 512}
    assert [(r.method, r.path) for r in fake_llama.requests] == [("GET", "/props")]


def test_a_silent_endpoint_times_out_naming_the_url():
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)
        url = f"http://127.0.0.1:{silent.getsockname()[1]}/v1/embeddings"
        with pytest.raises(EndpointError, match=re.escape(url)) as raised:
            EmbeddingEndpoint(Endpoint(url, "e5", timeout=0.3)).embed(["a"])
    assert raised.value.timed_out is True


def test_an_embeddings_reply_without_data_is_an_endpoint_error(fake_llama):
    fake_llama.replies["/v1/embeddings"] = {"object": "list"}
    with pytest.raises(EndpointError, match="unexpected embeddings reply"):
        EmbeddingEndpoint(ep(fake_llama)).embed(["a"])


def test_a_rerank_index_out_of_range_is_an_endpoint_error(fake_llama):
    fake_llama.replies["/v1/rerank"] = {"results": [{"index": 5, "relevance_score": 1.0}]}
    rr = RerankEndpoint(ep(fake_llama, "/v1/rerank"))
    with pytest.raises(EndpointError, match="no score for passage 0"):
        rr.scores("q", ["a"])
    assert "no score" in rr.last_error


def test_a_closed_client_opens_a_new_session_on_next_use(fake_llama):
    client = EmbeddingEndpoint(ep(fake_llama))
    client.embed(["a"])
    first = client._session
    client.close()
    assert client._session is None
    assert len(client.embed(["a"])) == 1
    assert client._session is not first


def test_clients_survive_pickling(fake_llama):
    embedding = EmbeddingEndpoint(ep(fake_llama))
    before = embedding.embed(["a", "b"])  # the session now exists
    clone = pickle.loads(pickle.dumps(embedding))
    assert clone.embed(["a", "b"]) == before and clone.endpoint == embedding.endpoint

    rerank = RerankEndpoint(ep(fake_llama, "/v1/rerank"))
    scores = rerank.scores("alpha beta", ["alpha", "gamma"])
    assert pickle.loads(pickle.dumps(rerank)).scores("alpha beta", ["alpha", "gamma"]) == scores
