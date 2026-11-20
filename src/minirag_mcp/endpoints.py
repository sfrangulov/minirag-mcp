"""HTTP clients for the routes of an OpenAI-style llama.cpp router."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import requests


class EndpointError(Exception):
    """A call to an endpoint failed; the message names the URL and the cause. `timed_out`:
    it accepted the connection but did not answer within the timeout. A busy server still
    completes the handshake in the kernel, so a connect timeout is not that."""

    def __init__(self, *args: object, timed_out: bool = False) -> None:
        super().__init__(*args)
        self.timed_out = timed_out


@dataclass(frozen=True)
class Endpoint:
    url: str
    name: str
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 120.0


EMBED_BATCH = 32
RERANK_TIMEOUT = 10.0
RERANK_RETRY_SECONDS = 60.0
_BODY_CHARS = 300


def _session() -> requests.Session:
    session = requests.Session()
    # Keeps HTTP(S)_PROXY from routing a router's traffic through a corporate proxy.
    session.trust_env = False
    return session


def read_api_key(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EndpointError(f"cannot read the API key file {path}: {exc}") from exc
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    raise EndpointError(f"the API key file {path} is empty")


def request_json(
    endpoint: Endpoint,
    url: str,
    body: dict | None = None,
    *,
    session: requests.Session | None = None,
) -> dict:
    if session is None:
        with _session() as own:
            return request_json(endpoint, url, body, session=own)
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    try:
        if body is None:
            response = session.get(url, headers=headers, timeout=endpoint.timeout)
        else:
            payload = {**body, "model": endpoint.name}
            response = session.post(url, json=payload, headers=headers, timeout=endpoint.timeout)
    except requests.RequestException as exc:
        timed_out = isinstance(exc, requests.ReadTimeout)
        raise EndpointError(f"cannot reach {url}: {exc}", timed_out=timed_out) from exc
    status = response.status_code
    if not response.ok:
        raise EndpointError(f"{url} answered {status}: {response.text[:_BODY_CHARS]}")
    try:
        data = response.json()
    except ValueError as exc:
        raise EndpointError(f"{url} answered {status} but the reply was not JSON") from exc
    if not isinstance(data, dict):
        raise EndpointError(f"{url} answered {status} but the JSON reply was not an object")
    return data


class _Client:
    def __init__(self, endpoint: Endpoint) -> None:
        self.endpoint = endpoint
        self._session: requests.Session | None = None
        # A sync thread and query threads may share one client.
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_lock"]
        state["_session"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _post(self, body: dict, timeout: float | None = None) -> dict:
        with self._lock:
            if self._session is None:
                self._session = _session()
            session = self._session
        endpoint = self.endpoint
        if timeout is not None:
            endpoint = replace(endpoint, timeout=min(endpoint.timeout, timeout))
        return request_json(endpoint, endpoint.url, body, session=session)

    def close(self) -> None:
        with self._lock:
            session, self._session = self._session, None
        if session is not None:
            session.close()


class EmbeddingEndpoint(_Client):
    def embed(self, texts: Sequence[str], *, timeout: float | None = None) -> list[list[float]]:
        """`timeout`, when given, shortens the endpoint's own for each request."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = list(texts[start : start + EMBED_BATCH])
            vectors += self._embed_batch(batch, timeout)
        return vectors

    def _embed_batch(self, batch: list[str], timeout: float | None) -> list[list[float]]:
        url = self.endpoint.url
        reply = self._post({"input": batch}, timeout)
        try:
            rows = sorted(reply["data"], key=lambda row: row["index"])
            if [row["index"] for row in rows] != list(range(len(batch))):
                raise EndpointError(f"{url} returned {len(rows)} embeddings for {len(batch)} texts")
            return [_unit([float(x) for x in row["embedding"]]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise EndpointError(f"{url} returned an unexpected embeddings reply: {exc!r}") from exc


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return vector if norm == 0 else [x / norm for x in vector]


class RerankEndpoint(_Client):
    """Reranking sits on the path of every query, so a hung server must cost one short
    wait and then nothing: after a failure the endpoint is not called for
    RERANK_RETRY_SECONDS."""

    def __init__(self, endpoint: Endpoint) -> None:
        super().__init__(replace(endpoint, timeout=min(endpoint.timeout, RERANK_TIMEOUT)))
        self.last_error: str | None = None
        self._failed_at: float | None = None

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        if self._failed_at is not None:
            wait = RERANK_RETRY_SECONDS - (time.monotonic() - self._failed_at)
            if wait > 0:
                raise EndpointError(f"{self.last_error}; not retried for another {wait:.0f} s")
        try:
            result = self._scores(query, list(passages))
        except EndpointError as exc:
            self.last_error = str(exc)
            self._failed_at = time.monotonic()
            raise EndpointError(f"{exc}; not retried for {RERANK_RETRY_SECONDS:g} s") from exc
        self.last_error = None
        self._failed_at = None
        return result

    def _scores(self, query: str, passages: list[str]) -> list[float]:
        url = self.endpoint.url
        reply = self._post({"query": query, "documents": passages, "top_n": len(passages)})
        try:
            by_index = {int(r["index"]): float(r["relevance_score"]) for r in reply["results"]}
        except (KeyError, TypeError, ValueError) as exc:
            raise EndpointError(f"{url} returned an unexpected rerank reply: {exc!r}") from exc
        missing = [i for i in range(len(passages)) if i not in by_index]
        if missing:
            raise EndpointError(f"{url} returned no score for passage {missing[0]}")
        return [by_index[i] for i in range(len(passages))]
