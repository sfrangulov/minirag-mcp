"""Shared fixtures: deterministic fake embedder (no model download), offline DNS."""

from __future__ import annotations

import hashlib
import ipaddress
import math
import socket
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from minirag_mcp import security

PUBLIC_IP = "93.184.216.34"
# The name the redirect guard must refuse, and the address it stands for. The address
# is the one that makes this worth guarding: AWS/GCP/Azure instance metadata.
BLOCKED_HOST = "metadata.test"
BLOCKED_IP = "169.254.169.254"
# Served by the stand-in metadata endpoint. Its absence is how a test proves the
# fetch never happened, rather than merely that an exception was raised.
SECRET_BODY = "SECRET-METADATA-TOKEN"


class FakeEmbedder:
    """Deterministic 8-dim embeddings from sha256 — same text, same vector."""

    dim = 8
    model_name = "fake"

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [int.from_bytes(h[i : i + 4], "big") / 2**32 - 0.5 for i in range(0, 32, 4)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every hostname to one fixed public address, without a resolver.

    ingest_url validates the host before fetching, so a test that ingests a URL
    would otherwise depend on a working resolver and on what a real name happens
    to resolve to today.
    """

    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port or 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)


class _RedirectHandler(BaseHTTPRequestHandler):
    """Pages a redirect guard has to reason about: a plain document, a redirect to a
    blocked host, the blocked host's own content, and an arbitrarily long chain."""

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's naming
        self.server.hits.append(f"{self.headers.get('Host', '')}{self.path}")
        port = self.server.server_address[1]
        if self.path == "/doc":
            self._html(
                "<html><head><title>Public Page</title></head>"
                "<body><p>PUBLIC-BODY</p></body></html>"
            )
        elif self.path == "/redirect-to-metadata":
            self._redirect(f"http://{BLOCKED_HOST}:{port}/latest/meta-data/")
        elif self.path == "/latest/meta-data/":
            self._html(f"<html><body><p>{SECRET_BODY}</p></body></html>")
        elif self.path.startswith("/chain/"):
            n = int(self.path.rsplit("/", 1)[1])
            if n <= 0:
                self._redirect(f"http://public.test:{port}/doc")
            else:
                # Alternating public hostnames: a legitimate chain may cross hosts.
                self._redirect(f"http://hop-{'a' if n % 2 else 'b'}.test:{port}/chain/{n - 1}")
        else:
            self.send_error(404)

    def _html(self, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _redirect(self, target: str) -> None:
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args) -> None:  # keep pytest output clean
        pass


@dataclass(frozen=True)
class RedirectServer:
    port: int
    hits: list[str]

    def url(self, path: str, host: str = "public.test") -> str:
        return f"http://{host}:{self.port}{path}"


@pytest.fixture
def redirect_server(monkeypatch):
    """A real HTTP server on 127.0.0.1, reachable under a public and a blocked name.

    Everything the guard depends on is real here: a real requests session, real 302
    responses, real redirect following, the real check. Only name resolution is
    faked, and it has to be — the guard's verdict comes from the address a name
    resolves to, and a test may not touch a resolver. Two seams are patched, and
    they are patched to disagree on purpose: `security._resolve_host` (what the
    guard sees) reports a public address for every name but `metadata.test`, which
    it reports as 169.254.169.254; `socket.getaddrinfo` (what the socket sees) sends
    every `.test` name to the local server. That is what lets one process play both
    an allowed public site and cloud metadata without a network.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    server.hits = []
    threading.Thread(target=server.serve_forever, daemon=True).start()

    real_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if isinstance(host, str) and host.endswith(".test"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]
        return real_getaddrinfo(host, port, *args, **kwargs)

    def fake_resolve_host(host: str):
        return [ipaddress.ip_address(BLOCKED_IP if host == BLOCKED_HOST else PUBLIC_IP)]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(security, "_resolve_host", fake_resolve_host)
    try:
        yield RedirectServer(port=server.server_address[1], hits=server.hits)
    finally:
        server.shutdown()
        server.server_close()
