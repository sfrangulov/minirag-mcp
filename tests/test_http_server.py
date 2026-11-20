"""`serve`: every index of indexes.toml over streamable HTTP on this machine, behind a
bearer token, a loopback Host and a loopback Origin, with the URL choosing what a
connection may use."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
import uvicorn
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from starlette.middleware import Middleware

from http_helpers import AUTH, INITIALIZE, MCP_HEADERS, TOKEN, serving
from index_helpers import file_registry, run_meta
from minirag_mcp.config import ConfigError
from minirag_mcp.http_server import (
    LoopbackGuard,
    ensure_token,
    parse_connection,
    read_token,
    serve,
)
from minirag_mcp.registry import Connection, IndexRegistry
from minirag_mcp.security import LOOPBACK_NAMES
from minirag_mcp.server import create_app

NOT_FOUND = "Not Found: MCP is served at /mcp and /mcp/<index>"


def test_the_token_file_is_created_once_with_mode_0600(tmp_path):
    p = tmp_path / "cfg" / "token"
    # The usual umask: under 077 a file created without a mode would be 0600 anyway.
    previous = os.umask(0o022)
    try:
        t = ensure_token(p)
        again = ensure_token(p)
    finally:
        os.umask(previous)
    assert len(t) >= 32 and stat.S_IMODE(p.stat().st_mode) == 0o600 and again == t
    assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700


def test_a_token_file_other_users_can_read_is_refused(tmp_path):
    p = tmp_path / "token"
    p.write_text("a-token-someone-else-may-have-read\n", encoding="utf-8")
    p.chmod(0o644)
    with pytest.raises(ConfigError, match=re.escape(f"chmod 600 {p}")):
        ensure_token(p)
    with pytest.raises(ConfigError, match=re.escape(f"chmod 600 {p}")):
        read_token(p)


def test_an_empty_token_file_is_refused(tmp_path):
    p = tmp_path / "token"
    p.write_text("\n", encoding="utf-8")
    p.chmod(0o600)
    with pytest.raises(ConfigError, match=f"{re.escape(str(p))}: .*empty"):
        ensure_token(p)


@pytest.mark.parametrize(
    "path, qs, expected",
    [
        ("/mcp", "", Connection(None, None)),
        ("/mcp/", "", Connection(None, None)),
        ("/mcp/alpha", "", Connection("alpha", None)),
        ("/mcp/alpha", "allow=beta,%20gamma", Connection("alpha", frozenset({"beta", "gamma"}))),
        ("/mcp/alpha", "allow=beta&allow=gamma", Connection("alpha", frozenset({"beta", "gamma"}))),
        ("/mcp/alpha", "allow=", Connection("alpha", frozenset())),
        ("/mcp/alpha", "allow=,", Connection("alpha", frozenset())),
        ("/mcp/alpha/beta", "", None),
        ("/mcp/alpha/", "", None),
        ("/mcp//", "", None),
        ("/api", "", None),
        ("/mcpx", "", None),
    ],
)
def test_parse_connection(path, qs, expected):
    assert parse_connection(path, qs) == expected


async def _post(url: str, **headers: str) -> httpx.Response:
    async with httpx.AsyncClient() as http:
        return await http.post(url, json=INITIALIZE, headers={**MCP_HEADERS, **headers})


async def test_a_request_without_the_token_is_refused(site):
    r = await _post(f"{site}/mcp/alpha")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert r.text == (
        "Unauthorized: send Authorization: Bearer <the token in the server's token file>"
    )


async def test_a_wrong_token_is_refused(site):
    r = await _post(f"{site}/mcp/alpha", Authorization="Bearer not-the-token")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


async def test_a_foreign_origin_is_refused_even_with_the_token(site):
    r = await _post(f"{site}/mcp/alpha", Origin="http://evil.example", **AUTH)
    assert (r.status_code, r.text) == (403, "Forbidden: Origin is not a loopback origin")


async def test_an_opaque_origin_is_refused(site):
    r = await _post(f"{site}/mcp/alpha", Origin="null", **AUTH)
    assert (r.status_code, r.text) == (403, "Forbidden: Origin is not a loopback origin")


async def test_a_host_that_is_not_loopback_is_refused(site):
    port = urlsplit(site).port
    r = await _post(f"{site}/mcp/alpha", Host=f"evil.example:{port}", **AUTH)
    assert (r.status_code, r.text) == (421, "Misdirected Request: Host must be a loopback name")


@pytest.mark.parametrize(
    "origin", ["http://localhost:5173", "http://127.0.0.1", "http://[::1]:8080"]
)
async def test_a_loopback_origin_with_the_token_is_served(site, origin):
    r = await _post(f"{site}/mcp/alpha", Origin=origin, **AUTH)
    assert r.status_code == 200, r.text


async def test_the_bearer_scheme_is_matched_in_any_case(site):
    r = await _post(f"{site}/mcp/alpha", Authorization=f"bEaReR {TOKEN}")
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("path", ["/api", "/mcp/alpha/beta"])
async def test_an_unknown_path_is_not_found(site, path):
    r = await _post(f"{site}{path}", **AUTH)
    assert (r.status_code, r.text) == (404, NOT_FOUND)


async def test_the_url_names_the_default_and_the_allow_list(site):
    url = f"{site}/mcp/beta?allow=alpha"
    async with Client(StreamableHttpTransport(url, headers=AUTH)) as c:
        listed = (await c.call_tool("list_indexes", {})).data
        assert listed["default"] == "beta"
        assert {r["name"]: r["reachable"] for r in listed["indexes"]} == {
            "alpha": True,
            "beta": True,
            "gamma": False,
        }
        assert (await c.call_tool("status", {})).data["index"] == "beta"
        r = await c.call_tool("status", {"index": "gamma"}, raise_on_error=False)
        assert r.is_error and "not allowed" in r.content[0].text


async def test_an_allow_list_in_the_url_keeps_every_other_index_out_of_reach(site):
    url = f"{site}/mcp/alpha?allow=beta"
    async with Client(StreamableHttpTransport(url, headers=AUTH)) as c:
        allowed = await c.call_tool("query_documents", {"query": "bananas", "index": "beta"})
        refused = await c.call_tool(
            "query_documents", {"query": "grapes", "index": "gamma"}, raise_on_error=False
        )
    assert allowed.data["index"] == "beta"
    assert refused.is_error and refused.content[0].text == (
        "Index 'gamma' is not allowed on this connection (allow=beta). "
        "Reachable indexes: alpha, beta."
    )


async def test_mcp_without_an_index_has_no_default(site):
    async with Client(StreamableHttpTransport(f"{site}/mcp", headers=AUTH)) as c:
        r = await c.call_tool("status", {}, raise_on_error=False)
        named = (await c.call_tool("status", {"index": "gamma"})).data
    assert r.is_error and r.content[0].text == (
        "This connection has no default index, so name one with `index`. "
        "Reachable indexes: alpha, beta, gamma."
    )
    assert named["index"] == "gamma"


async def test_a_request_the_path_middleware_did_not_see_reaches_no_index(
    indexes_file, fake_embedder, index_states
):
    registry = file_registry(indexes_file, lambda c, s: fake_embedder)
    # Without a connection, create_app's own reaches every index: only the request decides.
    app = create_app(registry).http_app(
        path="/mcp",
        transport="http",
        middleware=[Middleware(LoopbackGuard, token=TOKEN, allowed_hosts=LOOPBACK_NAMES)],
    )
    async with (
        serving(app) as base,
        Client(StreamableHttpTransport(f"{base}/mcp", headers=AUTH)) as c,
    ):
        named = await c.call_tool("status", {"index": "alpha"}, raise_on_error=False)
        unnamed = await c.call_tool("status", {}, raise_on_error=False)
    assert named.is_error and named.content[0].text == (
        "Index 'alpha' is not allowed on this connection (allow=). Reachable indexes: none."
    )
    assert unnamed.is_error and unnamed.content[0].text == (
        "This connection has no default index, so name one with `index`. Reachable indexes: none."
    )


@pytest.fixture
def started(monkeypatch) -> list[uvicorn.Config]:
    """The configurations uvicorn was asked to serve, none of them served."""
    configs: list[uvicorn.Config] = []

    async def serve_nothing(self, sockets=None) -> None:
        configs.append(self.config)

    monkeypatch.setattr(uvicorn.Server, "serve", serve_nothing)
    return configs


def test_serve_lets_a_call_outside_any_request_reach_no_index(indexes_file, started, capsys):
    serve(indexes_file.path, indexes_file.env)
    [config] = started
    token = indexes_file.path.parent / "token"
    assert (config.host, config.port) == ("127.0.0.1", 8765)
    assert capsys.readouterr().err.splitlines()[-1] == (
        "minirag-mcp serve: http://127.0.0.1:8765/mcp/<index> for alpha, beta, gamma; "
        f"token in {token}"
    )

    async def outside_any_request() -> tuple:
        async with Client(config.app.state.fastmcp_server) as c:
            return (
                await c.call_tool("status", {}, raise_on_error=False),
                await c.call_tool("status", {"index": "alpha"}, raise_on_error=False),
            )

    unnamed, named = asyncio.run(outside_any_request())
    assert unnamed.is_error and "Reachable indexes: none." in unnamed.content[0].text
    assert named.is_error and "not allowed on this connection" in named.content[0].text


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_sighup_reloads_the_file_off_the_event_loop(indexes_file, monkeypatch):
    reloads: list[threading.Thread] = []
    monkeypatch.setattr(
        IndexRegistry, "reload", lambda self: reloads.append(threading.current_thread())
    )

    async def hang_up(self, sockets=None) -> None:
        os.kill(os.getpid(), signal.SIGHUP)
        deadline = time.monotonic() + 10
        while not reloads and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

    monkeypatch.setattr(uvicorn.Server, "serve", hang_up)
    # Should serve register no handler of its own, this one keeps SIGHUP from ending pytest.
    caught: list[int] = []
    previous = signal.signal(signal.SIGHUP, lambda sig, frame: caught.append(sig))
    try:
        serve(indexes_file.path, indexes_file.env)
    finally:
        signal.signal(signal.SIGHUP, previous)
    assert caught == [] and len(reloads) == 1
    assert reloads[0] is not threading.main_thread()


def test_serve_refuses_a_non_loopback_host(indexes_file, started, capsys, monkeypatch):
    monkeypatch.setenv("HOME", indexes_file.env["HOME"])
    indexes_file.write('[server]\nhost = "0.0.0.0"\n\n' + indexes_file.text)
    code, out, err = run_meta(["--config", str(indexes_file.path), "serve"], capsys)
    assert (code, out, started) == (1, "", [])
    assert "loopback" in err and "Traceback" not in err


def test_serve_without_indexes_toml_says_where_it_looked(tmp_path, started, capsys, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("MINIRAG_CONFIG", raising=False)
    code, out, err = run_meta(["serve"], capsys)
    expected = tmp_path / ".config" / "minirag" / "indexes.toml"
    assert (code, out, started) == (1, "", [])
    assert err == (
        f"error: no indexes.toml at {expected}; create one "
        "(docs/launchd/local.minirag-mcp.plist shows how to run it)\n"
    )


def test_serve_refuses_index_rather_than_ignoring_it(indexes_file, started, capsys, monkeypatch):
    monkeypatch.setenv("HOME", indexes_file.env["HOME"])
    code, out, err = run_meta(
        ["--config", str(indexes_file.path), "--index", "alpha", "serve"], capsys
    )
    assert (code, out, started) == (1, "", [])
    assert "--index" in err and "every index" in err


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait(until, proc: subprocess.Popen, log: Path, what: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not until():
        if proc.poll() is not None:
            pytest.fail(f"serve exited with {proc.returncode} before {what}:\n{log.read_text()}")
        if time.monotonic() >= deadline:
            pytest.fail(f"no {what} within {timeout:g}s:\n{log.read_text()}")
        time.sleep(0.05)


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _refusal(url: str, token: str) -> str:
    """What `query_documents` without `index` is told at `url`."""

    async def call() -> str:
        auth = {"Authorization": f"Bearer {token}"}
        async with Client(StreamableHttpTransport(url, headers=auth)) as c:
            r = await c.call_tool("query_documents", {"query": "x"}, raise_on_error=False)
        assert r.is_error
        return r.content[0].text

    return asyncio.run(call())


def test_serve_requires_the_token_and_reloads_on_sighup(indexes_file, tmp_path):
    port = _free_port()
    indexes_file.write(f"[server]\nport = {port}\n\n" + indexes_file.text)
    log = tmp_path / "serve.log"
    base = f"http://127.0.0.1:{port}"
    with log.open("wb") as stderr:
        proc = subprocess.Popen(
            [sys.executable, "-m", "minirag_mcp", "--config", str(indexes_file.path), "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            env={**os.environ, "HOME": str(tmp_path)},
            start_new_session=True,
        )
    try:
        _wait(lambda: _listening(port), proc, log, f"listener on port {port}")
        token_file = indexes_file.path.parent / "token"
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        assert httpx.post(f"{base}/mcp", json=INITIALIZE, headers=MCP_HEADERS).status_code == 401

        token = read_token(token_file)
        listed = "Reachable indexes: alpha, beta, gamma."
        assert listed in _refusal(f"{base}/mcp", token)
        assert "reloaded" not in log.read_text()

        proc.send_signal(signal.SIGHUP)
        _wait(lambda: "reloaded" in log.read_text(), proc, log, "reload after SIGHUP")
        assert listed in _refusal(f"{base}/mcp", token)
        assert proc.poll() is None

        (tmp_path / "corpora" / "delta").mkdir()
        indexes_file.write(
            indexes_file.path.read_text(encoding="utf-8")
            + '\n[index.delta]\nroots = ["~/corpora/delta"]\n'
        )
        assert "Reachable indexes: alpha, beta, gamma, delta." in _refusal(f"{base}/mcp", token)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    assert f"minirag-mcp serve: {base}/mcp/<index> for alpha, beta, gamma" in log.read_text()
