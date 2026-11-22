"""`minirag-mcp dashboard`: asks the running server for a one-time code and opens the page
signed in, never showing the token."""

from __future__ import annotations

import asyncio
import re
import socket
import threading
import webbrowser

import httpx
import pytest

import minirag_mcp.cli as cli
from http_helpers import serving
from index_helpers import file_registry, run_meta
from minirag_mcp.http_server import build_http_app, ensure_token
from minirag_mcp.security import LOOPBACK_NAMES
from minirag_mcp.server import create_app


@pytest.fixture(autouse=True)
def fake_model(monkeypatch, fake_embedder):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)


def _set_port(indexes_file, port: int) -> None:
    text = indexes_file.path.read_text(encoding="utf-8")
    assert "[server]" not in text
    indexes_file.path.write_text(f"[server]\nport = {port}\n\n{text}", encoding="utf-8")


@pytest.fixture
async def served(indexes_file, fake_embedder, index_states, monkeypatch):
    """The dashboard served on a free port that indexes.toml names, with the token its token
    file holds."""
    monkeypatch.setenv("HOME", indexes_file.env["HOME"])
    token = ensure_token(indexes_file.path.parent / "token")
    registry = file_registry(indexes_file, lambda c, s: fake_embedder)
    app = build_http_app(
        create_app(registry), token=token, allowed_hosts=LOOPBACK_NAMES, registry=registry
    )
    async with serving(app) as base:
        _set_port(indexes_file, int(base.rsplit(":", 1)[1]))
        yield base, token


async def test_dashboard_prints_a_signed_in_url_without_the_token(served, indexes_file, capsys):
    base, token = served
    argv = ["--config", str(indexes_file.path), "dashboard", "--no-open"]
    code, out, err = await asyncio.to_thread(run_meta, argv, capsys)
    assert code == 0, err
    url = out.strip()
    assert re.fullmatch(rf"{re.escape(base)}/dashboard/login\?code=[\w-]+", url)
    assert token not in out + err
    async with httpx.AsyncClient(follow_redirects=True) as http:
        r = await http.get(url)
    assert r.status_code == 200
    assert "alpha" in r.text and "beta" in r.text


async def test_dashboard_opens_the_browser(served, indexes_file, capsys, monkeypatch):
    base, token = served
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    argv = ["--config", str(indexes_file.path), "dashboard"]
    code, out, err = await asyncio.to_thread(run_meta, argv, capsys)
    assert code == 0, err
    assert out.strip() == "Opened the minirag dashboard in your browser."
    assert len(opened) == 1 and opened[0].startswith(f"{base}/dashboard/login?code=")
    assert token not in out + err


@pytest.fixture
def proxy(monkeypatch):
    """A listener configured as the HTTP proxy of every scheme, with no exemption: what it
    was sent, connection by connection."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.05)
    received: list[bytes] = []
    stop = threading.Event()

    def accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            with conn:
                conn.settimeout(2)
                try:
                    received.append(conn.recv(65536))
                except OSError:
                    received.append(b"")

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, url)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    yield received
    stop.set()
    thread.join(5)
    listener.close()


async def test_dashboard_never_sends_the_token_to_a_proxy(served, indexes_file, capsys, proxy):
    base, token = served
    argv = ["--config", str(indexes_file.path), "dashboard", "--no-open"]
    code, out, err = await asyncio.to_thread(run_meta, argv, capsys)
    assert proxy == []
    assert code == 0, err
    url = out.strip()
    assert url.startswith(f"{base}/dashboard/login?code=")
    async with httpx.AsyncClient(follow_redirects=True, trust_env=False) as http:
        assert (await http.get(url)).status_code == 200


def test_dashboard_without_a_server_fails_in_one_line(indexes_file, capsys, monkeypatch):
    monkeypatch.setenv("HOME", indexes_file.env["HOME"])
    ensure_token(indexes_file.path.parent / "token")
    _set_port(indexes_file, 1)
    code, out, err = run_meta(["--config", str(indexes_file.path), "dashboard"], capsys)
    assert code == 1 and out == ""
    assert err.strip() == (
        "error: No minirag server answers at http://127.0.0.1:1. Start it with: minirag-mcp serve"
    )
