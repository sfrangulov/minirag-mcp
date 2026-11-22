"""`minirag-mcp serve`: every index of indexes.toml over streamable HTTP on this machine.

The server listens on a loopback address only (indexes.toml refuses any other), and every
request must pass `LoopbackGuard`: a loopback Host, no Origin but a loopback one, and the
bearer token of the token file. A web page the user visits can make the browser send
requests to 127.0.0.1, and the Host and Origin rules turn away the DNS-rebinding and
cross-site forms of that; the token turns away every local process that cannot read the
token file.

`/mcp/<index>` makes `<index>` the connection's default, `?allow=a,b` lets it name `a` and
`b` besides, and `/mcp` has no default; the connection reaches each tool through the ASGI
scope, under `CONNECTION_SCOPE_KEY`.

With a registry, the guard also serves the dashboard under `/dashboard`, whose pages a
browser opens with a session cookie in place of the token: `/dashboard/code` hands a
one-time code to a holder of the token, and `/dashboard/login?code=` spends it for the
cookie. The cookie opens the dashboard's pages and nothing else.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import os
import secrets
import signal
import sys
from collections.abc import Collection, Mapping
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import uvicorn
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import cookie_parser
from starlette.responses import PlainTextResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from minirag_mcp.config import ConfigError
from minirag_mcp.dashboard.app import (
    CODE_PATH,
    DASHBOARD_PATH,
    LOGIN_PATH,
    dashboard_app,
    is_dashboard_path,
    not_signed_in,
)
from minirag_mcp.dashboard.sessions import COOKIE, Sessions
from minirag_mcp.registry import CONNECTION_SCOPE_KEY, Connection, IndexRegistry
from minirag_mcp.security import LOOPBACK_NAMES, host_name
from minirag_mcp.server import create_app

MCP_PATH = "/mcp"
NOT_FOUND = "Not Found: MCP is served at /mcp and /mcp/<index>"
MISDIRECTED = "Misdirected Request: Host must be a loopback name"
FOREIGN_ORIGIN = "Forbidden: Origin is not a loopback origin"
UNAUTHORIZED = "Unauthorized: send Authorization: Bearer <the token in the server's token file>"
# What a call that arrives outside any HTTP request may use: nothing.
_CLOSED = Connection(None, frozenset())


def read_token(path: Path) -> str:
    """The token in `path`; ConfigError when other users may read the file, or it holds
    none."""
    try:
        with open(path, encoding="utf-8") as f:
            mode = os.fstat(f.fileno()).st_mode
            token = f.read().strip()
    except (OSError, UnicodeDecodeError) as e:
        why = e.strerror if isinstance(e, OSError) and e.strerror else e
        raise ConfigError(f"{path}: cannot read the token: {why}") from e
    # Windows reports no POSIX permission bits; the file's ACL guards it there.
    if os.name == "posix" and mode & 0o077:
        raise ConfigError(
            f"{path}: other users may read this token file; make it yours alone: chmod 600 {path}"
        )
    if not token:
        raise ConfigError(
            f"{path}: the token file is empty; delete it, and the next start writes a new token"
        )
    return token


def ensure_token(path: Path) -> str:
    """The token in `path`, written there first when the file does not exist: readable by
    this user alone, in a folder only this user may enter when it is created here."""
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return read_token(path)
    except OSError as e:
        raise ConfigError(f"{path}: cannot create the token file: {e.strerror or e}") from e
    unsaved = memoryview((secrets.token_urlsafe(32) + "\n").encode("ascii"))
    try:
        try:
            while unsaved:
                unsaved = unsaved[os.write(fd, unsaved) :]
        finally:
            os.close(fd)
    except OSError as e:
        # An empty or partial file would be refused, or read as a shorter token, at every
        # start after this one.
        with contextlib.suppress(OSError):
            path.unlink()
        raise ConfigError(f"{path}: cannot write the token file: {e.strerror or e}") from e
    return read_token(path)


def parse_connection(path: str, query_string: str) -> Connection | None:
    """What a request to `path` with `query_string` may use; None for a path that is not
    MCP's. The index the path names is not checked here: a call that uses it gets the
    registry's refusal, which names the indexes there are."""
    parts = path.split("/")
    if parts[:2] != ["", "mcp"] or len(parts) > 3:
        return None
    default = parts[2] if len(parts) == 3 and parts[2] else None
    # Blank values kept: a present but empty `allow=` allows no other index, not every one.
    values = parse_qs(query_string, keep_blank_values=True).get("allow")
    if values is None:
        return Connection(default, None)
    names = (name.strip() for value in values for name in value.split(","))
    return Connection(default, frozenset(name for name in names if name))


def _origin_host(origin: str) -> str:
    """The host of the one origin `origin` names; "" for an opaque, malformed or listed
    one."""
    if any(c.isspace() for c in origin):
        return ""
    try:
        authority = urlsplit(origin).netloc
    except ValueError:
        return ""
    return host_name(authority)


def _checked(token: str, allowed_hosts: Collection[str]) -> tuple[bytes, frozenset[str]]:
    if not token.strip():
        raise ValueError("the bearer token is empty: any request would carry it")
    hosts = frozenset(allowed_hosts)
    # host_name reads "" from a value it cannot read a host from.
    if "" in hosts:
        raise ValueError("an allowed host is empty: a Host or Origin without a host would pass")
    return token.encode("utf-8"), hosts


def _unauthorized() -> PlainTextResponse:
    return PlainTextResponse(UNAUTHORIZED, status_code=401, headers={"WWW-Authenticate": "Bearer"})


class LoopbackGuard:
    """Refuses an HTTP request from anywhere but this machine's own clients: one whose
    Host is not a loopback name, whose Origin is another site's or opaque, or that lacks
    the token.

    With `dashboard` and its `sessions`, a request under `/dashboard` goes to `dashboard`
    instead, and `/` is redirected there: its login needs only its code, its code needs
    the token, and every other page a session. No other path accepts a session."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str,
        allowed_hosts: Collection[str],
        dashboard: ASGIApp | None = None,
        sessions: Sessions | None = None,
    ) -> None:
        if (dashboard is None) != (sessions is None):
            raise ValueError("the dashboard and its sessions are given together or not at all")
        self.app = app
        self._token, self._hosts = _checked(token, allowed_hosts)
        self._dashboard = dashboard
        self._sessions = sessions

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # Closed before it is accepted, which the server answers with 403.
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            raise RuntimeError(f"minirag-mcp serve answers no {scope['type']!r} connection")
        answer = self._answer(scope["path"], Headers(scope=scope))
        await answer(scope, receive, send)

    def _answer(self, path: str, headers: Headers) -> ASGIApp:
        """What answers a request for `path`: a refusal, the dashboard or the app."""
        if host_name(headers.get("host", "")) not in self._hosts:
            return PlainTextResponse(MISDIRECTED, status_code=421)
        if "origin" in headers and _origin_host(headers["origin"]) not in self._hosts:
            return PlainTextResponse(FOREIGN_ORIGIN, status_code=403)
        if self._dashboard is not None:
            if path == "/":
                return RedirectResponse(DASHBOARD_PATH, status_code=303)
            if is_dashboard_path(path):
                return self._dashboard_answer(path, headers)
        if not self._authorized(headers.get("authorization", "")):
            return _unauthorized()
        return self.app

    def _dashboard_answer(self, path: str, headers: Headers) -> ASGIApp:
        if path == LOGIN_PATH:
            # Its code is the credential: one the token's holder asked for, spent here.
            return self._dashboard
        if path == CODE_PATH:
            authorized = self._authorized(headers.get("authorization", ""))
            return self._dashboard if authorized else _unauthorized()
        cookies = cookie_parser("; ".join(headers.getlist("cookie")))
        return self._dashboard if self._sessions.valid(cookies.get(COOKIE)) else not_signed_in()

    def _authorized(self, authorization: str) -> bool:
        scheme, _, credentials = authorization.partition(" ")
        # Headers arrive decoded as latin-1, so encoding them back gives the bytes sent.
        sent = credentials.strip().encode("latin-1")
        return scheme.lower() == "bearer" and hmac.compare_digest(sent, self._token)


class IndexPathMiddleware:
    """Answers 404 for a path that is not MCP's, and hands MCP every other request at
    `/mcp`, with the connection its URL asks for in the scope."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        connection = parse_connection(scope["path"], scope["query_string"].decode("latin-1"))
        if connection is None:
            await PlainTextResponse(NOT_FOUND, status_code=404)(scope, receive, send)
            return
        routed = dict(scope, path=MCP_PATH, raw_path=MCP_PATH.encode("ascii"))
        routed[CONNECTION_SCOPE_KEY] = connection
        await self.app(routed, receive, send)


def build_http_app(
    mcp: FastMCP,
    *,
    token: str,
    allowed_hosts: Collection[str],
    registry: IndexRegistry | None = None,
) -> Starlette:
    """`mcp` over streamable HTTP, behind the guard and the path middleware, and with a
    `registry` the dashboard of its indexes beside it. ValueError for an empty token or
    allowed host, here rather than at the first request."""
    _checked(token, allowed_hosts)
    sessions = None if registry is None else Sessions()
    dashboard = None if registry is None else dashboard_app(registry, sessions)

    # A closure, not Middleware(LoopbackGuard, token=...): Middleware's repr prints its
    # arguments, so the token would show wherever the app's middleware list is printed.
    def loopback_guard(app: ASGIApp) -> LoopbackGuard:
        return LoopbackGuard(
            app, token=token, allowed_hosts=allowed_hosts, dashboard=dashboard, sessions=sessions
        )

    return mcp.http_app(
        path=MCP_PATH,
        transport="http",
        middleware=[Middleware(loopback_guard), Middleware(IndexPathMiddleware)],
    )


async def _run(server: uvicorn.Server, registry: IndexRegistry) -> None:
    loop = asyncio.get_running_loop()
    if hasattr(signal, "SIGHUP"):
        # Not signal.signal: a reload takes the registry's locks, which a tool's thread may
        # hold, and reads files that may sit on a slow mount; the loop must not wait on it.
        loop.add_signal_handler(signal.SIGHUP, lambda: loop.run_in_executor(None, registry.reload))
    await server.serve()


def serve(config_path: Path, env: Mapping[str, str]) -> None:
    """Serve every index `config_path` configures until SIGINT or SIGTERM. SIGHUP reads
    the file again; an edit is also picked up by the next call. The address and the token
    are read once, here: changing them takes a restart. ConfigError when the file does not
    load or the token file is refused."""
    registry = IndexRegistry.from_file(config_path, env)
    settings = registry.server
    token = ensure_token(settings.token_file)
    allowed_hosts = {*LOOPBACK_NAMES, host_name(settings.host)}
    mcp = create_app(registry, connection=_CLOSED, env=env)
    app = build_http_app(mcp, token=token, allowed_hosts=allowed_hosts, registry=registry)
    host = f"[{settings.host}]" if ":" in settings.host else settings.host
    print(
        f"minirag-mcp serve: http://{host}:{settings.port}/mcp/<index> for "
        f"{', '.join(registry.names)}; token in {settings.token_file}",
        file=sys.stderr,
    )
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        lifespan="on",
        access_log=False,
        timeout_graceful_shutdown=2,
    )
    asyncio.run(_run(uvicorn.Server(config), registry))
