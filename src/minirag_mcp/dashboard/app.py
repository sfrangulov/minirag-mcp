"""The dashboard's routes, all under `/dashboard`. Which request may reach them is
`LoopbackGuard`'s decision (http_server.py); every page is built off the event loop, so
an MCP client never waits on the statistics of an index, and after the registry has
followed an edit of the indexes file, as every MCP call does first."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from minirag_mcp import __version__
from minirag_mcp.dashboard.data import DashboardData
from minirag_mcp.dashboard.render import (
    render_folder,
    render_index,
    render_message,
    render_overview,
)
from minirag_mcp.dashboard.sessions import COOKIE, COOKIE_PATH, SESSION_TTL, Sessions
from minirag_mcp.registry import IndexRegistry

DASHBOARD_PATH = COOKIE_PATH
LOGIN_PATH = f"{DASHBOARD_PATH}/login"
CODE_PATH = f"{DASHBOARD_PATH}/code"
SIGN_IN = "Run `minirag-mcp dashboard` to open this page signed in."
NOT_SIGNED_IN = f"Not signed in. {SIGN_IN}"
CODE_REFUSED = f"This sign-in link was used already or has expired. {SIGN_IN}"
PAGE_FAILED = "This page could not be built; the server's log says why."
NOT_FOUND = "Not found"
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; form-action 'none'; "
    "frame-ancestors 'none'; base-uri 'none'"
)

logger = logging.getLogger(__name__)


def is_dashboard_path(path: str) -> bool:
    """Whether `path` is one a browser sends the session cookie to (RFC 6265 path-match
    of COOKIE_PATH)."""
    return path == DASHBOARD_PATH or path.startswith(DASHBOARD_PATH + "/")


def dashboard_headers(app: ASGIApp) -> ASGIApp:
    """`app` with each response kept out of every cache, which would copy the client file
    names a page lists to disk and show them after the session ended, and allowed to load,
    frame or post nothing."""

    async def answer(scope: Scope, receive: Receive, send: Send) -> None:
        async def sending(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "no-store"
                headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
            await send(message)

        await app(scope, receive, sending)

    return answer


def not_signed_in() -> HTMLResponse:
    return HTMLResponse(render_message("Not signed in", NOT_SIGNED_IN), status_code=401)


def _generated() -> str:
    return datetime.now().strftime("%d.%m.%Y %H:%M")


def _not_found(text: str) -> tuple[int, str]:
    return 404, render_message(NOT_FOUND, text)


async def _page(registry: IndexRegistry, build: Callable[[], tuple[int, str]]) -> HTMLResponse:
    def fresh() -> tuple[int, str]:
        registry.refresh()
        return build()

    try:
        status, page = await run_in_threadpool(fresh)
    except Exception:
        logger.exception("minirag dashboard: a page could not be built")
        status, page = 500, render_message("Something went wrong", PAGE_FAILED)
    return HTMLResponse(page, status_code=status)


def dashboard_app(registry: IndexRegistry, sessions: Sessions) -> Starlette:
    data = DashboardData(registry)

    async def overview(request: Request) -> Response:
        def build() -> tuple[int, str]:
            views = data.overview()
            page = render_overview(
                views, unloaded=data.unloaded(), version=__version__, generated=_generated()
            )
            return 200, page

        return await _page(registry, build)

    async def index(request: Request) -> Response:
        name = request.path_params["name"]

        def build() -> tuple[int, str]:
            view = data.index(name)
            if view is None:
                return _not_found(f"No index is named {name} here.")
            return 200, render_index(view, version=__version__, generated=_generated())

        return await _page(registry, build)

    async def folder(request: Request) -> Response:
        name = request.path_params["name"]
        path = request.query_params.get("path", "")

        def build() -> tuple[int, str]:
            view = data.index(name)
            if view is None:
                return _not_found(f"No index is named {name} here.")
            files = data.folder(name, path)
            if files is None:
                return _not_found(f"{name} holds no indexed folder {path}.")
            page = render_folder(view, path, files, version=__version__, generated=_generated())
            return 200, page

        return await _page(registry, build)

    async def login(request: Request) -> Response:
        session_id = sessions.redeem(request.query_params.get("code", ""))
        if session_id is None:
            return HTMLResponse(render_message("Not signed in", CODE_REFUSED), status_code=401)
        response = RedirectResponse(DASHBOARD_PATH, status_code=303)
        response.set_cookie(
            COOKIE,
            session_id,
            max_age=int(SESSION_TTL),
            path=COOKIE_PATH,
            httponly=True,
            samesite="Strict",
        )
        return response

    async def code(request: Request) -> Response:
        return JSONResponse({"code": sessions.new_code()})

    return Starlette(
        routes=[
            Route(DASHBOARD_PATH, overview, methods=["GET"]),
            Route(f"{DASHBOARD_PATH}/index/{{name}}", index, methods=["GET"]),
            Route(f"{DASHBOARD_PATH}/index/{{name}}/folder", folder, methods=["GET"]),
            Route(LOGIN_PATH, login, methods=["GET"]),
            Route(CODE_PATH, code, methods=["POST"]),
        ]
    )
