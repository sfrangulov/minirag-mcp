"""An ASGI app served over real HTTP on a free loopback port, and what an MCP client sends
to one."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    CancelledNotification,
    CancelledNotificationParams,
    ClientNotification,
)
from sse_starlette.sse import AppStatus

TOKEN = "test-token-0123456789abcdefghijklmnopqrstuvwxyz"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": LATEST_PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "minirag-tests", "version": "0"},
    },
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
STARTUP_TIMEOUT_SECONDS = 10.0


@asynccontextmanager
async def serving(app) -> AsyncIterator[str]:
    """`app` served by uvicorn on 127.0.0.1 in this event loop, with its lifespan; its base
    URL. A ClosedResourceError traceback logged at shutdown is noise."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        app, lifespan="on", access_log=False, log_level="warning", timeout_graceful_shutdown=2
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while not server.started:
            if task.done():
                task.result()
                raise AssertionError("uvicorn stopped before it started serving")
            if time.monotonic() >= deadline:
                raise AssertionError(f"uvicorn did not start within {STARTUP_TIMEOUT_SECONDS:g}s")
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
        sock.close()
        # sse_starlette latches this process-wide when a server shuts down while a response
        # stream is open; every later stream in the process would then end at once.
        AppStatus.should_exit = False


@asynccontextmanager
async def _session(base: str, index: str) -> AsyncIterator[ClientSession]:
    async with httpx.AsyncClient(headers=AUTH) as http:
        async with streamable_http_client(f"{base}/mcp/{index}", http_client=http) as (r, w, _):
            async with ClientSession(r, w) as session:
                await session.initialize()
                yield session


async def call_tool(base: str, index: str, tool: str, args: dict):
    """One ordinary call, awaited to its answer."""
    async with _session(base, index) as session:
        return await session.call_tool(tool, args)


async def abandon_call(base: str, index: str, tool: str, args: dict, after_s: float) -> None:
    """Call `tool`, then, `after_s` seconds in, cancel it and leave the session without
    waiting for the answer, as a client with a timeout does."""
    async with _session(base, index) as session:
        # mcp >= 2 clients send this frame when they abandon a request; 1.29 does not.
        request_id = session._request_id
        call = asyncio.create_task(session.call_tool(tool, args))
        await asyncio.sleep(after_s)
        await session.send_notification(
            ClientNotification(
                CancelledNotification(
                    method="notifications/cancelled",
                    params=CancelledNotificationParams(requestId=request_id),
                )
            )
        )
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
