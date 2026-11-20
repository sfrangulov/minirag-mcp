"""An ASGI app served over real HTTP on a free loopback port, and what an MCP client sends
to one."""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from mcp.types import LATEST_PROTOCOL_VERSION

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
