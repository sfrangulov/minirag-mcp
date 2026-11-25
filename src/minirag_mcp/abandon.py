"""Stop work for a call its client cancelled.

`Abandoned` derives from `BaseException` on purpose: the core's fallback handlers catch
`Exception` and turn a failure into a degraded answer, which would swallow the stop and
keep computing.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextvars import ContextVar
from typing import TypeVar

import anyio
import anyio.to_thread

T = TypeVar("T")

_STOP: ContextVar[threading.Event | None] = ContextVar("minirag_stop", default=None)


class Abandoned(BaseException):
    """The client of the current call cancelled it."""


def checkpoint() -> None:
    stop = _STOP.get()
    if stop is not None and stop.is_set():
        raise Abandoned


async def run_abandonable(fn: Callable[[], T]) -> T:
    # anyio frees its thread-limiter slot on cancel while the thread keeps running, so
    # abandoned work stops counting against the limit of 40. Reads stop at their next
    # checkpoint; writes, a single LanceDB query and a single endpoint call run to the end.
    stop = threading.Event()
    token = _STOP.set(stop)
    try:
        return await anyio.to_thread.run_sync(fn, abandon_on_cancel=True)
    except anyio.get_cancelled_exc_class():
        stop.set()
        raise
    finally:
        _STOP.reset(token)
