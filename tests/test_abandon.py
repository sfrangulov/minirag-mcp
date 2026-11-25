import threading
import time

import anyio

from minirag_mcp.abandon import Abandoned, checkpoint, run_abandonable


def test_checkpoint_outside_a_call_does_nothing():
    assert checkpoint() is None


def _looping(seconds: float, ended: threading.Event, abandoned: list[bool]):
    def fn():
        try:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                checkpoint()
                time.sleep(0.01)
            abandoned.append(False)
            return "done"
        except Abandoned:
            abandoned.append(True)
            raise
        finally:
            ended.set()

    return fn


async def test_an_abandoned_call_stops_at_its_next_checkpoint():
    ended = threading.Event()
    abandoned: list[bool] = []
    started = time.monotonic()
    with anyio.move_on_after(0.1):
        await run_abandonable(_looping(5, ended, abandoned))
    assert time.monotonic() - started < 1.0
    assert ended.wait(1.0)
    assert abandoned == [True]


async def test_a_finished_call_returns_its_value():
    assert await run_abandonable(lambda: 42) == 42


async def test_a_concurrent_call_is_not_stopped_by_another_abandoned_one():
    first_ended, second_ended = threading.Event(), threading.Event()
    first: list[bool] = []
    second: list[bool] = []
    results: list[str] = []

    async def abandoned_one():
        with anyio.move_on_after(0.1):
            await run_abandonable(_looping(0.5, first_ended, first))

    async def surviving_one():
        results.append(await run_abandonable(_looping(0.5, second_ended, second)))

    async with anyio.create_task_group() as tg:
        tg.start_soon(abandoned_one)
        tg.start_soon(surviving_one)
    assert first_ended.wait(1.0)
    assert first == [True]
    assert second == [False]
    assert results == ["done"]
