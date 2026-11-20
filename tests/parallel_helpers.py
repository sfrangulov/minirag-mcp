"""Preparers for spawned sync workers, which import this module by name."""

from __future__ import annotations

import os
from pathlib import Path

from minirag_mcp.endpoints import EndpointError

SLOW_ROUTER = "http://router.test/v1/embeddings did not answer within 120 s"


class Preparer:
    """Reads a file as its own document; told to, writes to stdout, dies, raises, or
    fails as an embedding endpoint that timed out."""

    def __init__(
        self,
        *,
        crash_on: str = "",
        fail_on: str = "",
        endpoint_down_on: str = "",
        shout: bool = False,
    ) -> None:
        self.crash_on = crash_on
        self.fail_on = fail_on
        self.endpoint_down_on = endpoint_down_on
        self.shout = shout

    def prepare_file(self, path: Path, *, rules_checked: bool) -> str:
        if self.shout:
            print("LEAK-print")
            os.write(1, b"LEAK-fd\n")
        if path.name == self.crash_on:
            os._exit(3)
        if path.name == self.fail_on:
            raise ValueError(f"cannot read {path.name}")
        if path.name == self.endpoint_down_on:
            raise EndpointError(SLOW_ROUTER, timed_out=True)
        return path.read_text()


def broken_factory():
    raise RuntimeError("no model here")


class Unloadable:
    """Pickles in the parent; unpickling it, in a worker, raises as `broken_factory` does."""

    def __reduce__(self):
        return broken_factory, ()
