"""Preparers and embedders for spawned sync workers, which import this module by name."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

from conftest import FakeEmbedder
from minirag_mcp.endpoints import EndpointError

SLOW_ROUTER = "http://router.test/v1/embeddings did not answer within 120 s"
CRASH_MARK = "KILLS-ITS-WORKER"


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


class CrashingEmbedder(FakeEmbedder):
    """Kills the worker process embedding a text that holds `CRASH_MARK`, as a native abort
    would. In the process that runs the tests it raises instead."""

    def embed_documents(self, texts):
        if any(CRASH_MARK in t for t in texts):
            if multiprocessing.parent_process() is None:
                raise RuntimeError(f"{CRASH_MARK} embedded outside a worker")
            os._exit(3)
        return super().embed_documents(texts)
