"""How many sync workers, and how many threads each may use."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from fastembed import TextEmbedding

FILES_PER_WORKER = 4
# Interpreter and libraries per worker process, measured.
WORKER_BASE_BYTES = 3 * 2**29


def _logical_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _linux_physical_cores(cpuinfo: str) -> int | None:
    cores: set[tuple[str, str]] = set()
    for block in re.split(r"\n\s*\n", cpuinfo):
        phys = re.search(r"^physical id\s*:\s*(\S+)", block, re.MULTILINE)
        core = re.search(r"^core id\s*:\s*(\S+)", block, re.MULTILINE)
        if phys and core:
            cores.add((phys.group(1), core.group(1)))
    return len(cores) or None


def physical_cores() -> int:
    logical = _logical_cpus()
    found: int | None = None
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout
            found = int(out.strip())
        elif sys.platform.startswith("linux"):
            found = _linux_physical_cores(Path("/proc/cpuinfo").read_text())
    except (OSError, ValueError, subprocess.SubprocessError):
        found = None
    return max(1, min(found or logical, logical))


def total_memory() -> int | None:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def worker_footprint(embedder) -> int:
    if getattr(embedder, "backend", "") == "endpoint":
        return WORKER_BASE_BYTES
    for entry in TextEmbedding.list_supported_models():
        if entry["model"] == getattr(embedder, "model_name", None):
            return WORKER_BASE_BYTES + int(entry["size_in_GB"] * 10**9)
    return WORKER_BASE_BYTES


def default_workers(cores: int, memory: int | None, footprint: int) -> int:
    workers = max(1, cores - 1)
    if memory is not None:
        workers = min(workers, max(1, memory // 2 // footprint))
    return workers


def worker_count(configured: int | None, files: int, *, default: int) -> int:
    return max(1, min(configured or default, files // FILES_PER_WORKER))


def worker_threads(workers: int) -> int:
    return max(1, physical_cores() // workers)
