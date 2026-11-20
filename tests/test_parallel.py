import os
from types import SimpleNamespace

import pytest

from minirag_mcp import parallel

GiB = 2**30

SAMPLE = "".join(
    f"processor\t: {n}\nphysical id\t: {phys}\ncore id\t\t: {core}\n\n"
    for n, (phys, core) in enumerate([(p, c) for p in (0, 1) for c in (0, 1)] * 2)
)


@pytest.mark.parametrize(
    "cores,memory,footprint,expected",
    [
        (18, 64 * GiB, int(1.5 * GiB), 17),  # endpoint: the cores decide
        (18, 64 * GiB, int(3.6 * GiB), 8),  # in-process e5: memory decides
        (8, 16 * GiB, int(3.6 * GiB), 2),
        (8, None, int(3.6 * GiB), 7),  # memory unknown
        (1, 64 * GiB, GiB, 1),
        (4, GiB, 4 * GiB, 1),
    ],
)
def test_default_workers(cores, memory, footprint, expected):
    assert parallel.default_workers(cores, memory, footprint) == expected


@pytest.mark.parametrize(
    "configured,files,default,expected",
    [
        (None, 3, 8, 1),
        (None, 100, 8, 8),
        (2, 12, 8, 2),
        (16, 20, 8, 5),
        (4, 7, 8, 1),
        (1, 1000, 8, 1),
    ],
)
def test_worker_count(configured, files, default, expected):
    assert parallel.worker_count(configured, files, default=default) == expected


def test_worker_footprint_counts_the_model_only_when_it_runs_in_process(fake_embedder):
    e5 = SimpleNamespace(model_name="intfloat/multilingual-e5-large", backend="in-process")
    assert parallel.worker_footprint(e5) == parallel.WORKER_BASE_BYTES + 2_240_000_000
    e5.backend = "endpoint"
    assert parallel.worker_footprint(e5) == parallel.WORKER_BASE_BYTES
    assert parallel.worker_footprint(fake_embedder) == parallel.WORKER_BASE_BYTES


def test_linux_physical_cores_counts_distinct_cores():
    assert parallel._linux_physical_cores(SAMPLE) == 4


def test_linux_physical_cores_is_none_without_topology():
    assert parallel._linux_physical_cores("processor\t: 0\nmodel name\t: x\n") is None


def test_physical_cores_is_within_the_logical_count():
    assert 1 <= parallel.physical_cores() <= os.cpu_count()


def test_worker_threads_share_the_cores(monkeypatch):
    monkeypatch.setattr(parallel, "physical_cores", lambda: 8)
    assert [parallel.worker_threads(w) for w in (1, 3, 16)] == [8, 2, 1]


def test_total_memory_is_positive_or_unknown():
    memory = parallel.total_memory()
    assert memory is None or memory > 0
