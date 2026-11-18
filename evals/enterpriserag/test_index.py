from __future__ import annotations

import json

import pytest
from index import sync, unaccounted


def test_unaccounted_counts_files_nobody_handled():
    counts = {"scanned": 10, "ingested": 7, "skipped": 1, "moved": 1, "failed": 0}
    assert unaccounted(counts) == 1
    assert unaccounted({**counts, "ingested": 8}) == 0


def _fake_bin(tmp_path, stdout: str):
    b = tmp_path / "fake-minirag"
    b.write_text(f"#!/bin/sh\necho 'progress' >&2\ncat <<'EOF'\n{stdout}\nEOF\n")
    b.chmod(0o755)
    return b


def test_sync_returns_counts_and_logs_stderr(tmp_path):
    payload = {"counts": {"scanned": 1, "ingested": 1}, "errors": []}
    out = sync(_fake_bin(tmp_path, json.dumps(payload)), tmp_path, tmp_path / "db", tmp_path / "l")
    assert out["counts"] == payload["counts"]
    assert "seconds" in out
    assert (tmp_path / "l").read_text().strip() == "progress"


def test_sync_without_json_is_loud(tmp_path):
    with pytest.raises(SystemExit):
        sync(_fake_bin(tmp_path, "not json"), tmp_path, tmp_path / "db", tmp_path / "l")
