from __future__ import annotations

import json
import os

from minirag_mcp import sync_report
from minirag_mcp.sync import LARGE_TABLE_ERROR_PREFIX
from minirag_mcp.sync_report import SyncRecord, read_report, record_sync


def make(**kw) -> SyncRecord:
    fields = dict(
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:05+00:00",
        ran_by="cli",
        scope=None,
        outcome="succeeded",
        error=None,
        counts={"ingested": 2, "failed": 1},
        errors=[],
        rebuild=None,
    )
    fields.update(kw)
    return SyncRecord(**fields)


def large(source):
    return {"source": source, "error": f"{LARGE_TABLE_ERROR_PREFIX}5 data rows exceeds"}


def test_a_first_full_sync_writes_latest_and_coverage(tmp_path):
    errors = [{"source": "a.pdf", "error": "boom"}, large("big.csv")]
    assert record_sync(tmp_path, make(errors=errors)) is None
    report = read_report(tmp_path)
    assert report["format"] == 1
    assert report["latest"]["ranBy"] == "cli" and report["latest"]["scope"] is None
    assert report["coverage"]["errors"] == [{"source": "a.pdf", "error": "boom"}]
    assert report["coverage"]["errorsTotal"] == 1
    assert report["coverage"]["largeTables"] == ["big.csv"]


def test_a_scoped_sync_replaces_latest_and_keeps_coverage(tmp_path):
    record_sync(tmp_path, make())
    before = read_report(tmp_path)["coverage"]
    scope = tmp_path / "docs"
    record_sync(tmp_path, make(scope=str(scope), ran_by="server", counts={"ingested": 9}))
    report = read_report(tmp_path)
    assert report["latest"]["scope"] == str(scope)
    assert report["latest"]["ranBy"] == "server"
    assert report["coverage"] == before


def test_a_failed_sync_replaces_latest_and_keeps_coverage(tmp_path):
    record_sync(tmp_path, make())
    before = read_report(tmp_path)["coverage"]
    record_sync(tmp_path, make(outcome="failed", error="disk full", counts={}))
    report = read_report(tmp_path)
    assert report["latest"]["outcome"] == "failed"
    assert report["latest"]["error"] == "disk full"
    assert report["coverage"] == before


def test_errors_are_capped_with_their_total(tmp_path):
    errors = [{"source": f"f{i}", "error": "x"} for i in range(1500)]
    record_sync(tmp_path, make(errors=errors))
    coverage = read_report(tmp_path)["coverage"]
    assert len(coverage["errors"]) == sync_report.ERRORS_CAP == 1000
    assert coverage["errorsTotal"] == 1500


def test_the_report_is_written_atomically(tmp_path, monkeypatch):
    record_sync(tmp_path, make(counts={"ingested": 1}))
    assert list(tmp_path.glob("*.tmp")) == []
    before = (tmp_path / sync_report.REPORT_FILE).read_bytes()

    def crash(*a, **k):
        raise OSError("no space")

    monkeypatch.setattr(os, "replace", crash)
    warning = record_sync(tmp_path, make(counts={"ingested": 2}))
    assert warning is not None and "no space" in warning
    assert (tmp_path / sync_report.REPORT_FILE).read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_an_unreadable_report_is_reported_not_raised(tmp_path):
    path = tmp_path / sync_report.REPORT_FILE
    for body in (b"\xff\x00garbage", b'{"format": 99}', b"[1, 2]"):
        path.write_bytes(body)
        assert "unreadable" in read_report(tmp_path)
    path.write_bytes(b"\xff\x00garbage")
    record_sync(tmp_path, make(scope="/x"))
    report = read_report(tmp_path)
    assert report["latest"]["scope"] == "/x" and report["coverage"] is None
    path.write_bytes(b"[1, 2]")
    record_sync(tmp_path, make())
    assert read_report(tmp_path)["coverage"]["counts"] == {"ingested": 2, "failed": 1}
    json.loads(path.read_text())


def test_a_missing_report_reads_as_none(tmp_path):
    assert read_report(tmp_path) is None
