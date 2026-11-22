"""The last-sync report: one file next to the index, rewritten after every sync.

`latest` is the most recent sync of any scope and outcome. `coverage` is the most recent
unscoped sync that succeeded: the only one that says what the index as a whole holds and
what it could not read, so scoped and failed syncs leave it alone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from minirag_mcp import __version__
from minirag_mcp.sync import LARGE_TABLE_ERROR_PREFIX

REPORT_FILE = "last-sync.json"
ERRORS_CAP = 1000
FORMAT = 1


@dataclass(frozen=True)
class SyncRecord:
    started_at: str
    finished_at: str
    ran_by: str  # "cli" or "server"
    scope: str | None
    outcome: str  # "succeeded" or "failed"
    error: str | None
    counts: dict[str, int]
    errors: list[dict]
    rebuild: dict | None


def split_large_tables(errors: list[dict]) -> tuple[list[dict], list[str]]:
    """The failures among `errors`, and the sources skipped as large tables."""
    failures: list[dict] = []
    large: list[str] = []
    for entry in errors:
        if str(entry.get("error", "")).startswith(LARGE_TABLE_ERROR_PREFIX):
            large.append(entry["source"])
        else:
            failures.append(entry)
    return failures, large


def _load(path: Path) -> dict | None:
    """The report at `path`; None when absent; ValueError when it is not a report."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as e:
        raise ValueError(f"not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    if data.get("format") != FORMAT:
        raise ValueError(f"unknown format {data.get('format')!r}")
    if not isinstance(data.get("latest"), dict):
        raise ValueError("no latest sync")
    _check_shape(data)
    return data


def _check_shape(data: dict) -> None:
    if not isinstance(data["latest"].get("counts", {}), dict):
        raise ValueError("latest.counts is not an object")
    coverage = data.get("coverage")
    if coverage is None:
        return
    if not isinstance(coverage, dict):
        raise ValueError("coverage is not an object")
    errors = coverage.get("errors", [])
    if not isinstance(errors, list) or not all(
        isinstance(e, dict) and isinstance(e.get("source"), str) and isinstance(e.get("error"), str)
        for e in errors
    ):
        raise ValueError("coverage.errors is not a list of {source, error} strings")
    total = coverage.get("errorsTotal", len(errors))
    if type(total) is not int or total < 0:
        raise ValueError("coverage.errorsTotal is not a count")
    large = coverage.get("largeTables", [])
    if not isinstance(large, list) or not all(isinstance(x, str) for x in large):
        raise ValueError("coverage.largeTables is not a list of strings")


def read_report(db_path: Path) -> dict | None:
    path = Path(db_path) / REPORT_FILE
    try:
        return _load(path)
    except (ValueError, OSError) as e:
        return {"unreadable": f"{path} {e}"}


def record_sync(db_path: Path, record: SyncRecord) -> str | None:
    """Merge `record` into the report beside the index. A one-line warning when it cannot
    be written; never raises for the file's sake."""
    path = Path(db_path) / REPORT_FILE
    tmp = path.with_name(REPORT_FILE + ".tmp")
    try:
        try:
            previous = _load(path)
        except (ValueError, OSError):
            previous = None
        latest = {
            "startedAt": record.started_at,
            "finishedAt": record.finished_at,
            "ranBy": record.ran_by,
            "scope": record.scope,
            "version": __version__,
            "outcome": record.outcome,
            "error": record.error,
            "counts": record.counts,
            "rebuild": record.rebuild,
        }
        coverage = previous.get("coverage") if previous else None
        if record.scope is None and record.outcome == "succeeded":
            failures, large = split_large_tables(record.errors)
            coverage = {
                "startedAt": record.started_at,
                "finishedAt": record.finished_at,
                "ranBy": record.ran_by,
                "version": __version__,
                "counts": record.counts,
                "errors": failures[:ERRORS_CAP],
                "errorsTotal": len(failures),
                "largeTables": large,
            }
        body = json.dumps({"format": FORMAT, "latest": latest, "coverage": coverage})
        try:
            tmp.write_text(body, encoding="utf-8")
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    except (OSError, ValueError) as e:
        return f"could not write the last-sync report {path}: {e}"
    return None
