"""Build the benchmark index with the server binary's own `sync`, and report it.

    uv run python evals/enterpriserag/index.py                     # full corpus
    uv run python evals/enterpriserag/index.py --report-only       # report an existing index

A file the scanner saw but nobody ingested, skipped or moved has vanished without a
failure: the report counts those as unaccounted and exits non-zero, like a failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import CORPUS, INDEX, REPO, open_index, outside_root

SERVER_BIN = REPO / ".venv" / "bin" / "minirag-mcp"


def unaccounted(counts: dict) -> int:
    return counts["scanned"] - counts["ingested"] - counts["skipped"] - counts["moved"]


def sync(server_bin: Path, base_dir: Path, db: Path, log: Path) -> dict:
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            [str(server_bin), "sync", "--base-dir", str(base_dir), "--db-path", str(db), "--json"],
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            env=dict(os.environ, HF_HUB_OFFLINE="1"),
        )
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise SystemExit(f"sync printed no JSON (exit {proc.returncode}); see {log}") from e
    out["seconds"] = round(time.monotonic() - started, 1)
    return out


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def write_report(result: dict, db: Path, out: Path) -> dict:
    from importlib.metadata import version

    counts = result["counts"]
    store = open_index(db, db.parent).get().store
    report = {
        "counts": counts,
        "failed_examples": result.get("errors", [])[:20],
        "unaccounted": unaccounted(counts),
        "seconds": result.get("seconds"),
        "chunks": store.chunk_count(),
        "sources": store.source_count(),
        "minirag_version": version("minirag-mcp"),
        "commit": git_commit(),
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--db", type=Path, default=INDEX)
    ap.add_argument("--server-bin", type=Path, default=SERVER_BIN)
    ap.add_argument("--report-only", type=Path, help="a sync --json output file to report on")
    args = ap.parse_args()
    db = outside_root(args.db, args.corpus)
    if args.report_only:
        result = json.loads(args.report_only.read_text(encoding="utf-8"))
    else:
        result = sync(args.server_bin, args.corpus, db, db.parent / "sync.log")
    report = write_report(result, db, db.parent / "index_report.json")
    print(json.dumps(report, indent=2))
    bad = report["counts"]["failed"] + report["unaccounted"]
    if bad:
        print(f"{bad} document(s) not in the index", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
