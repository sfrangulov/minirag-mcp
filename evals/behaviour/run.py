"""Run the behaviour eval: every item, `--reps` times, through `claude -p` with only this
server attached. Then score the transcripts (score.py).

    uv run python evals/behaviour/run.py --dry-run     # the plan and the count, nothing launched
    uv run python evals/behaviour/run.py --reps 3 --model opus
    uv run python evals/behaviour/run.py --items R1,R2 --server-bin <other>/.venv/bin/minirag-mcp

What a run sees, and why:

- Only minirag. `--strict-mcp-config` drops every other MCP server and claude.ai
  connector, and `--tools ""` every built-in tool, so a model that does not search
  cannot answer from the web or the filesystem instead.
- No user or project configuration. The run's working directory is a fresh temporary
  folder outside any repository and `--setting-sources project` reads settings only
  from there, so no CLAUDE.md, plugin, hook or permission rule of the person running
  the eval reaches the model. `--disable-slash-commands` keeps skills out.
- Read-only permissions. The read tools are allowed; the tools that change the index
  are not, so `claude -p` denies them. The attempt still appears in the transcript, which
  is what the injection items count, and the shared index stays the same for every run.
- The server under test is a binary, `--server-bin`, defaulting to this checkout's. To
  measure another revision (main, or a branch with one edited block), point it at that
  worktree's `.venv/bin/minirag-mcp`. The index is built with the same binary.

Raw transcripts land in <out>/raw/<item>/<rep>.jsonl (out defaults to
evals/behaviour/runs/<UTC timestamp>, which git ignores).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from items import CORPUS, HERE, load_items, select
from score import score_dir

REPO = HERE.parent.parent
READ_TOOLS = (
    "query_documents",
    "status",
    "read_file",
    "read_chunk_neighbors",
    "list_files",
    "sync_status",
)


def claude_command(prompt: str, mcp_config: Path, model: str | None) -> list[str]:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--mcp-config",
        str(mcp_config),
        "--strict-mcp-config",
        "--tools",
        "",
        "--allowedTools",
        ",".join(f"mcp__minirag__{t}" for t in READ_TOOLS),
        "--setting-sources",
        "project",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    if model:
        cmd += ["--model", model]
    return cmd


def write_mcp_config(path: Path, server_bin: Path, db: Path) -> None:
    server = {
        "type": "stdio",
        "command": str(server_bin),
        "args": [],
        "env": {"BASE_DIR": str(CORPUS), "DB_PATH": str(db)},
    }
    path.write_text(json.dumps({"mcpServers": {"minirag": server}}, indent=2), encoding="utf-8")


# Claude Code gives a stdio server this long to answer `initialize`. A server binary's
# first start in a fresh environment — no bytecode yet, every native library read and
# scanned for the first time — can take longer than the default, and a run whose server
# did not connect measures nothing.
MCP_TIMEOUT_MS = "120000"


def prewarm(server_bin: Path, db: Path, limit: float = 180.0) -> float:
    """Start the server once and wait for its `initialize` answer, so the first real run
    does not pay the cold start; fail loudly if it never answers."""
    env = dict(os.environ, BASE_DIR=str(CORPUS), DB_PATH=str(db))
    started = time.monotonic()
    proc = subprocess.Popen(
        [str(server_bin)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        text=True,
    )
    hello = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "prewarm", "version": "0"},
        },
    }
    try:
        proc.stdin.write(json.dumps(hello) + "\n")
        proc.stdin.flush()
        with ThreadPoolExecutor(max_workers=1) as pool:
            line = pool.submit(proc.stdout.readline).result(timeout=limit)
    except Exception as e:
        raise SystemExit(f"{server_bin} did not answer initialize within {limit:g}s: {e}") from e
    finally:
        proc.kill()
    if '"result"' not in line:
        raise SystemExit(f"{server_bin} answered initialize with: {line[:300]}")
    return time.monotonic() - started


def build_index(server_bin: Path, db: Path) -> None:
    proc = subprocess.run(
        [str(server_bin), "sync", "--base-dir", str(CORPUS), "--db-path", str(db), "--json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"building the index failed:\n{proc.stdout}\n{proc.stderr}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--items", help="comma-separated item ids (default: all)")
    ap.add_argument("--model", help="passed to claude --model; recorded either way")
    ap.add_argument("--server-bin", type=Path, default=REPO / ".venv" / "bin" / "minirag-mcp")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--jobs", type=int, default=1, help="parallel claude processes (default 1)")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = select(load_items(), args.items)
    runs = [(item, rep) for item in items for rep in range(1, args.reps + 1)]
    print(f"{len(items)} item(s) x {args.reps} rep(s) = {len(runs)} `claude -p` run(s)")
    if args.dry_run:
        example = claude_command(items[0].prompt, Path("<work>/mcp-corpus.json"), args.model)
        print("example:", " ".join(repr(a) if " " in a or not a else a for a in example))
        return 0
    if shutil.which("claude") is None:
        raise SystemExit("claude is not on PATH")
    if not args.server_bin.exists():
        raise SystemExit(f"server binary not found: {args.server_bin} (run `uv sync` first)")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or HERE / "runs" / stamp
    (out / "raw").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="minirag-behaviour-"))
    cwd = work / "cwd"
    cwd.mkdir()
    configs = {}
    for index in sorted({i.index for i in items}):
        db = work / f"db-{index}"
        if index == "corpus":
            build_index(args.server_bin, db)
        configs[index] = work / f"mcp-{index}.json"
        write_mcp_config(configs[index], args.server_bin, db)
    print(f"server answered initialize in {prewarm(args.server_bin, work / 'db-prewarm'):.1f}s")
    (out / "run.json").write_text(
        json.dumps(
            {
                "started": stamp,
                "model": args.model,
                "server_bin": str(args.server_bin),
                "reps": args.reps,
                "items": [i.id for i in items],
                "work": str(work),
                "command": claude_command("<prompt>", Path("<config>"), args.model),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    # A run whose server did not connect measures nothing; neither would the rest.
    broken = threading.Event()

    def one(job):
        item, rep = job
        if broken.is_set():
            print(f"  {item.id}#{rep}: skipped, an earlier run had no server", flush=True)
            return
        dest = out / "raw" / item.id / f"{rep}.jsonl"
        dest.parent.mkdir(exist_ok=True)
        started = time.monotonic()
        try:
            proc = subprocess.run(
                claude_command(item.prompt, configs[item.index], args.model),
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                stdin=subprocess.DEVNULL,
                env=dict(os.environ, MCP_TIMEOUT=MCP_TIMEOUT_MS),
            )
            dest.write_text(proc.stdout, encoding="utf-8")
            if proc.stderr.strip():
                dest.with_suffix(".err").write_text(proc.stderr, encoding="utf-8")
            status = f"exit {proc.returncode}"
        except subprocess.TimeoutExpired as e:
            dest.write_text(e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))
            status = "TIMEOUT"
        connected = '"name":"minirag","status":"connected"' in dest.read_text(encoding="utf-8")
        if not connected:
            status += ", minirag NOT connected"
            broken.set()
        print(f"  {item.id}#{rep}: {status} in {time.monotonic() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(one, runs))

    rows, summary = score_dir(out)
    (out / "scored.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"transcripts and scores: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
