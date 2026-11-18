"""Answer the benchmark's questions through `claude -p`, in one of two arms.

    uv run python evals/enterpriserag/run.py --arm files   --ids pilot --out <run_dir>
    uv run python evals/enterpriserag/run.py --arm minirag --ids all   --out <run_dir>

minirag: only this server's read tools, no built-in tool. files: no MCP server, Claude
Code's own read-only file tools over the same corpus. Same model, same prompt except the
one line that says where the documents are. Each run gets a fresh working directory
outside any repository and no user settings, as in evals/behaviour/run.py.

A run that hits a usage or rate limit is not an answer: it is left undone, and after a
few in a row the whole run stops so it can be resumed later. A question that ran out
its time is an answer, an empty one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from answers import final_result
from common import (
    BENCH,
    CORPUS,
    INDEX,
    REPO,
    Question,
    load_questions,
    outside_root,
    pilot_ids,
    sample_ids,
)

ARMS = ("minirag", "files")
MODEL = "claude-opus-5-5"
TIME_LIMIT = 600
MCP_TIMEOUT_MS = "120000"
READ_TOOLS = (
    "query_documents",
    "find_mentions",
    "status",
    "read_file",
    "read_chunk_neighbors",
    "list_files",
)
FILE_TOOLS = "Read,Grep,Glob,Bash"
FILE_ALLOWED = (
    "Read",
    "Grep",
    "Glob",
    *(f"Bash({c}:*)" for c in ("grep", "find", "ls", "head", "tail", "wc", "cat", "sort", "uniq")),
)
LIMIT_RE = re.compile(r"usage limit|rate limit|overloaded|too many requests|limit reached", re.I)
STOP_AFTER_LIMITS = 3


def knowledge_sources(bench: Path = BENCH) -> str:
    text = (bench / "src/scripts/answer_generation/agents_example.md").read_text(encoding="utf-8")
    m = re.search(r"^## Knowledge Sources\n(.*?)(?=^## )", text, re.S | re.M)
    if not m:
        raise SystemExit("agents_example.md has no Knowledge Sources section")
    return "## Knowledge Sources\n" + m.group(1).strip()


def build_prompt(q: Question, sources_note: str, arm: str, corpus: Path) -> str:
    where = {
        "minirag": "The documents are indexed in the minirag search tools available to you.",
        "files": f"The documents are plain-text files under {corpus}.",
    }[arm]
    return "\n\n".join(
        [
            "You answer questions about the internal documents of a company called "
            "Redwood Inference.",
            where,
            sources_note,
            f"Question: {q.question}",
            "Answer the question thoroughly, paying attention to every qualifier in it. If "
            "the documents do not contain the answer, say so. End your reply with a list of "
            "the documents that support your answer, most important first, one per line, "
            "each identified by its file name or path; a document's file name starts with "
            "its dsid_ id.",
        ]
    )


def claude_command(arm: str, prompt: str, mcp_config: Path | None, model: str, corpus: Path):
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--strict-mcp-config",
        "--setting-sources",
        "project",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--model",
        model,
    ]
    if arm == "minirag":
        cmd += ["--mcp-config", str(mcp_config), "--tools", ""]
        cmd += ["--allowedTools", ",".join(f"mcp__minirag__{t}" for t in READ_TOOLS)]
    else:
        cmd += ["--tools", FILE_TOOLS, "--add-dir", str(corpus)]
        cmd += ["--allowedTools", ",".join(FILE_ALLOWED)]
    return cmd


def limited(result: dict | None) -> bool:
    if not result or not result.get("is_error"):
        return False
    return bool(LIMIT_RE.search(json.dumps(result)))


def utilization(transcript: Path) -> float | None:
    """The account's seven-day usage as the last rate-limit event of a run reported it."""
    last = None
    if not transcript.exists():
        return None
    for line in transcript.read_text(encoding="utf-8").splitlines():
        if '"rate_limit_event"' not in line:
            continue
        try:
            info = json.loads(line).get("rate_limit_info") or {}
        except json.JSONDecodeError:
            continue
        window = (info.get("unifiedWindows") or {}).get("seven_day") or {}
        if window.get("utilization") is not None:
            last = float(window["utilization"])
    return last


def done(raw: Path, qid: str) -> bool:
    if (raw / f"{qid}.timeout").exists():
        return True
    transcript = raw / f"{qid}.jsonl"
    if not transcript.exists():
        return False
    result = final_result(transcript)
    return result is not None and not limited(result)


def write_mcp_config(path: Path, server_bin: Path, db: Path, base_dir: Path) -> None:
    server = {
        "type": "stdio",
        "command": str(server_bin),
        "args": [],
        "env": {"BASE_DIR": str(base_dir), "DB_PATH": str(db), "HF_HUB_OFFLINE": "1"},
    }
    path.write_text(json.dumps({"mcpServers": {"minirag": server}}, indent=2), encoding="utf-8")


def prewarm(server_bin: Path, db: Path, base_dir: Path, limit: float = 300.0) -> float:
    """Start the server once and wait for `initialize`, so no run pays the cold start."""
    env = dict(os.environ, BASE_DIR=str(base_dir), DB_PATH=str(db), HF_HUB_OFFLINE="1")
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


def run_env() -> dict[str, str]:
    # A command past its timeout is otherwise moved to the background, where it outlives
    # the run: in the first pilot twenty orphaned `grep -r` scans of the whole corpus
    # starved every later question.
    return dict(os.environ, MCP_TIMEOUT=MCP_TIMEOUT_MS, CLAUDE_CODE_DISABLE_BACKGROUND_TASKS="1")


def _stop(proc: subprocess.Popen) -> None:
    """SIGTERM first, so Claude Code ends the shells it started in groups of their own."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_one(cmd: list[str], raw: Path, qid: str, limit: int = TIME_LIMIT) -> dict | None:
    cwd = Path(tempfile.mkdtemp(prefix="minirag-erb-run-"))
    transcript = raw / f"{qid}.jsonl"
    env = run_env()
    try:
        with transcript.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=out,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                _stop(proc)
                (raw / f"{qid}.timeout").write_text(str(limit))
                return None
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    return final_result(transcript)


def select_ids(questions: list[Question], spec: str) -> list[str]:
    if spec == "pilot":
        return pilot_ids(questions)
    if spec == "sample":
        return sample_ids(questions)
    if spec == "all":
        return [q.question_id for q in questions]
    wanted = spec.split(",")
    known = {q.question_id for q in questions}
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise SystemExit(f"unknown question ids: {unknown}")
    return wanted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--ids", default="pilot")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument(
        "--max-utilization",
        type=float,
        required=True,
        help="stop once the account's seven-day usage reaches this share (0-1)",
    )
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--db", type=Path, default=INDEX)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--server-bin", type=Path, default=REPO / ".venv" / "bin" / "minirag-mcp")
    args = ap.parse_args()

    questions = load_questions()
    by_id = {q.question_id: q for q in questions}
    ids = select_ids(questions, args.ids)
    corpus = args.corpus.resolve()
    raw = args.out / args.arm / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ids_file = args.out / "ids.json"
    if not ids_file.exists():
        ids_file.write_text(json.dumps(ids), encoding="utf-8")
    (args.out / args.arm / "config.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "model": args.model,
                "time_limit": TIME_LIMIT,
                "corpus": str(corpus),
                "db": str(args.db),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mcp_config = None
    if args.arm == "minirag":
        db = outside_root(args.db, corpus)
        mcp_config = args.out / args.arm / "mcp.json"
        write_mcp_config(mcp_config, args.server_bin, db, corpus)
        print(f"server ready in {prewarm(args.server_bin, db, corpus):.1f}s", flush=True)

    note = knowledge_sources()
    todo = [qid for qid in ids if not done(raw, qid)]
    print(f"{args.arm}: {len(ids) - len(todo)} done, {len(todo)} to run", flush=True)
    stop = threading.Event()
    over = threading.Event()
    streak = [0]
    lock = threading.Lock()

    def work(qid: str) -> None:
        if stop.is_set():
            return
        cmd = claude_command(
            args.arm,
            build_prompt(by_id[qid], note, args.arm, corpus),
            mcp_config,
            args.model,
            corpus,
        )
        started = time.monotonic()
        result = run_one(cmd, raw, qid)
        with lock:
            if limited(result):
                streak[0] += 1
                print(f"{qid}: limit hit ({streak[0]} in a row)", flush=True)
                if streak[0] >= STOP_AFTER_LIMITS:
                    stop.set()
                return
            streak[0] = 0
            state = "timeout" if result is None else ("error" if result.get("is_error") else "ok")
            used = utilization(raw / f"{qid}.jsonl")
            print(
                f"{qid}: {state} in {time.monotonic() - started:.0f}s, week at {used}", flush=True
            )
            if used is not None and used >= args.max_utilization:
                print(f"weekly usage {used} reached the cap {args.max_utilization}", flush=True)
                over.set()
                stop.set()

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(work, todo))
    left = [qid for qid in ids if not done(raw, qid)]
    if over.is_set():
        print(f"stopped at the weekly usage cap; {len(left)} left; resume with a higher cap")
        return 3
    if stop.is_set():
        print(f"stopped on usage limits; {len(left)} left; re-run the same command to resume")
        return 2
    print(f"{args.arm}: finished, {len(left)} left")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
