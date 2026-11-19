"""A thin tool-calling agent for models Claude Code does not suit.

    python agent.py --arm minirag|files --model qwen/qwen3.8-27b --prompt "<text>"

Claude Code's system prompt and tool set are tuned for Claude; a 27B open model loses
its way in them. This loop is the smallest one that still answers by searching: one
system paragraph, the question, the arm's tools, and a forced final answer when time
runs out. It prints events in the shape `claude -p --output-format stream-json` does,
so answers.py and run.py read both the same way.

minirag arm: the server's read tools over MCP, with the server's own instructions added
to the system prompt, as an MCP client passes them. files arm: grep/read/list/find over
the corpus inside a Linux container — on the owner's Mac, Defender and Jamf Protect scan
every file open and a full-corpus search takes three to five minutes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import posixpath
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
TIME_LIMIT = 540
MAX_TURNS = 40
TOOL_CHARS = 12_000
CONTAINER = "erb-corpus"
CONTAINER_ROOT = "/corpus"
READ_TOOLS = (
    "query_documents",
    "find_mentions",
    "status",
    "read_file",
    "read_chunk_neighbors",
    "list_files",
)
SYSTEM = (
    "You answer questions from a company's internal documents, which you can reach only "
    "through your tools. Search before answering, read the documents that look relevant, "
    "and base the answer only on what they say. Call tools as often as you need; when you "
    "have enough, reply with the final answer and no tool call."
)
FINISH = (
    "Time is up. Answer now from what you have found, and end with the list of documents "
    "that support the answer."
)


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY is not set and not in .env")


def chat(model: str, messages: list, tools: list | None, key: str) -> tuple[dict, dict]:
    body: dict = {"model": model, "messages": messages, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
    last = None
    for attempt in range(4):
        try:
            r = requests.post(
                OPENROUTER,
                headers={"Authorization": f"Bearer {key}"},
                json=body,
                timeout=240,
            )
            data = r.json()
            if r.status_code == 200 and data.get("choices"):
                return data["choices"][0]["message"], data.get("usage") or {}
            last = data.get("error") or r.text[:300]
            if r.status_code in (400, 401, 402, 403):
                break
        except (requests.RequestException, ValueError) as e:
            last = str(e)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"OpenRouter call failed: {last}")


class Events:
    def __init__(self, out=sys.stdout):
        self.out = out

    def emit(self, event: dict) -> None:
        self.out.write(json.dumps(event) + "\n")
        self.out.flush()


def _clip(text: str, limit: int = TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters cut]"


class FileTools:
    """Read-only search over the corpus inside a container; no shell reaches the model."""

    def __init__(self, container: str = CONTAINER, root: str = CONTAINER_ROOT):
        self.container = container
        self.root = root

    def specs(self) -> list[dict]:
        def fn(name, desc, props, required):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }

        path = {"type": "string", "description": f"a path under {self.root}, default the root"}
        return [
            fn(
                "grep",
                "Search file contents with ripgrep (regex). Returns up to 100 lines: file "
                "paths only by default, or matching lines with files_only=false.",
                {
                    "pattern": {"type": "string"},
                    "path": path,
                    "ignore_case": {"type": "boolean"},
                    "files_only": {"type": "boolean"},
                },
                ["pattern"],
            ),
            fn(
                "read",
                "Read a file's lines, numbered. offset is the first line (1-based).",
                {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                ["path"],
            ),
            fn("list", "List a directory's entries (up to 200).", {"path": path}, []),
            fn(
                "find",
                "Find files whose name contains a text (case-insensitive), up to 100.",
                {"name": {"type": "string"}, "path": path},
                ["name"],
            ),
        ]

    def resolve(self, path: str | None) -> str:
        rel = (path or "").strip()
        if rel.startswith(self.root):
            rel = rel[len(self.root) :]
        full = posixpath.normpath(posixpath.join(self.root, rel.lstrip("/")))
        if full != self.root and not full.startswith(self.root + "/"):
            raise ValueError(f"{path!r} is outside {self.root}")
        return full

    def _sh(self, script: str, *args: str, timeout: int = 120) -> str:
        cmd = ["docker", "exec", self.container, "sh", "-c", script, "sh", *args]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"Error: the command took longer than {timeout}s"
        out = p.stdout if p.stdout.strip() else p.stderr
        return out if out.strip() else "(no output)"

    def call(self, name: str, args: dict) -> str:
        try:
            if name == "grep":
                flags = "-i" if args.get("ignore_case", True) else "-s"
                mode = "-l" if args.get("files_only", True) else "-n -m 3 --max-columns 300"
                return self._sh(
                    f'rg {flags} {mode} -e "$1" -- "$2" | head -100',
                    str(args["pattern"]),
                    self.resolve(args.get("path")),
                )
            if name == "read":
                start = max(1, int(args.get("offset") or 1))
                end = start + min(int(args.get("limit") or 400), 2000) - 1
                return self._sh(
                    f'sed -n "{start},{end}p" -- "$1" | nl -ba -v {start}',
                    self.resolve(args["path"]),
                )
            if name == "list":
                return self._sh('ls -1 -- "$1" | head -200', self.resolve(args.get("path")))
            if name == "find":
                return self._sh(
                    'find "$2" -iname "*$1*" | head -100',
                    str(args["name"]),
                    self.resolve(args.get("path")),
                )
        except (KeyError, ValueError, TypeError) as e:
            return f"Error: {e}"
        return f"Error: unknown tool {name}"


def _mcp_spec(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


async def answer(
    arm: str,
    model: str,
    prompt: str,
    events: Events,
    key: str,
    *,
    server: dict | None = None,
    files: FileTools | None = None,
    limit: float = TIME_LIMIT,
) -> None:
    started = time.monotonic()
    cost, turns, usage_total = 0.0, 0, {"prompt_tokens": 0, "completion_tokens": 0}

    async def loop(system: str, tools: list[dict], call) -> None:
        nonlocal cost, turns
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        final = None
        error = None
        try:
            while turns < MAX_TURNS and time.monotonic() - started < limit:
                msg, usage = await asyncio.to_thread(chat, model, messages, tools, key)
                turns += 1
                cost += float(usage.get("cost") or 0)
                for k in usage_total:
                    usage_total[k] += int(usage.get(k) or 0)
                calls = msg.get("tool_calls") or []
                text = msg.get("content") or ""
                events.emit(
                    {
                        "type": "assistant",
                        "message": {
                            "content": ([{"type": "text", "text": text}] if text else [])
                            + [
                                {
                                    "type": "tool_use",
                                    "name": c["function"]["name"],
                                    "input": c["function"].get("arguments"),
                                }
                                for c in calls
                            ]
                        },
                    }
                )
                if not calls:
                    final = text
                    break
                messages.append({"role": "assistant", "content": text, "tool_calls": calls})
                for c in calls:
                    try:
                        args = json.loads(c["function"].get("arguments") or "{}")
                        out = await call(c["function"]["name"], args)
                    except json.JSONDecodeError:
                        out = "Error: the arguments were not valid JSON"
                    out = _clip(out)
                    events.emit(
                        {
                            "type": "user",
                            "message": {"content": [{"type": "tool_result", "content": out}]},
                        }
                    )
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
            if final is None:
                messages.append({"role": "user", "content": FINISH})
                msg, usage = await asyncio.to_thread(chat, model, messages, None, key)
                turns += 1
                cost += float(usage.get("cost") or 0)
                final = msg.get("content") or ""
        except RuntimeError as e:
            error = str(e)
        events.emit(
            {
                "type": "result",
                "subtype": "success" if error is None and final else "error",
                "is_error": error is not None or not final,
                "result": final or "",
                "error": error,
                "duration_ms": int((time.monotonic() - started) * 1000),
                "total_cost_usd": round(cost, 6),
                "num_turns": turns,
                "usage": usage_total,
            }
        )

    events.emit({"type": "system", "subtype": "init", "model": model, "arm": arm})
    if arm == "files":
        tools = files or FileTools()
        await loop(SYSTEM, tools.specs(), lambda n, a: asyncio.to_thread(tools.call, n, a))
        return

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=server["command"], args=[], env={**os.environ, **server["env"]}
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        listed = await session.list_tools()
        specs = [_mcp_spec(t) for t in listed.tools if t.name in READ_TOOLS]
        system = SYSTEM
        if init.instructions:
            system += "\n\n# Instructions from the minirag server\n\n" + init.instructions

        async def call(name: str, args: dict) -> str:
            if name not in READ_TOOLS:
                return f"Error: unknown tool {name}"
            result = await session.call_tool(name, args)
            text = "\n".join(getattr(c, "text", "") for c in result.content)
            return ("Error: " if result.isError else "") + text

        await loop(system, specs, call)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("minirag", "files"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--server-bin", type=Path, default=REPO / ".venv" / "bin" / "minirag-mcp")
    ap.add_argument("--db", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--container", default=CONTAINER)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    args = ap.parse_args()
    server = None
    if args.arm == "minirag":
        server = {
            "command": str(args.server_bin),
            "env": {
                "BASE_DIR": str(args.corpus),
                "DB_PATH": str(args.db),
                "HF_HUB_OFFLINE": "1",
            },
        }
    asyncio.run(
        answer(
            args.arm,
            args.model,
            args.prompt,
            Events(),
            api_key(),
            server=server,
            files=FileTools(args.container),
            limit=args.time_limit,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
