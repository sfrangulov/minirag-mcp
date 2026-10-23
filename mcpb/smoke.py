"""Install and start a packed bundle the way Claude Desktop does, then talk MCP to it.

    uv run python mcpb/smoke.py dist/minirag-mcp.mcpb [--uv PATH | --uv-version V] [--search]

For a `uv` bundle, Claude Desktop unpacks the archive into a folder of its own, runs
`uv sync --quiet` there, and starts the server as `<its uv> <mcp_config.args>` with
that folder as the working directory. The child's environment is mcp_config.env on
top of the few variables the MCP SDK passes through (HOME, LOGNAME, PATH, SHELL, TERM
and USER outside Windows), not the user's shell environment. This script repeats each
of those steps, with PATH cut down to the system directories, and fails unless the
server answers `initialize`, lists exactly the tools the manifest declares, and
reports through `status` the folder the user picked and the bundle's own version.

--search also syncs a one-document folder and searches it, which loads the embedding
model (a ~220 MB download on a machine that has never run minirag-mcp).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

INHERITED = (
    (
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
        "PROGRAMFILES",
    )
    if os.name == "nt"
    else ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER")
)
GUI_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# What Claude Desktop downloads and uses when the machine has no uv of its own, which
# is the case for anyone who has never installed developer tools.
DESKTOP_UV = "0.9.7"

DOCUMENT = (
    "# Engagement letters\n\nEngagement letters are kept for seven years under code QX-4471.\n"
)
TERM = "QX-4471"

STARTUP_SECONDS = 180
CALL_SECONDS = 60
SYNC_SECONDS = 900


def fail(message: str, server: Server | None = None) -> None:
    if server is not None:
        server.stop()
        message += "\n\nserver stderr:\n" + server.stderr_tail()
    sys.exit(f"FAIL: {message}")


def substitute(value: str, variables: dict[str, str]) -> str:
    def one(match: re.Match) -> str:
        name = match.group(1)
        if name not in variables:
            fail(f"the manifest uses ${{{name}}}, which this smoke test cannot supply")
        return variables[name]

    return re.sub(r"\$\{([^}]+)\}", one, value)


class Server:
    def __init__(self, cmd: list[str], cwd: Path, env: dict[str, str], stderr_path: Path):
        self.stderr_path = stderr_path
        self.stderr = stderr_path.open("wb")
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr
        )
        self.lines: queue.Queue[bytes] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.next_id = 0

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(b"")

    def send(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message).encode() + b"\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict, timeout: float = CALL_SECONDS) -> dict:
        self.next_id += 1
        self.send({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                fail(f"no reply to {method} within {timeout:g}s", self)
            if not line:
                fail(f"the server exited ({self.proc.wait()}) before replying to {method}", self)
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                fail(f"the server wrote a non-JSON line to stdout: {line[:200]!r}", self)
            if message.get("id") != self.next_id:
                continue
            if "error" in message:
                fail(f"{method} returned an error: {message['error']}", self)
            return message["result"]

    def call(self, tool: str, arguments: dict | None = None) -> dict:
        result = self.request("tools/call", {"name": tool, "arguments": arguments or {}})
        if result.get("isError"):
            fail(f"{tool} failed: {result.get('content')}", self)
        if isinstance(result.get("structuredContent"), dict):
            return result["structuredContent"]
        return json.loads(result["content"][0]["text"])

    def stop(self) -> int:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.stderr.close()
        return self.proc.returncode

    def stderr_tail(self, lines: int = 40) -> str:
        text = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])


def search(server: Server, docs: Path) -> None:
    job = server.call("sync_start")["jobId"]
    deadline = time.monotonic() + SYNC_SECONDS
    while (state := server.call("sync_status", {"jobId": job}))["state"] not in (
        "succeeded",
        "failed",
    ):
        if time.monotonic() > deadline:
            fail(f"sync did not finish within {SYNC_SECONDS}s: {state}", server)
        time.sleep(1)
    if state["state"] != "succeeded" or state["counts"]["ingested"] != 1:
        fail(f"sync did not index the one document: {state}", server)
    hits = server.call("query_documents", {"query": TERM})["results"]
    expected = str((docs / "letters.md").resolve())
    if not hits or hits[0]["source"] != expected:
        fail(f"searching for {TERM} did not return {expected} first: {hits[:1]}", server)
    print(f"sync and search: {TERM} found in {hits[0]['source']}")


def smoke(bundle: Path, uv: str, with_search: bool, workdir: Path) -> None:
    # Desktop installs under a folder named "Claude Extensions"; the space is the point.
    extension = workdir / "Claude Extensions" / "local.mcpb.minirag-mcp"
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(extension)
    manifest = json.loads((extension / "manifest.json").read_text(encoding="utf-8"))
    docs = workdir / "Client files"
    docs.mkdir()
    (docs / "letters.md").write_text(DOCUMENT, encoding="utf-8")

    base_env = {name: os.environ[name] for name in INHERITED if name in os.environ}
    if os.name != "nt":
        base_env["PATH"] = GUI_PATH

    lock = (extension / "uv.lock").read_bytes()
    install = subprocess.run(
        [uv, "sync", "--quiet"], cwd=extension, env=base_env, capture_output=True, text=True
    )
    if install.returncode != 0:
        fail(f"uv sync exited {install.returncode}:\n{install.stdout}{install.stderr}")
    if (extension / "uv.lock").read_bytes() != lock:
        fail("uv sync rewrote the shipped uv.lock instead of installing from it")
    print(f"install: uv sync --quiet in {extension}")

    (key,) = manifest["user_config"]
    variables = {"__dirname": str(extension), f"user_config.{key}": str(docs)}
    config = manifest["server"]["mcp_config"]
    args = [substitute(arg, variables) for arg in config["args"]]
    env = {**base_env, **{k: substitute(v, variables) for k, v in config["env"].items()}}
    print(f"launch: {uv} {' '.join(args)}")

    server = Server([uv, *args], extension, env, workdir / "server.stderr")
    started = time.monotonic()
    init = server.request(
        "initialize",
        {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "mcpb-smoke", "version": "0"},
        },
        timeout=STARTUP_SECONDS,
    )
    server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    if init["serverInfo"]["name"] != "minirag-mcp":
        fail(f"initialize answered as {init['serverInfo']}", server)
    print(f"initialize: {init['serverInfo']} after {time.monotonic() - started:.1f}s")

    tools = {t["name"] for t in server.request("tools/list", {})["tools"]}
    declared = {t["name"] for t in manifest["tools"]}
    if tools != declared:
        fail(
            f"the server lists {sorted(tools - declared)} beyond the manifest, "
            f"and the manifest declares {sorted(declared - tools)} the server lacks",
            server,
        )
    print(f"tools/list: the {len(tools)} tools the manifest declares")

    status = server.call("status")
    problems = [k for k in ("configError", "indexError") if k in status]
    if problems:
        fail(f"status reports {problems}: {status}", server)
    if status["roots"] != [str(docs.resolve())]:
        fail(f"status roots are {status['roots']}, not the picked folder {docs}", server)
    if status["version"] != manifest["version"]:
        fail(f"status version {status['version']} != manifest {manifest['version']}", server)
    print(
        f"status: version {status['version']}, roots {status['roots']}, "
        f"chunkCount {status['chunkCount']}, dbPath {status['dbPath']}"
    )

    if with_search:
        search(server, docs)

    code = server.stop()
    print(f"shutdown: exit {code} after stdin closed")


def fetch_uv(version: str) -> str:
    probe = "import uv; print(uv.find_uv_bin())"
    cmd = [shutil.which("uv") or "uv", "tool", "run", "--from", f"uv=={version}"]
    proc = subprocess.run([*cmd, "python", "-c", probe], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"could not fetch uv {version}:\n{proc.stderr}")
    return proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path)
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--uv", help="the uv executable to install and launch with")
    which.add_argument(
        "--uv-version",
        default=DESKTOP_UV,
        help=f"fetch this uv release and use it (default {DESKTOP_UV})",
    )
    parser.add_argument("--search", action="store_true", help="also sync and search")
    args = parser.parse_args()
    uv = str(Path(args.uv or fetch_uv(args.uv_version)).resolve())
    version = subprocess.run([uv, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"uv: {uv} ({version})")
    with tempfile.TemporaryDirectory(
        prefix="minirag-mcpb-smoke-", ignore_cleanup_errors=True
    ) as tmp:
        smoke(args.bundle.resolve(), uv, args.search, Path(tmp).resolve())
    print("PASS")


if __name__ == "__main__":
    main()
