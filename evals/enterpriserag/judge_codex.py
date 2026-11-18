"""The benchmark's judge LLM, served by `codex exec` on the owner's ChatGPT account.

The benchmark's evaluators reach their LLM only through `get_llm(tools=None)` /
`get_cheap_llm`, iterate `generate(messages)` and parse JSON out of the text. This
supplies an object with that one method; `install` swaps it in on every loaded module of
the benchmark's `src` package, which stays untouched at its pinned commit.

`gpt-5.4`, the benchmark's default judge, is refused on a ChatGPT account. CODEX_HOME
must point at a home holding only the owner's auth.json and a minimal config: the
owner's own Codex config carries plugins, MCP servers, hooks and memories that must not
reach a judge prompt.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

ATTEMPTS = 3
TIMEOUT = 300
STATS = {"calls": 0, "failures": 0, "retries": 0, "seconds": 0.0}
_stats_lock = threading.Lock()


def flatten(messages) -> str:
    parts = []
    for m in messages:
        role = getattr(m, "role", None) or m["role"]
        content = getattr(m, "content", None)
        content = m["content"] if content is None else content
        parts.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(parts)


def codex_command(model: str, effort: str, out_file: Path) -> list[str]:
    return [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-o",
        str(out_file),
        "-",
    ]


class CodexLLM:
    def __init__(self, model: str, effort: str = "medium"):
        self.model = model
        self.effort = effort

    def _once(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="erb-judge-") as work:
            out = Path(work) / "last.txt"
            proc = subprocess.run(
                codex_command(self.model, self.effort, out),
                input=prompt,
                cwd=work,
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
            if proc.returncode != 0 or not out.exists():
                tail = (proc.stderr or proc.stdout or "")[-400:]
                raise RuntimeError(f"codex exec exited {proc.returncode}: {tail}")
            return out.read_text(encoding="utf-8")

    def generate(self, messages) -> Iterator[str]:
        prompt = flatten(messages)
        started = time.monotonic()
        last: Exception | None = None
        for attempt in range(ATTEMPTS):
            try:
                text = self._once(prompt)
            except Exception as e:  # rate limit, auth refresh, timeout
                last = e
                with _stats_lock:
                    STATS["retries"] += 1
                time.sleep(5 * (attempt + 1))
                continue
            with _stats_lock:
                STATS["calls"] += 1
                STATS["seconds"] += time.monotonic() - started
            yield text
            return
        with _stats_lock:
            STATS["failures"] += 1
        raise RuntimeError(f"judge call failed {ATTEMPTS} times: {last}")


def install(model: str, cheap_model: str, effort: str = "medium") -> int:
    """Replace the benchmark's LLM factories; the number of module attributes swapped."""

    def get_llm(tools=None, quiet=False, reasoning_level="medium", model_override=None, **_):
        if tools:
            raise RuntimeError("the Codex judge serves tool-free calls only")
        return CodexLLM(model, effort)

    def get_cheap_llm(tools=None, quiet=False, reasoning_level="medium", **_):
        if tools:
            raise RuntimeError("the Codex judge serves tool-free calls only")
        return CodexLLM(cheap_model, "low")

    swapped = 0
    for name, module in list(sys.modules.items()):
        if not (name == "src" or name.startswith("src.")) or module is None:
            continue
        for attr, fn in (("get_llm", get_llm), ("get_cheap_llm", get_cheap_llm)):
            if hasattr(module, attr):
                setattr(module, attr, fn)
                swapped += 1
    if not os.environ.get("CODEX_HOME"):
        raise SystemExit("CODEX_HOME is not set: the judge must not run in the owner's Codex home")
    return swapped
