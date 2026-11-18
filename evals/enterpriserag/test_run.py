from __future__ import annotations

import json
from pathlib import Path

from common import Question
from run import build_prompt, claude_command, done, limited

Q = Question("qst_0001", "basic", "What is the upload limit?", ())
CORPUS = Path("/data/corpus")


def _value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_minirag_arm_has_no_built_in_tools_and_only_minirag_read_tools():
    cmd = claude_command("minirag", "p", Path("/x/mcp.json"), "claude-opus-5-5", CORPUS)
    assert _value(cmd, "--tools") == ""
    allowed = _value(cmd, "--allowedTools").split(",")
    assert allowed and all(t.startswith("mcp__minirag__") for t in allowed)
    assert not any(
        t.endswith(("ingest_file", "ingest_data", "ingest_url", "delete_file", "sync_start"))
        for t in allowed
    )
    assert "--add-dir" not in cmd


def test_files_arm_has_no_mcp_server_and_no_write_tool():
    cmd = claude_command("files", "p", None, "claude-opus-5-5", CORPUS)
    assert "--mcp-config" not in cmd and "--strict-mcp-config" in cmd
    assert _value(cmd, "--add-dir") == str(CORPUS)
    allowed = _value(cmd, "--allowedTools").split(",")
    assert "Bash" not in allowed
    assert not any(t in allowed for t in ("Write", "Edit", "NotebookEdit"))
    assert all(not t.startswith("Bash(") or t.endswith(":*)") for t in allowed)


def test_both_arms_use_the_same_model_and_isolation():
    a = claude_command("minirag", "p", Path("/x"), "m", CORPUS)
    b = claude_command("files", "p", None, "m", CORPUS)
    for flag in ("--model", "--setting-sources"):
        assert _value(a, flag) == _value(b, flag)
    for flag in ("--disable-slash-commands", "--no-session-persistence"):
        assert flag in a and flag in b


def test_prompts_differ_in_exactly_one_paragraph():
    a = build_prompt(Q, "## Knowledge Sources\nslack/", "minirag", CORPUS).split("\n\n")
    b = build_prompt(Q, "## Knowledge Sources\nslack/", "files", CORPUS).split("\n\n")
    assert len(a) == len(b)
    assert sum(x != y for x, y in zip(a, b, strict=True)) == 1
    assert Q.question in a[3]


def _write(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def test_done_needs_a_result_or_a_timeout_marker(tmp_path):
    _write(tmp_path / "q1.jsonl", [{"type": "system"}])
    assert not done(tmp_path, "q1")
    (tmp_path / "q1.timeout").write_text("600")
    assert done(tmp_path, "q1")
    _write(tmp_path / "q2.jsonl", [{"type": "result", "is_error": False, "result": "x"}])
    assert done(tmp_path, "q2")
    assert not done(tmp_path, "q3")


def test_a_usage_limit_is_not_an_answer(tmp_path):
    hit = {"type": "result", "is_error": True, "result": "Claude AI usage limit reached|1763500000"}
    assert limited(hit)
    assert not limited({"type": "result", "is_error": True, "result": "Tool failed"})
    _write(tmp_path / "q.jsonl", [hit])
    assert not done(tmp_path, "q")
