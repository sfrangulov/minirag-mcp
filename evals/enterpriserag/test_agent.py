from __future__ import annotations

import asyncio
import io
import json

import agent
import pytest
from agent import Events, FileTools, answer


def test_paths_stay_inside_the_corpus():
    t = FileTools(root="/corpus")
    assert t.resolve(None) == "/corpus"
    assert t.resolve("slack/eng") == "/corpus/slack/eng"
    assert t.resolve("/corpus/gmail/x.txt") == "/corpus/gmail/x.txt"
    assert t.resolve("/slack") == "/corpus/slack"
    for bad in ("../etc/passwd", "slack/../../etc", "/corpus/../root"):
        with pytest.raises(ValueError):
            t.resolve(bad)


def test_a_path_outside_the_corpus_is_an_error_not_a_call(monkeypatch):
    t = FileTools(root="/corpus")
    monkeypatch.setattr(t, "_sh", lambda *a, **k: pytest.fail("no command may run"))
    assert t.call("read", {"path": "../../etc/passwd"}).startswith("Error:")


class _Tools(FileTools):
    def __init__(self):
        super().__init__(root="/corpus")
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        return "slack/x/dsid_" + "a" * 32 + "__m.txt"


def _run(replies, monkeypatch, limit=100.0):
    seen = []

    def fake_chat(model, messages, tools, key):
        seen.append({"messages": [dict(m) for m in messages], "tools": tools})
        msg = replies.pop(0)
        return msg, {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 2}

    monkeypatch.setattr(agent, "chat", fake_chat)
    out = io.StringIO()
    tools = _Tools()
    asyncio.run(answer("files", "m", "Q?", Events(out), "k", files=tools, limit=limit))
    events = [json.loads(line) for line in out.getvalue().splitlines()]
    return events, seen, tools


def _tool_call(name, args, cid="c1"):
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_the_loop_runs_tools_then_returns_the_final_text(monkeypatch):
    replies = [
        {"content": "", "tool_calls": [_tool_call("grep", {"pattern": "limit"})]},
        {"content": "The limit is 10 MiB.\n\n- dsid_" + "a" * 32},
    ]
    events, seen, tools = _run(replies, monkeypatch)
    assert tools.calls == [("grep", {"pattern": "limit"})]
    assert seen[1]["messages"][-1]["role"] == "tool"
    result = events[-1]
    assert result["type"] == "result" and not result["is_error"]
    assert result["result"].startswith("The limit is 10 MiB.")
    assert result["num_turns"] == 2 and result["total_cost_usd"] == 0.002
    uses = [
        b
        for e in events
        if e["type"] == "assistant"
        for b in e["message"]["content"]
        if b["type"] == "tool_use"
    ]
    assert [u["name"] for u in uses] == ["grep"]


def test_out_of_time_forces_a_final_answer_without_tools(monkeypatch):
    replies = [{"content": "Found nothing more; the answer is X.\n\n- dsid_" + "b" * 32}]
    events, seen, _ = _run(replies, monkeypatch, limit=0.0)
    assert seen[0]["tools"] is None
    assert seen[0]["messages"][-1]["content"] == agent.FINISH
    assert events[-1]["result"].startswith("Found nothing more")


def test_a_failed_model_call_is_an_error_result(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("OpenRouter call failed: 402")

    monkeypatch.setattr(agent, "chat", broken)
    out = io.StringIO()
    asyncio.run(answer("files", "m", "Q?", Events(out), "k", files=_Tools(), limit=100.0))
    result = json.loads(out.getvalue().splitlines()[-1])
    assert result["is_error"] and "402" in result["error"]
