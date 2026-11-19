from __future__ import annotations

import judge_codex
import pytest
from judge_codex import CodexLLM, codex_command, flatten


def test_flatten_keeps_roles_and_order():
    msgs = [{"role": "system", "content": "be strict"}, {"role": "user", "content": "judge"}]
    assert flatten(msgs) == "SYSTEM:\nbe strict\n\nUSER:\njudge"


def test_codex_command_is_read_only_and_ephemeral(tmp_path):
    cmd = codex_command("gpt-5.6-sol", "medium", tmp_path / "o")
    assert cmd[:2] == ["codex", "exec"]
    assert "--ephemeral" in cmd and cmd[cmd.index("-s") + 1] == "read-only"
    assert cmd[cmd.index("-m") + 1] == "gpt-5.6-sol"
    assert cmd[-1] == "-"


def _fake_codex(tmp_path, monkeypatch, script: str):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "codex"
    fake.write_text("#!/bin/sh\n" + script)
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    monkeypatch.setattr(judge_codex.time, "sleep", lambda s: None)
    for k in judge_codex.STATS:
        monkeypatch.setitem(judge_codex.STATS, k, 0)


WRITE_OUT = """
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; fi
  shift
done
cat > /dev/null
printf '{"aligned": true}' > "$out"
"""


def test_generate_returns_the_last_message(tmp_path, monkeypatch):
    _fake_codex(tmp_path, monkeypatch, WRITE_OUT)
    text = "".join(CodexLLM("m").generate([{"role": "user", "content": "x"}]))
    assert text == '{"aligned": true}'
    assert judge_codex.STATS["calls"] == 1 and judge_codex.STATS["failures"] == 0


def test_three_failures_raise_and_are_counted(tmp_path, monkeypatch):
    _fake_codex(tmp_path, monkeypatch, "echo 'usage limit' >&2\nexit 1\n")
    with pytest.raises(RuntimeError, match="usage limit"):
        "".join(CodexLLM("m").generate([{"role": "user", "content": "x"}]))
    assert judge_codex.STATS["failures"] == 1 and judge_codex.STATS["retries"] == 3
