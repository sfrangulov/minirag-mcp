"""Needs the [classify] extra and a local GGUF: MINIRAG_TEST_GGUF=<file> pytest -m classify"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from minirag_mcp.labels.model import LlamaChoiceModel, available

pytestmark = pytest.mark.classify

GGUF = os.environ.get("MINIRAG_TEST_GGUF")
OPTIONS = [
    ("transcript", "A verbatim record of a conversation, with speakers and timestamps."),
    ("minutes", "Meeting minutes: agenda, decisions and action items."),
    ("mail", "An email message with sender, recipient and subject."),
]
EVIDENCE = "File name: call.vtt\n[00:00–02:00] Anna: good morning\n[00:02] Ben: let us start"


def _need_model() -> Path:
    if not available():
        pytest.skip("the [classify] extra is not installed")
    if not GGUF or not Path(GGUF).is_file():
        pytest.skip("MINIRAG_TEST_GGUF does not name a file")
    return Path(GGUF)


@pytest.fixture(scope="module")
def llm():
    m = LlamaChoiceModel(_need_model(), 2.07)
    yield m
    m.close()


def test_probabilities_sum_to_one(llm):
    probs = llm.probabilities(EVIDENCE, "What kind of document is this?", OPTIONS)
    assert set(probs) == {"transcript", "minutes", "mail"}
    assert sum(probs.values()) == pytest.approx(1.0)


def test_a_transcript_ranks_transcript_first(llm):
    probs = llm.probabilities(EVIDENCE, "What kind of document is this?", OPTIONS)
    assert max(probs, key=probs.get) == "transcript"


def test_close_is_idempotent():
    m = LlamaChoiceModel(_need_model(), 2.07)
    m.close()
    m.close()


def test_loading_and_using_the_model_never_redirects_stdout(monkeypatch):
    """stdout is the MCP channel, written by the event loop while a sync thread loads."""
    gguf = _need_model()
    targets: list[int] = []
    real_dup2 = os.dup2

    def recording_dup2(fd, fd2, *args, **kwargs):
        targets.append(fd2)
        return real_dup2(fd, fd2, *args, **kwargs)

    monkeypatch.setattr(os, "dup2", recording_dup2)
    m = LlamaChoiceModel(gguf, 2.07)
    try:
        m.probabilities(EVIDENCE, "What kind of document is this?", OPTIONS)
    finally:
        m.close()
    assert 1 not in targets


def test_a_process_running_the_model_writes_nothing_to_stdout_and_no_load_log():
    gguf = _need_model()
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from minirag_mcp.labels.model import LlamaChoiceModel\n"
        "m = LlamaChoiceModel(Path(sys.argv[1]), 2.07)\n"
        f"m.probabilities({EVIDENCE!r}, 'What kind of document is this?', {OPTIONS!r})\n"
        "m.close()\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(gguf)], capture_output=True, timeout=300
    )
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    assert done.stdout == b""
    assert b"llama_model_loader" not in done.stderr
