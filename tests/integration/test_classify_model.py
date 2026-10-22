"""Needs the [classify] extra and a local GGUF: MINIRAG_TEST_GGUF=<file> pytest -m classify"""

import os
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


@pytest.fixture(scope="module")
def llm():
    if not available():
        pytest.skip("the [classify] extra is not installed")
    if not GGUF or not Path(GGUF).is_file():
        pytest.skip("MINIRAG_TEST_GGUF does not name a file")
    m = LlamaChoiceModel(Path(GGUF), 2.07)
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
    if not available():
        pytest.skip("the [classify] extra is not installed")
    if not GGUF or not Path(GGUF).is_file():
        pytest.skip("MINIRAG_TEST_GGUF does not name a file")
    m = LlamaChoiceModel(Path(GGUF), 2.07)
    m.close()
    m.close()
