"""The server-level `instructions` string, and its trip through a real handshake.

Also the second channel the citation policy travels on. `instructions` is
optional for a client and several drop it — Claude Desktop among them, which is
where the policy was observed to have no effect at all — while tool descriptions
are passed to the model by every client. So the minimum of the policy is stated
again in `query_documents`'s description, and the last two tests here hold the
two statements to the same rules: one asserts the description a client is
actually handed carries the policy, the other fails if either text is edited
into disagreeing with the other.
"""

from __future__ import annotations

import re

import pytest
from fastmcp import Client

from minirag_mcp.config import load_config
from minirag_mcp.instructions import (
    APPEND_ENV_VAR,
    MAX_BUILTIN_INSTRUCTIONS_CHARS,
    MAX_INSTRUCTIONS_CHARS,
    SERVER_INSTRUCTIONS,
    build_instructions,
)
from minirag_mcp.server import create_app

#: `build_instructions` joins the built-in text and the append with a blank line.
APPEND_SEPARATOR_CHARS = 2

# pytest-asyncio runs in auto mode (see pyproject) — bare `async def` tests are collected as-is.

CORPUS_NOTE = "This corpus is Russian-language financial specifications."

#: The parts of the citation policy that must read the same way wherever it is
#: stated — the server `instructions`, which several clients drop, and the
#: `query_documents` description, which every client passes to the model. Exact
#: phrases rather than paraphrases: an assertion loose enough to survive a
#: reworded copy would not catch the two texts drifting apart, which is the only
#: failure worth a test here. Claude Code is handed both at once.
CITATION_INVARIANTS = (
    "so the user can verify it",  # verification, not a claim that the prose is true
    "[1], [2]",  # the marker style,
    "Sources list",  # ...and the end-of-answer list the markers point into
    "the document, not the chunk",  # the granularity
    "`title` and `source` verbatim",  # ...and the two fields that identify one
    "never a markdown link or file://",  # both arrive broken in every client checked
)


def _flat(text: str) -> str:
    """Collapse to one line, so a phrase broken across a docstring wrap still matches.

    The instructions are written with backslash continuations and arrive
    unwrapped; a tool description is a docstring and arrives with its source
    line breaks intact. Normalising both means the tests pin the wording rather
    than where the author happened to wrap it.
    """
    return " ".join(text.split())


@pytest.fixture
def app(tmp_path, fake_embedder):
    def make(env: dict[str, str] | None = None):
        root = tmp_path / "docs"
        root.mkdir(exist_ok=True)
        cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
        return create_app(cfg, embedder=fake_embedder, env=env or {})

    return make


async def test_instructions_reach_the_client_through_the_handshake(app):
    """The whole point of the field: a connecting client is handed it at initialize.

    Asserted against `initialize_result` rather than against the FastMCP object,
    because everything downstream of the constructor — FastMCP, the in-memory
    transport, the MCP protocol layer — is between us and the model, and a field
    that any of them dropped would be an unused constant.
    """
    async with Client(app()) as c:
        assert c.initialize_result.instructions == SERVER_INSTRUCTIONS


async def test_env_var_appends_to_the_instructions_the_client_receives(app):
    async with Client(app({APPEND_ENV_VAR: CORPUS_NOTE})) as c:
        received = c.initialize_result.instructions
    assert received.startswith(SERVER_INSTRUCTIONS)
    assert received.endswith(CORPUS_NOTE)
    # Appended as its own paragraph, not run onto the end of the last block.
    assert received == f"{SERVER_INSTRUCTIONS}\n\n{CORPUS_NOTE}"


async def test_instructions_survive_a_broken_configuration(app, fake_embedder):
    """A server that cannot serve still tells the client what it is.

    `status` is documented to answer through a configuration error; the routing
    policy is what lets the model reach `status` in the first place, so it is
    built from the environment rather than from Config.
    """
    mcp = create_app(None, config_error="BASE_DIRS must be a JSON array", embedder=fake_embedder)
    async with Client(mcp) as c:
        assert c.initialize_result.instructions == SERVER_INSTRUCTIONS


async def test_the_citation_policy_reaches_the_client_through_the_handshake(app):
    """The citation block, checked at the far end of the connection.

    Asserted against the string the client was handed rather than against
    SERVER_INSTRUCTIONS, which would be the constant matched against itself and
    would still pass with the policy deleted. The substrings are the parts the
    policy would be useless without, not its phrasing.
    """
    async with Client(app()) as c:
        received = c.initialize_result.instructions

    assert "<citations>" in received, "no citation policy in the instructions"
    citations = received.split("<citations>", 1)[1].split("</citations>", 1)[0]

    # Inline markers plus an end-of-answer list — the two halves of the format.
    assert "[1]" in citations and "Sources:" in citations
    # Keyed to the document. `chunkIndex`/`parentId` are internal identifiers
    # that locate nothing for a human opening the file, so they are not citable.
    assert "`title`" in citations and "`source`" in citations
    assert "chunkIndex" not in citations and "parentId" not in citations
    # Plain paths. Claude Desktop denylists the `file:` scheme, Claude Code
    # hyperlinks only http/https, Cursor hands file:// to the system handler —
    # and a scheme-less markdown link renders as a broken relative URL.
    assert "plain text" in citations
    assert "file://" in citations and "markdown link" in citations
    assert "](" not in citations, "the policy or its example contains a markdown link"
    # The worked example: marker, title, bare absolute path. Without a literal
    # instance the model emits the markers and drops the path from the list.
    assert re.search(r"^\[1\] .+ /\S+$", citations, re.M), "no worked example of a Sources line"
    # Citations are for the user's verification, not a claim that the prose is
    # true — measured support rates are far too low for the stronger reading.
    assert "verify" in citations


async def test_the_citation_rule_reaches_the_client_in_the_query_tool_description(app):
    """The channel that survives a client which drops `instructions`.

    Asserted against `list_tools()` rather than against the function's
    `__doc__`, because the docstring is only the input: FastMCP is what turns it
    into the `description` field, and a policy that lived in the source but not
    on the wire is exactly the failure this test exists to catch — the same one
    the `instructions` field hit in Claude Desktop.
    """
    async with Client(app()) as c:
        descriptions = {t.name: t.description or "" for t in await c.list_tools()}
    desc = _flat(descriptions["query_documents"])

    # The format, in the two halves an answer needs to be written in it.
    assert "[1], [2]" in desc and "Sources list" in desc
    # Keyed to the document: `chunkIndex`/`parentId` are internal identifiers,
    # and this tool's own results are where a model would otherwise reach for one.
    assert "the document, not the chunk" in desc
    assert "`title` and `source` verbatim" in desc
    # Plain paths. A `file://` URL or a scheme-less markdown link arrives broken
    # in every client checked, and models emit both spontaneously.
    assert "never a markdown link or file://" in desc
    assert "](" not in desc, "the description contains a markdown link"
    # For the user's verification, not as a claim that the answer is true.
    assert "so the user can verify it" in desc


async def test_the_two_statements_of_the_citation_policy_agree(app):
    """One policy, two channels — Claude Code receives both at once.

    Both texts are read off the wire, so this fails on a divergence introduced
    at either end: trimming the instructions block, or rewording the tool
    description. That is the point of pinning phrases rather than paraphrases —
    a client handed a contradiction has no way to resolve it, and the two texts
    are edited months apart.
    """
    async with Client(app()) as c:
        instructions = _flat(c.initialize_result.instructions or "")
        descriptions = {t.name: t.description or "" for t in await c.list_tools()}
    description = _flat(descriptions["query_documents"])

    for phrase in CITATION_INVARIANTS:
        assert phrase in instructions, f"{phrase!r} left the server instructions"
        assert phrase in description, f"{phrase!r} left the query_documents description"


def test_instructions_stay_within_the_client_truncation_budget():
    """Claude Code cuts each server's instructions at exactly 2048 characters.

    Past that the text is not ignored, it is amputated mid-sentence and stamped
    `… [truncated]`. This is the hard fact about the client; the two tests below
    are how the budget is actually divided up underneath it.
    """
    assert len(SERVER_INSTRUCTIONS) < MAX_INSTRUCTIONS_CHARS


def test_the_builtin_text_leaves_room_for_a_user_append():
    """The built-in text gets 1700 of the 2048; the rest is the user's.

    The 2048 cap alone cannot enforce this. The built-in text can grow to 2047
    and still satisfy it while leaving a user's RAG_INSTRUCTIONS_APPEND nowhere
    to land — the append would push the whole string past the cap and be cut
    mid-sentence, which is worse than never offering the setting. So the
    reserve needs a ceiling of its own, and this is it: every block added here
    is paid for by tightening another. The citation policy was paid for that
    way.
    """
    used = len(SERVER_INSTRUCTIONS)
    assert used <= MAX_BUILTIN_INSTRUCTIONS_CHARS, (
        f"built-in instructions are {used} chars, "
        f"{used - MAX_BUILTIN_INSTRUCTIONS_CHARS} over the {MAX_BUILTIN_INSTRUCTIONS_CHARS} "
        f"reserved for them; trim a block rather than raising the ceiling"
    )


def test_an_append_filling_the_reserve_still_fits_under_the_client_cap():
    """The reserve is a real ~350 characters, end to end through build_instructions.

    Checked with an append that spends all of it, because the failure being
    guarded against is off-by-a-paragraph: the blank line the append is joined
    with is part of what the client counts.
    """
    reserve = MAX_INSTRUCTIONS_CHARS - MAX_BUILTIN_INSTRUCTIONS_CHARS - APPEND_SEPARATOR_CHARS
    assert reserve >= 340, f"only {reserve} chars reserved for {APPEND_ENV_VAR}"
    built = build_instructions({APPEND_ENV_VAR: "x" * reserve})
    assert len(built) <= MAX_INSTRUCTIONS_CHARS


@pytest.mark.parametrize("raw", ["", "   ", "\n\n"])
def test_blank_append_reads_as_unset(raw):
    """`RAG_INSTRUCTIONS_APPEND=` in a client config must not add a dangling paragraph."""
    assert build_instructions({APPEND_ENV_VAR: raw}) == SERVER_INSTRUCTIONS


def test_append_is_trimmed_not_reflowed():
    assert build_instructions({APPEND_ENV_VAR: f"  {CORPUS_NOTE}\n"}).endswith(f"\n\n{CORPUS_NOTE}")
