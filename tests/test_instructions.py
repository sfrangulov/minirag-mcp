"""The server-level `instructions` string, and its trip through a real handshake."""

from __future__ import annotations

import pytest
from fastmcp import Client

from minirag_mcp.config import load_config
from minirag_mcp.instructions import (
    APPEND_ENV_VAR,
    MAX_INSTRUCTIONS_CHARS,
    SERVER_INSTRUCTIONS,
    build_instructions,
)
from minirag_mcp.server import create_app

# pytest-asyncio runs in auto mode (see pyproject) — bare `async def` tests are collected as-is.

CORPUS_NOTE = "This corpus is Russian-language financial specifications."


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


def test_instructions_stay_within_the_client_truncation_budget():
    """Claude Code cuts each server's instructions at exactly 2048 characters.

    Past that the text is not ignored, it is amputated mid-sentence and stamped
    `… [truncated]`. The built-in text is held well under the cap so a user's
    append has somewhere to land inside the same budget; the headroom assertion
    is what turns "we left room" from an intention into a fact.
    """
    assert len(SERVER_INSTRUCTIONS) < MAX_INSTRUCTIONS_CHARS
    headroom = MAX_INSTRUCTIONS_CHARS - len(SERVER_INSTRUCTIONS)
    assert headroom >= 400, f"only {headroom} chars left for {APPEND_ENV_VAR}"


@pytest.mark.parametrize("raw", ["", "   ", "\n\n"])
def test_blank_append_reads_as_unset(raw):
    """`RAG_INSTRUCTIONS_APPEND=` in a client config must not add a dangling paragraph."""
    assert build_instructions({APPEND_ENV_VAR: raw}) == SERVER_INSTRUCTIONS


def test_append_is_trimmed_not_reflowed():
    assert build_instructions({APPEND_ENV_VAR: f"  {CORPUS_NOTE}\n"}).endswith(f"\n\n{CORPUS_NOTE}")
