import json
import re
from pathlib import Path

from fastmcp import Client

from minirag_mcp.server import create_app

MANIFEST = json.loads(
    (Path(__file__).resolve().parent.parent / "mcpb" / "manifest.json").read_text(encoding="utf-8")
)


async def test_manifest_declares_exactly_the_tools_the_server_serves():
    """Claude Desktop shows the manifest's tool list on the install page, before the
    server has ever run, so a tool added to or removed from server.py must be added to
    or removed from mcpb/manifest.json too. The bundle smoke test checks the same
    thing against a running server, but only where Node and the network are at hand.
    """
    async with Client(create_app(None, config_error="not needed to list tools")) as c:
        served = {t.name for t in await c.list_tools()}
    assert {t["name"] for t in MANIFEST["tools"]} == served


def test_the_folder_the_user_picks_is_the_document_root():
    """BASE_DIR comes from a required, single-folder picker.

    An optional picker left empty would root the server at its working directory,
    which inside Desktop is the extension's own install folder. A multi-folder picker
    yields a list, and the MCPB host substitutes lists only into `args`, never into
    an environment variable, so BASE_DIR would arrive as the literal placeholder.
    """
    env = MANIFEST["server"]["mcp_config"]["env"]
    key = re.fullmatch(r"\$\{user_config\.(\w+)\}", env["BASE_DIR"]).group(1)
    picker = MANIFEST["user_config"][key]
    assert picker["type"] == "directory"
    assert picker["required"] is True
    assert not picker.get("multiple", False)
