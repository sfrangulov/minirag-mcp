import plistlib
from pathlib import Path

import minirag_mcp.cli as cli

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs" / "launchd" / "local.minirag-mcp.plist"


def _filled(raw: str) -> bytes:
    for placeholder, value in {
        "__UVX__": "/opt/uvx",
        "__PYTHON__": "/opt/python3",
        "__VERSION__": "2.0.0a1",
        "__HOME__": "/Users/u",
    }.items():
        raw = raw.replace(placeholder, value)
    return raw.encode()


def test_the_launch_agent_template_runs_serve():
    raw = TEMPLATE.read_text()
    plist = plistlib.loads(_filled(raw))
    assert plist["Label"] == "local.minirag-mcp" and plist["RunAtLoad"]
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["ProgramArguments"] == [
        "/opt/uvx",
        "--python",
        "/opt/python3",
        "--from",
        "minirag-mcp[ocr]==2.0.0a1",
        "minirag-mcp",
        "serve",
    ]
    assert "serve" in cli.app.resolved_commands()
    log = "/Users/u/Library/Logs/minirag-mcp.log"
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == log
    assert "launchctl kill HUP" in raw and "launchctl bootstrap" in raw
    assert "launchctl bootout" in raw
    assert "__" not in _filled(raw).decode()
