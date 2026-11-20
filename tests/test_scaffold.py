import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import minirag_mcp

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
SERVER_JSON = ROOT / "server.json"

SERVER_NAME = "io.github.sfrangulov/minirag-mcp"


def test_package_importable():
    assert minirag_mcp.__version__


def test_version_agrees_with_pyproject():
    """`__version__` is read from installed metadata; pyproject.toml declares it.

    Asserting against pyproject.toml instead of a literal is the point of the
    change that made `__version__` derived: a release bump edits one line and
    this test follows it, so there is nothing left to forget. It also catches
    the failure the indirection exists for — installed metadata that has
    drifted from the source tree, which would make `status` confidently report
    a version nobody is running — and the fallback in `__init__` firing when
    the package really is installed.
    """
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    assert minirag_mcp.__version__ == declared


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _server_json() -> dict:
    return json.loads(SERVER_JSON.read_text(encoding="utf-8"))


def test_server_json_versions_agree_with_pyproject():
    """server.json carries the version twice, and both must track pyproject.toml.

    The MCP Registry rejects a re-publish of a version it already has, and it
    verifies `packages[].version` against the artifact actually on PyPI. So a
    server.json left a release behind does not degrade quietly — it fails the
    publish step of a tag that has already shipped to PyPI, at which point the
    only fix is another release. bump-my-version rewrites both of these lines
    (see [[tool.bumpversion.files]] for server.json); this test is what catches
    a hand-edit that touched only one of them.
    """
    declared = _pyproject()["project"]["version"]
    server = _server_json()
    assert server["version"] == declared
    assert [p["version"] for p in server["packages"]] == [declared]


def test_server_json_names_the_published_package():
    """The registry entry has to point at *this* distribution, on PyPI.

    `identifier` is what a client resolves (`uvx <identifier>`), and the
    registry checks PyPI for an `mcp-name: <name>` marker in that package's
    README before it will accept the publish. Both sides are pinned here: the
    identifier against pyproject's own package name, and the server name
    against the literal the README marker spells out.
    """
    server = _server_json()
    assert server["name"] == SERVER_NAME
    assert [p["identifier"] for p in server["packages"]] == [_pyproject()["project"]["name"]]
    assert [p["registryType"] for p in server["packages"]] == ["pypi"]


def test_readme_carries_the_registry_ownership_marker():
    """PyPI ownership is proven by a marker in the README as published to PyPI.

    pyproject declares `readme = "README.md"`, so this file *is* the PyPI
    long description. The token must be followed by a boundary — newline,
    whitespace, an HTML tag, or the comment close — so the assertion below
    checks the exact `<!-- mcp-name: … -->` spelling rather than a substring
    that a stray trailing period would silently break. PyPI descriptions are
    immutable per version: lose this and the next release cannot publish.
    """
    readme = (ROOT / _pyproject()["project"]["readme"]).read_text(encoding="utf-8")
    assert f"<!-- mcp-name: {SERVER_NAME} -->" in readme


BUMP = Path(sys.executable).with_name("bump-my-version")


@pytest.mark.parametrize(
    ("current", "part", "expected"),
    [
        ("1.1.1", "patch", "1.1.2"),
        ("1.1.1", "major", "2.0.0"),
        ("2.0.0a1", "pre_n", "2.0.0a2"),
        ("2.0.0a2", "pre_l", "2.0.0b1"),
        ("2.0.0b1", "pre_l", "2.0.0rc1"),
        ("2.0.0rc1", "pre_l", "2.0.0"),
        ("2.0.0", "patch", "2.0.1"),
    ],
)
def test_bump_walks_pre_releases_to_the_final_release(current, part, expected):
    """release.yml compares the tag with the wheel filename, which carries the
    normalized PEP 440 form, so the result is `2.0.0a1` and not `2.0.0-alpha.1`."""
    proc = subprocess.run(
        [
            str(BUMP),
            "show",
            "--config-file",
            str(PYPROJECT),
            "--current-version",
            current,
            "--increment",
            part,
            "-f",
            "json",
            "new_version",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["new_version"] == expected
