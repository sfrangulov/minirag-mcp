import tomllib
from pathlib import Path

import minirag_mcp

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


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
