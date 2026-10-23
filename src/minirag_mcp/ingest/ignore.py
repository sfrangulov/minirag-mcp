"""`.miniragignore`: gitignore rules per document root.

One file at the top of each root, matched against paths relative to that root, by
pathspec's implementation of git's own rules rather than an approximation of them.

A file that is there and cannot be read stops everything under its root. Scanning past
it would index exactly what it was written to keep out, and nothing would say so.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pathspec

IGNORE_FILE = ".miniragignore"


class IgnoreFileError(Exception):
    """An ignore file exists but cannot be used."""


class IgnoredFileError(ValueError):
    """A file the caller named explicitly is excluded by an ignore file."""


@dataclass(frozen=True)
class Rules:
    file: Path
    count: int
    _spec: pathspec.GitIgnoreSpec

    def excludes(self, rel: str, is_dir: bool) -> bool:
        return self._spec.match_file(rel + "/" if is_dir else rel)


def load(root: Path) -> Rules | None:
    path = Path(root) / IGNORE_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, UnicodeDecodeError) as e:
        if isinstance(e, OSError) and not _listable(root):
            # The folder itself cannot be read: the scan reports it as unlisted, and
            # nothing in it can be indexed, so there is nothing for its rules to keep out.
            return None
        reason = e.strerror if isinstance(e, OSError) and e.strerror else str(e)
        raise IgnoreFileError(
            f"cannot read {path}: {reason}. Nothing under {root} is scanned until it can "
            "be read, so that what it excludes is never indexed by accident; fix or "
            "remove it."
        ) from e
    lines = text.splitlines()
    try:
        spec = pathspec.GitIgnoreSpec.from_lines(lines)
    except Exception as e:
        raise IgnoreFileError(f"{path} has a rule that cannot be parsed: {e}") from e
    count = sum(1 for line in lines if line.strip() and not line.lstrip().startswith("#"))
    return Rules(path, count, spec)


def _listable(folder: Path) -> bool:
    try:
        with os.scandir(folder):
            return True
    except OSError:
        return False


def owner(path: Path, roots: Sequence[Path]) -> Path | None:
    """The configured root `path` lies under, the innermost when roots nest."""
    found = [Path(r) for r in roots if Path(r) == path or Path(r) in path.parents]
    return max(found, key=lambda r: len(r.parts)) if found else None


def excluded_by(path: Path, roots: Sequence[Path]) -> Path | None:
    """The ignore file that excludes `path`, or None.

    A folder a rule excludes takes everything below it along, as in git, so every
    folder on the way down is checked as well as the path itself.
    """
    root = owner(path, roots)
    if root is None or (rules := load(root)) is None:
        return None
    rel = path.relative_to(root).parts
    for depth in range(1, len(rel)):
        if rules.excludes("/".join(rel[:depth]), is_dir=True):
            return rules.file
    if rel and rules.excludes("/".join(rel), is_dir=path.is_dir()):
        return rules.file
    return None
