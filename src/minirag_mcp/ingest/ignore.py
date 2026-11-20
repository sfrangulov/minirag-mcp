"""`.miniragignore`: gitignore rules per document root, and `RAG_EXCLUDE`.

One file at the top of each root, matched against paths relative to that root by
pathspec's implementation of git's rules. When roots nest, a path is excluded if the
rules of any root above it exclude it, so neither root's walk can get round the other's.

`RAG_EXCLUDE` patterns are held to every root as if written at the top of its file, ahead
of it: the file's own rules come last, so it can re-include with `!` what a pattern took
out. They exist so that a shared folder can be indexed in place without a file written
into it.

Matching ignores case and Unicode composition, as git does on macOS by default: a rule
written `private/` has to keep out `Private/`, and a folder name macOS stored decomposed
has to match the same rule typed composed. Each path is read both ways, with and without
case, and either reading excluding it is enough: an exclusion rule that fails to apply
puts private text in the index, so where the two readings differ, this one excludes.

Roots are recognised by the folder itself, not by how its path was typed: on macOS
`docs` and `Docs` are one folder, and nested roots written in different case must
still apply each other's rules.

A file that is there and cannot be read stops the scan of its root. Scanning past it
would index exactly what it was written to keep out, and nothing would say so.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pathspec

IGNORE_FILE = ".miniragignore"


class IgnoreFileError(Exception):
    """An ignore file exists but cannot be used."""


class IgnoredFileError(ValueError):
    """A path the caller named explicitly is excluded by an ignore file."""


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


EXCLUDE_VAR = "RAG_EXCLUDE"


@dataclass(frozen=True)
class Rules:
    file: Path | None
    count: int
    _exact: pathspec.GitIgnoreSpec
    _folded: pathspec.GitIgnoreSpec
    _lines: tuple[str, ...] = ()
    _configured: int = 0  # the first this many of `_lines` are RAG_EXCLUDE patterns

    def excludes(self, rel: str, is_dir: bool) -> bool:
        return self.source(rel, is_dir) is not None

    def source(self, rel: str, is_dir: bool) -> Path | str | None:
        """What excludes `rel`: this rule file, or a description of the RAG_EXCLUDE
        pattern. None when nothing does."""
        rel = _nfc(rel) + ("/" if is_dir else "")
        for spec, text in ((self._exact, rel), (self._folded, rel.lower())):
            hit = spec.check_file(text)
            if hit.include and hit.index is not None:
                if hit.index < self._configured:
                    return f"{EXCLUDE_VAR} pattern {self._lines[hit.index]!r}"
                break
        else:
            return None
        return self.file


def _listable(folder: Path) -> bool:
    try:
        with os.scandir(folder):
            return True
    except OSError:
        return False


def check_patterns(patterns: Sequence[str]) -> None:
    """Raise ValueError naming the first RAG_EXCLUDE pattern the parser rejects."""
    for pattern in patterns:
        try:
            pathspec.GitIgnoreSpec.from_lines([_nfc(pattern)])
            pathspec.GitIgnoreSpec.from_lines([_nfc(pattern).lower()])
        except Exception as e:
            raise ValueError(f"{pattern!r} cannot be parsed: {e}") from e


def load(root: Path, *, strict: bool = False, exclude: Sequence[str] = ()) -> Rules | None:
    """The rules at the top of `root`, or None when it has none.

    `exclude` are the RAG_EXCLUDE patterns, ahead of the file's own lines.

    Not strict, as a scan reads it, a root that cannot even be listed has no rules: the
    scan reports that folder itself, and nothing in it can be indexed. Strict, as an
    explicit ingest reads it, any file that is there and cannot be read is an error.
    """
    path = Path(root) / IGNORE_FILE
    configured = tuple(_nfc(p) for p in exclude if p.strip())
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        if os.path.islink(path):
            raise IgnoreFileError(
                f"{path} is a link to a file that is not there, so its rules cannot be "
                "applied; fix or remove it."
            ) from None
        return _rules(None, configured, [])
    except NotADirectoryError:
        return None
    except (OSError, UnicodeDecodeError) as e:
        if isinstance(e, OSError) and not strict and not _listable(root):
            return None
        reason = e.strerror if isinstance(e, OSError) and e.strerror else str(e)
        raise IgnoreFileError(
            f"cannot read {path}: {reason}. Nothing under {root} is scanned until it can "
            "be read, so that what it excludes is never indexed by accident; fix or "
            "remove it."
        ) from e
    lines = [_nfc(line) for line in text.splitlines()]
    try:
        return _rules(path, configured, lines)
    except Exception as e:
        raise IgnoreFileError(f"{path} has a rule that cannot be parsed: {e}") from e


def _rules(file: Path | None, configured: tuple[str, ...], lines: list[str]) -> Rules | None:
    if file is None and not configured:
        return None
    # Attribution is by index, which holds only because no configured item is empty
    # (pathspec drops those): `load` filters them.
    every = [*configured, *lines]
    exact = pathspec.GitIgnoreSpec.from_lines(every)
    folded = pathspec.GitIgnoreSpec.from_lines([line.lower() for line in every])
    count = sum(1 for line in lines if line.strip() and not line.lstrip().startswith("#"))
    return Rules(file, count, exact, folded, tuple(every), len(configured))


class Ignorer:
    """The rules of every configured root, each file read once for the life of this."""

    def __init__(self, roots: Sequence[Path], *, strict: bool = False, exclude: Sequence[str] = ()):
        self._strict = strict
        self._exclude = tuple(p for p in exclude if p.strip())
        self._cache: dict[Path, Rules | None | IgnoreFileError] = {}
        self._ids: dict[str, tuple[int, int] | None] = {}
        self._roots: dict[tuple[int, int], Path] = {}
        for r in roots:
            if (key := self._id(Path(r))) is not None:
                self._roots.setdefault(key, Path(r))

    def _id(self, p: Path) -> tuple[int, int] | None:
        key = os.fspath(p)
        if key not in self._ids:
            try:
                st = os.stat(p)
                self._ids[key] = (st.st_dev, st.st_ino)
            except OSError:
                self._ids[key] = None
        return self._ids[key]

    def broken(self) -> dict[Path, str]:
        """Each root whose ignore file cannot be used, with why."""
        out = {}
        for root in self._roots.values():
            try:
                self.rules(root)
            except IgnoreFileError as e:
                out[root] = str(e)
        return out

    def root_of(self, folder: Path) -> Path | None:
        """The configured root `folder` is, however its path is spelled."""
        return self._roots.get(self._id(folder))

    def rules(self, root: Path) -> Rules | None:
        if root not in self._cache:
            try:
                self._cache[root] = load(root, strict=self._strict, exclude=self._exclude)
            except IgnoreFileError as e:
                self._cache[root] = e
        got = self._cache[root]
        if isinstance(got, IgnoreFileError):
            raise got
        return got

    def _owners(self, path: Path) -> list[tuple[Path, Path]]:
        """Each configured root above `path` (or `path` itself), with the folder in
        `path`'s own spelling that is that root."""
        out = []
        for folder in (path, *path.parents):
            if (root := self.root_of(folder)) is not None:
                out.append((root, folder))
        return out

    def matches(self, path: Path, is_dir: bool) -> Path | str | None:
        """What excludes `path` itself (an ignore file, or a RAG_EXCLUDE pattern), checked
        against every root above it. A walk calls this top-down, so the folders above were
        checked on the way."""
        for root, folder in self._owners(path):
            if folder == path:
                continue
            rules = self.rules(root)
            if rules is None:
                continue
            if found := rules.source(path.relative_to(folder).as_posix(), is_dir):
                return found
        return None

    def excluded_by(self, path: Path) -> Path | str | None:
        """What excludes `path` or any folder above it: an ignore file, or a RAG_EXCLUDE
        pattern, described.

        A folder a rule excludes takes everything below it along, as in git, which is
        also what a walk that never enters it does.
        """
        for root, folder in self._owners(path):
            rules = self.rules(root)
            if rules is None:
                continue
            rel = path.relative_to(folder).parts
            for depth in range(1, len(rel)):
                if found := rules.source("/".join(rel[:depth]), is_dir=True):
                    return found
            if rel and (found := rules.source("/".join(rel), is_dir=path.is_dir())):
                return found
        return None


def excluded_by(
    path: Path, roots: Sequence[Path], exclude: Sequence[str] = ()
) -> Path | str | None:
    """What excludes a path the caller named, read strictly."""
    return Ignorer(roots, strict=True, exclude=exclude).excluded_by(path)


def scope_excluded_by(
    scope: Path, roots: Sequence[Path], exclude: Sequence[str] = ()
) -> Path | str | None:
    """What excludes a sync or listing scope, or None.

    Not strict, and silent about a rule file it cannot read: the scan that follows
    reports that root as held, which is the answer such a scope has always had.
    """
    try:
        return Ignorer(roots, exclude=exclude).excluded_by(scope)
    except IgnoreFileError:
        return None
