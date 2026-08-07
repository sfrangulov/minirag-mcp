"""Document-root containment and URL scheme checks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse


class SecurityError(Exception):
    pass


def resolve_in_roots(
    path_str: str, roots: Sequence[Path], *, require_absolute: bool = True
) -> Path:
    p = Path(path_str).expanduser()
    if require_absolute and not p.is_absolute():
        raise SecurityError(f"Path must be absolute: {path_str}")
    real = p.resolve()  # follows symlinks; the check below sees the true target
    for root in roots:
        real_root = root.resolve()
        if real == real_root or real_root in real.parents:
            return real
    raise SecurityError(f"Path outside configured document roots: {path_str}")


def check_url_scheme(url: str) -> None:
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise SecurityError(
            f"Only http/https URLs are allowed, got scheme {scheme or '(none)'!r}"
        )
