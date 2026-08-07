"""Document-root containment and URL validation."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from urllib.parse import urlparse

DNS_TIMEOUT_SECONDS = 5.0

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# Checked in order, so the reported reason is the most specific one that applies:
# `is_private` is also true of loopback, link-local and unspecified addresses.
_BLOCKED_KINDS = (
    ("loopback", "is_loopback"),
    ("link-local", "is_link_local"),
    ("unspecified", "is_unspecified"),
    ("private", "is_private"),
    ("reserved", "is_reserved"),
)


class SecurityError(Exception):
    pass


class URLResolutionError(Exception):
    """A URL's host could not be resolved. A fetch problem, not a security verdict."""


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


def _blocked_kind(ip: IPAddress) -> str | None:
    """Name why `ip` must not be fetched from, or None if it is fetchable.

    An IPv4-mapped IPv6 address (`::ffff:127.0.0.1`) is judged by the IPv4 address it
    carries: on Python 3.11/3.12 none of the properties below see through the wrapper.
    """
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return next((kind for kind, attr in _BLOCKED_KINDS if getattr(ip, attr)), None)


def _refusal(host: str, detail: str) -> str:
    return (
        f"Refusing to fetch from host {host!r}: it {detail}. Set ALLOW_PRIVATE_URLS=1 "
        f"if this server should be allowed to reach private and local addresses."
    )


def _resolve_host(host: str) -> list[IPAddress]:
    """Resolve `host` to every address it maps to, bounded by DNS_TIMEOUT_SECONDS.

    getaddrinfo takes no timeout of its own, so it runs in a worker thread the caller
    stops waiting on: a wedged resolver then costs one abandoned thread rather than a
    server that never answers again.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minirag-dns")
    try:
        future = pool.submit(socket.getaddrinfo, host, None, type=socket.SOCK_STREAM)
        try:
            infos = future.result(timeout=DNS_TIMEOUT_SECONDS)
        except FutureTimeoutError as e:
            raise URLResolutionError(
                f"Timed out after {DNS_TIMEOUT_SECONDS:g}s resolving host {host!r}"
            ) from e
        except OSError as e:  # gaierror and friends
            raise URLResolutionError(f"Could not resolve host {host!r}: {e}") from e
    finally:
        pool.shutdown(wait=False)
    addresses = [ipaddress.ip_address(info[4][0]) for info in infos]
    if not addresses:
        raise URLResolutionError(f"Host {host!r} resolved to no addresses")
    return addresses


def check_url(url: str, *, allow_private: bool = False) -> None:
    """Reject a URL this server must not fetch.

    Two rules. The scheme must be http or https, because markitdown's `convert_uri`
    would otherwise read `file:`/`data:` URIs and bypass the document-root boundary.
    And the host must not be an address on this machine or on a private network: the
    URL is usually chosen by an LLM that may be acting on text from an already-indexed
    document, which makes an unrestricted host a prompt-injection path to cloud
    metadata (169.254.169.254) and to internal services (localhost:8080/admin).

    A host given as an address literal is judged as written; a name is resolved and
    rejected if **any** of its addresses is blocked, since the fetch would pick one of
    them. Resolution failure raises URLResolutionError, not SecurityError — a name that
    does not resolve is a broken fetch, not an attack.

    `allow_private` (ALLOW_PRIVATE_URLS) turns the host rule off for a user who
    deliberately indexes an internal wiki. It also skips resolution: with no verdict
    to reach, there is nothing to resolve.
    """
    try:
        parsed = urlparse(url)
        scheme, host = parsed.scheme.lower(), parsed.hostname
    except ValueError as e:  # e.g. an unclosed IPv6 literal
        raise SecurityError(f"Malformed URL {url!r}: {e}") from e
    if scheme not in ("http", "https"):
        raise SecurityError(f"Only http/https URLs are allowed, got scheme {scheme or '(none)'!r}")
    if allow_private:
        return
    if not host:
        raise SecurityError(f"URL has no host to check: {url!r}")

    try:
        literal: IPAddress | None = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        kind = _blocked_kind(literal)
        if kind:
            raise SecurityError(_refusal(host, f"is a {kind} address"))
        return

    for ip in _resolve_host(host):
        kind = _blocked_kind(ip)
        if kind:
            raise SecurityError(_refusal(host, f"resolves to {ip}, a {kind} address"))
