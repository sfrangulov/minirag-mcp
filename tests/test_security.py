import socket
import threading
import time

import pytest

from minirag_mcp import security
from minirag_mcp.security import (
    SecurityError,
    URLResolutionError,
    check_url,
    resolve_in_roots,
)

PUBLIC_IP = "93.184.216.34"  # an address literal: checked without touching a resolver


def _resolves_to(*addresses: str):
    """A getaddrinfo stand-in — every test here stays offline."""

    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port or 80)) for a in addresses]

    return fake


def test_inside_root_ok(tmp_path):
    f = tmp_path / "docs" / "a.md"
    f.parent.mkdir()
    f.write_text("x")
    assert resolve_in_roots(str(f), [tmp_path]) == f.resolve()


def test_root_itself_ok(tmp_path):
    assert resolve_in_roots(str(tmp_path), [tmp_path]) == tmp_path.resolve()


def test_outside_root_rejected(tmp_path):
    other = tmp_path / "in"
    other.mkdir()
    with pytest.raises(SecurityError, match="outside"):
        resolve_in_roots("/etc/passwd", [other])


def test_dotdot_traversal_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(SecurityError, match="outside"):
        resolve_in_roots(str(root / ".." / "escape.md"), [root])


def test_relative_path_rejected_when_absolute_required(tmp_path):
    with pytest.raises(SecurityError, match="absolute"):
        resolve_in_roots("relative/a.md", [tmp_path])


def test_relative_ok_when_allowed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.md").write_text("x")
    expected = (tmp_path / "a.md").resolve()
    assert resolve_in_roots("a.md", [tmp_path], require_absolute=False) == expected


def test_symlink_escape_rejected(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.md"
    secret.write_text("s")
    link = root / "link.md"
    link.symlink_to(secret)
    with pytest.raises(SecurityError, match="outside"):
        resolve_in_roots(str(link), [root])


def test_multiple_roots_second_matches(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    f = b / "doc.md"
    f.write_text("x")
    assert resolve_in_roots(str(f), [a, b]) == f.resolve()


@pytest.mark.parametrize("url", [f"http://{PUBLIC_IP}/p", f"https://{PUBLIC_IP}/p"])
def test_public_address_literal_ok(url):
    check_url(url)  # no raise


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "data:text/html,hi",
        "ftp://x.io",
        "x.io/nope",
    ],
)
def test_url_bad_scheme(url):
    with pytest.raises(SecurityError, match="http"):
        check_url(url)


@pytest.mark.parametrize(
    "host,reason",
    [
        ("127.0.0.1", "loopback"),
        ("169.254.169.254", "link-local"),  # cloud instance metadata
        ("10.0.0.1", "private"),
        ("192.168.1.1", "private"),
        ("[::1]", "loopback"),
        ("[::ffff:127.0.0.1]", "loopback"),  # IPv4-mapped IPv6
        ("[::ffff:169.254.169.254]", "link-local"),
        ("[::ffff:10.0.0.1]", "private"),
        ("0.0.0.0", "unspecified"),
    ],
)
def test_blocked_address_literal_rejected(host, reason):
    with pytest.raises(SecurityError) as exc:
        check_url(f"http://{host}:8080/latest/meta-data/")
    msg = str(exc.value)
    assert host.strip("[]") in msg  # names the host
    assert reason in msg  # names the reason
    assert "ALLOW_PRIVATE_URLS" in msg  # names the way out


def test_ipv4_mapped_addresses_are_judged_by_the_address_they_carry():
    """`::ffff:x.x.x.x` must get whatever verdict `x.x.x.x` gets — blocked when the
    IPv4 address is blocked, and *fetchable* when it is not.

    The second half is the load-bearing one, and it is why this test exists rather
    than an unwrap in `_blocked_kind`. Every property the check reads already looks
    through the wrapper (verified identical on 3.11.13, 3.12.11, 3.13.6 and 3.14.6),
    so no unwrapping of our own is needed. On an interpreter that stopped doing so, a
    mapped public address would land inside `::ffff:0:0/96` — listed private — and be
    refused: fail-closed, but wrong, and this assertion is what would catch it.
    """
    check_url(f"http://[::ffff:{PUBLIC_IP}]/p")  # no raise
    with pytest.raises(SecurityError, match="loopback"):
        check_url("http://[::ffff:127.0.0.1]/p")


def test_blocked_address_literal_needs_no_resolver(monkeypatch):
    """An address literal is judged as written — no DNS round trip to be poisoned."""

    def never(*args, **kwargs):
        raise AssertionError("getaddrinfo must not be called for an address literal")

    monkeypatch.setattr(socket, "getaddrinfo", never)
    with pytest.raises(SecurityError, match="loopback"):
        check_url("http://127.0.0.1/admin")


def test_name_resolving_to_a_public_address_ok(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _resolves_to(PUBLIC_IP))
    check_url("https://example.com/docs")  # no raise


def test_every_resolved_address_is_checked_not_just_the_first(monkeypatch):
    """A name under an attacker's control can answer with a public address first."""
    monkeypatch.setattr(socket, "getaddrinfo", _resolves_to(PUBLIC_IP, "127.0.0.1"))
    with pytest.raises(SecurityError) as exc:
        check_url("http://rebind.example.com/")
    msg = str(exc.value)
    assert "rebind.example.com" in msg and "127.0.0.1" in msg and "loopback" in msg


def test_allow_private_permits_a_blocked_host():
    check_url("http://127.0.0.1:8080/admin", allow_private=True)
    check_url("http://169.254.169.254/latest/meta-data/", allow_private=True)


def test_allow_private_does_not_widen_the_scheme_rule():
    with pytest.raises(SecurityError, match="http"):
        check_url("file:///etc/passwd", allow_private=True)


def test_allow_private_skips_resolution_entirely(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("nothing to resolve when the host rule is off")

    monkeypatch.setattr(socket, "getaddrinfo", never)
    check_url("http://internal.wiki/", allow_private=True)


def test_unresolvable_host_is_a_fetch_error_not_a_security_verdict(monkeypatch):
    def boom(*args, **kwargs):
        raise socket.gaierror(8, "nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    assert not issubclass(URLResolutionError, SecurityError)
    with pytest.raises(URLResolutionError) as exc:
        check_url("http://nowhere.invalid/x")
    assert "nowhere.invalid" in str(exc.value)


def test_resolution_cannot_hang_forever(monkeypatch):
    entered = threading.Event()

    def slow(*args, **kwargs):
        entered.set()
        time.sleep(0.5)
        raise AssertionError("the caller should have stopped waiting long before this")

    monkeypatch.setattr(socket, "getaddrinfo", slow)
    monkeypatch.setattr(security, "DNS_TIMEOUT_SECONDS", 0.01)
    started = time.monotonic()
    with pytest.raises(URLResolutionError, match="[Tt]imed out"):
        check_url("http://slow.example.com/")
    assert time.monotonic() - started < 0.4
    assert entered.is_set()


def test_url_without_a_host_rejected():
    with pytest.raises(SecurityError, match="host"):
        check_url("http:///etc/passwd")
