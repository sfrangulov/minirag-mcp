import pytest

from minirag_mcp.security import SecurityError, check_url_scheme, resolve_in_roots


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


@pytest.mark.parametrize("url", ["http://x.io/p", "https://x.io/p"])
def test_url_ok(url):
    check_url_scheme(url)  # no raise


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
        check_url_scheme(url)
