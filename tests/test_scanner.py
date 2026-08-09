from pathlib import Path

from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.ingest.pipeline import file_sha256
from minirag_mcp.ingest.scanner import ScanEntry, compute_diff, compute_states, scan_roots
from minirag_mcp.store import SourceInfo


def info(
    source, *, source_type="file", file_hash="", mtime=0.0, scheme=SCHEME_VERSION, ocr_engine=""
):
    return SourceInfo(
        source=source,
        source_type=source_type,
        title="T",
        chunk_count=1,
        file_hash=file_hash,
        mtime=mtime,
        scheme_version=scheme,
        ocr_engine=ocr_engine,
    )


def make_tree(root: Path):
    (root / "sub" / "deep").mkdir(parents=True)
    (root / "a.md").write_text("a")
    (root / "sub" / "b.txt").write_text("b")
    (root / "sub" / "deep" / "c.pdf").write_bytes(b"%PDF-fake")
    (root / "skip.py").write_text("nope")  # not whitelisted
    (root / ".hidden.md").write_text("hidden file")  # hidden file
    (root / ".git").mkdir()
    (root / ".git" / "x.md").write_text("in hidden dir")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "y.md").write_text("in skip dir")


def test_scan_recursive_whitelist_and_skips(tmp_path):
    make_tree(tmp_path)
    entries = scan_roots([tmp_path])
    names = [e.path.relative_to(tmp_path).as_posix() for e in entries]
    assert names == ["a.md", "sub/b.txt", "sub/deep/c.pdf"]
    assert all(isinstance(e, ScanEntry) and e.size > 0 for e in entries)


def test_diff_new_changed_unchanged_deleted(tmp_path):
    make_tree(tmp_path)
    entries = scan_roots([tmp_path])
    a = tmp_path / "a.md"
    b = tmp_path / "sub" / "b.txt"
    indexed = [
        info(str(a), file_hash=file_sha256(a), mtime=a.stat().st_mtime),  # unchanged (mtime match)
        info(str(b), file_hash="stale-hash", mtime=-1.0),  # changed (hash differs)
        info(str(tmp_path / "gone.md")),  # deleted from disk
        info("https://x.io/p", source_type="url"),  # never deleted by sync
        info("note-1", source_type="data"),  # never deleted by sync
    ]
    diff = compute_diff(entries, indexed, max_file_size=10**9)
    assert diff.unchanged == [a]
    assert b in diff.to_ingest and (tmp_path / "sub" / "deep" / "c.pdf") in diff.to_ingest
    assert diff.to_delete == [str(tmp_path / "gone.md")]


def test_diff_mtime_differs_but_hash_same_is_unchanged(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("stable content")
    entries = [ScanEntry(path=f, size=f.stat().st_size, mtime=999.0)]
    indexed = [info(str(f), file_hash=file_sha256(f), mtime=1.0)]
    diff = compute_diff(entries, indexed, max_file_size=10**9)
    assert diff.unchanged == [f] and diff.to_ingest == []


def test_diff_oversized(tmp_path):
    f = tmp_path / "big.md"
    f.write_text("0123456789ABCDEF")
    entries = scan_roots([tmp_path])
    diff = compute_diff(entries, [], max_file_size=4)
    assert diff.oversized == [f] and diff.to_ingest == []


def test_diff_scope_limits_both_sides(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    fin = tmp_path / "in" / "f.md"
    fin.write_text("x")
    entries = scan_roots([tmp_path])
    indexed = [info(str(tmp_path / "out" / "gone.md"))]
    diff = compute_diff(entries, indexed, max_file_size=10**9, scope=tmp_path / "in")
    assert diff.to_ingest == [fin]
    assert diff.to_delete == []  # out/gone.md is outside scope: not touched


def test_compute_states(tmp_path):
    from minirag_mcp.ingest.scanner import compute_states

    a = tmp_path / "a.md"
    a.write_text("ingested content")
    b = tmp_path / "b.md"
    b.write_text("changed on disk")
    c = tmp_path / "c.md"
    c.write_text("never ingested")
    entries = scan_roots([tmp_path])
    indexed = [
        info(str(a), file_hash=file_sha256(a), mtime=a.stat().st_mtime),
        info(str(b), file_hash="old-hash", mtime=-1.0),
        info("note-1", source_type="data"),
        info("https://x.io/p", source_type="url"),
    ]
    states = {s.source: s.state for s in compute_states(entries, indexed)}
    assert states[str(a)] == "ingested"
    assert states[str(b)] == "stale"
    assert states[str(c)] == "not_ingested"
    assert states["note-1"] == "ingested" and states["https://x.io/p"] == "ingested"


def test_a_scheme_stale_source_is_not_listed_as_ingested(tmp_path):
    """`status` says the index is stale and to re-sync, and `compute_diff` re-ingests
    the file — but the listing called it "ingested", so an agent picking work off the
    listing saw nothing to do. The two views have to agree."""
    from minirag_mcp.ingest.scanner import compute_diff, compute_states

    a = tmp_path / "a.md"
    a.write_text("unchanged on disk, cut by the old scheme")
    entries = scan_roots([tmp_path])
    indexed = [
        info(str(a), file_hash=file_sha256(a), mtime=a.stat().st_mtime, scheme=SCHEME_VERSION - 1),
        info("note-1", source_type="data", scheme=SCHEME_VERSION - 1),
    ]
    states = {s.source: s.state for s in compute_states(entries, indexed)}
    assert states[str(a)] == "stale_scheme"
    assert states["note-1"] == "stale_scheme"
    # and that agrees with what a sync would actually do with the file
    assert compute_diff(entries, indexed, max_file_size=10**9).to_ingest == [a]


def test_a_file_changed_on_disk_and_scheme_stale_reports_the_disk_change(tmp_path):
    from minirag_mcp.ingest.scanner import compute_states

    b = tmp_path / "b.md"
    b.write_text("changed on disk too")
    entries = scan_roots([tmp_path])
    indexed = [info(str(b), file_hash="old-hash", mtime=-1.0, scheme=SCHEME_VERSION - 1)]
    assert [s.state for s in compute_states(entries, indexed)] == ["stale"]


def test_compute_states_ignores_data_source_colliding_with_file_path(tmp_path):
    from minirag_mcp.ingest.scanner import compute_states

    f = tmp_path / "a.md"
    f.write_text("on disk, never ingested")
    entries = scan_roots([tmp_path])
    indexed = [info(str(f), source_type="data", file_hash="unrelated", mtime=-1.0)]
    states = compute_states(entries, indexed)
    disk_rows = [s for s in states if s.source_type == "file"]
    assert [s.state for s in disk_rows] == ["not_ingested"]


def test_compute_states_threads_ocr_engine(tmp_path):
    """`FileState.ocr_engine` must come from the indexed `prior`, in every branch that
    builds one: unchanged-on-disk, changed-on-disk, and the data/url tail loop."""
    from minirag_mcp.ingest.scanner import compute_states

    a = tmp_path / "a.md"
    a.write_text("ocr'd content")
    b = tmp_path / "b.md"
    b.write_text("ocr'd content, changed on disk")
    entries = scan_roots([tmp_path])
    indexed = [
        info(str(a), file_hash=file_sha256(a), mtime=a.stat().st_mtime, ocr_engine="rapidocr"),
        info(str(b), file_hash="stale-hash", mtime=-1.0, ocr_engine="rapidocr"),
        info("note-1", source_type="data", ocr_engine="rapidocr"),
    ]
    states = {s.source: s for s in compute_states(entries, indexed)}
    assert states[str(a)].state == "ingested" and states[str(a)].ocr_engine == "rapidocr"
    assert states[str(b)].state == "stale" and states[str(b)].ocr_engine == "rapidocr"
    assert states["note-1"].ocr_engine == "rapidocr"


def test_scan_roots_skips_symlink_escaping_root(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret_no_extension"
    secret.write_text("PRIVATE")
    (root / "innocent.md").symlink_to(secret)
    (root / "real.md").write_text("real content")
    assert [e.path.name for e in scan_roots([root])] == ["real.md"]


def test_scan_roots_keeps_symlink_pointing_inside_root(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    target = root / "sub" / "target.md"
    target.write_text("inside the root")
    (root / "alias.md").symlink_to(target)
    # The on-disk path is reported (not the resolved target), and the target is
    # deduped away because it resolves to the same file.
    assert [e.path.name for e in scan_roots([root])] == ["alias.md"]


def test_scan_roots_dedupes_overlapping_roots(tmp_path):
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "i.md").write_text("x")
    entries = scan_roots([tmp_path, inner])
    assert [e.path for e in entries] == [inner / "i.md"]


def test_diff_never_deletes_an_indexed_source_this_install_cannot_read(tmp_path, monkeypatch):
    """A capability that went away is not a file that went away.

    Without the [ocr] extra images leave the scan whitelist, so an indexed image is
    absent from the scan for exactly the same reason a deleted file is — and the diff
    used to propose deleting it, shrinking the corpus with nothing on disk having
    changed."""
    from minirag_mcp import ocr

    doc = tmp_path / "doc.md"
    doc.write_text("plain text")
    img = tmp_path / "scan.png"
    img.write_bytes(b"...")
    indexed = [
        info(str(doc), file_hash=file_sha256(doc), mtime=doc.stat().st_mtime),
        info(str(img), file_hash=file_sha256(img), mtime=img.stat().st_mtime),
    ]

    monkeypatch.setattr(ocr, "available", lambda: False)
    diff = compute_diff(scan_roots([tmp_path]), indexed, max_file_size=10**9)

    assert diff.to_delete == []
    assert diff.unreadable == [str(img)]
    assert diff.unchanged == [doc]


def test_diff_deletes_a_vanished_image_while_ocr_is_installed(tmp_path, monkeypatch):
    """Retention is about the whitelist shrinking, not about images: with the extra in
    place an image that really left the disk is still deleted."""
    from minirag_mcp import ocr

    monkeypatch.setattr(ocr, "available", lambda: True)
    (tmp_path / "doc.md").write_text("plain text")
    indexed = [info(str(tmp_path / "gone.png"))]

    diff = compute_diff(scan_roots([tmp_path]), indexed, max_file_size=10**9)

    assert diff.to_delete == [str(tmp_path / "gone.png")]
    assert diff.unreadable == []


def test_diff_deletes_an_image_that_left_the_disk_even_without_the_extra(tmp_path, monkeypatch):
    """Retention answers "why did the scan not find it"; the disk answers "is it still
    there". A file the user really deleted must not be kept forever behind a message
    about the file *type*, which is true and beside the point."""
    from minirag_mcp import ocr

    (tmp_path / "doc.md").write_text("plain text")
    indexed = [info(str(tmp_path / "gone.png"))]

    monkeypatch.setattr(ocr, "available", lambda: False)
    diff = compute_diff(scan_roots([tmp_path]), indexed, max_file_size=10**9)

    assert diff.to_delete == [str(tmp_path / "gone.png")]
    assert diff.unreadable == []


def test_states_report_an_indexed_source_this_install_cannot_read(tmp_path, monkeypatch):
    """`status` counts its chunks and `query_documents` returns them, so a listing that
    omits the file entirely contradicts the rest of the tool."""
    from minirag_mcp import ocr

    doc = tmp_path / "doc.md"
    doc.write_text("plain text")
    img = tmp_path / "scan.png"
    img.write_bytes(b"...")
    indexed = [
        info(str(doc), file_hash=file_sha256(doc), mtime=doc.stat().st_mtime),
        info(str(img), ocr_engine="rapidocr"),
    ]

    monkeypatch.setattr(ocr, "available", lambda: False)
    states = compute_states(scan_roots([tmp_path]), indexed)

    by_source = {s.source: s for s in states}
    assert by_source[str(img)].state == "unreadable"
    assert by_source[str(img)].chunk_count == 1
    assert by_source[str(img)].ocr_engine == "rapidocr"
    assert by_source[str(doc)].state == "ingested"


def test_states_omit_an_unreadable_source_whose_file_is_gone(tmp_path, monkeypatch):
    """The same split the diff makes: with the file gone there is nothing to list, and
    the next sync from any install deletes it."""
    from minirag_mcp import ocr

    (tmp_path / "doc.md").write_text("plain text")

    monkeypatch.setattr(ocr, "available", lambda: False)
    states = compute_states(scan_roots([tmp_path]), [info(str(tmp_path / "gone.png"))])

    assert [s.source for s in states] == [str(tmp_path / "doc.md")]


def test_scan_roots_sees_images_only_with_ocr(tmp_path, monkeypatch):
    from minirag_mcp import ocr

    (tmp_path / "doc.md").write_text("# hi")
    (tmp_path / "scan.png").write_bytes(b"...")

    monkeypatch.setattr(ocr, "available", lambda: False)
    names = {e.path.name for e in scan_roots([tmp_path])}
    assert names == {"doc.md"}

    monkeypatch.setattr(ocr, "available", lambda: True)
    names = {e.path.name for e in scan_roots([tmp_path])}
    assert names == {"doc.md", "scan.png"}
