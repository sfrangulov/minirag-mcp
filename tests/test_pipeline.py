import pytest

from minirag_mcp.config import load_config
from minirag_mcp.ingest.pipeline import (
    EmptyDocumentError,
    FileTooLargeError,
    Pipeline,
    UnsupportedFormatError,
    file_sha256,
)
from minirag_mcp.security import SecurityError
from minirag_mcp.store import Store


@pytest.fixture
def pipe(tmp_path, fake_embedder):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    store = Store(tmp_path / ".minirag" / "lancedb", dim=fake_embedder.dim)
    return Pipeline(store, fake_embedder, cfg), store, tmp_path


def test_ingest_file_roundtrip(pipe):
    p, store, root = pipe
    f = root / "doc.md"
    f.write_text("# Title\n\n" + "Sentence about topic. " * 30, encoding="utf-8")
    res = p.ingest_file(f)
    assert res.source == str(f) and res.title == "Title" and res.chunk_count >= 1
    info = store.get_source(str(f))
    assert info.source_type == "file"
    assert info.file_hash == file_sha256(f)
    assert info.mtime == f.stat().st_mtime


def test_reingest_replaces(pipe):
    p, store, root = pipe
    f = root / "doc.md"
    f.write_text("# A\n\n" + "words " * 200, encoding="utf-8")
    p.ingest_file(f)
    first = store.get_source(str(f)).chunk_count
    f.write_text("# A\n\nshort now", encoding="utf-8")
    res = p.ingest_file(f)
    assert res.chunk_count <= first
    assert store.get_source(str(f)).chunk_count == res.chunk_count


def test_unsupported_extension(pipe):
    p, _, root = pipe
    f = root / "script.py"
    f.write_text("print('hi')")
    with pytest.raises(UnsupportedFormatError):
        p.ingest_file(f)


def test_file_too_large(pipe, tmp_path, fake_embedder):
    from minirag_mcp.config import load_config

    cfg = load_config({"BASE_DIR": str(tmp_path), "MAX_FILE_SIZE": "10"}, cwd=tmp_path)
    store = Store(tmp_path / "db2", dim=fake_embedder.dim)
    p = Pipeline(store, fake_embedder, cfg)
    f = tmp_path / "big.md"
    f.write_text("more than ten bytes of content")
    with pytest.raises(FileTooLargeError):
        p.ingest_file(f)


def test_empty_document_rejected(pipe):
    p, _, root = pipe
    f = root / "empty.md"
    f.write_text("   \n\n  ")
    with pytest.raises(EmptyDocumentError):
        p.ingest_file(f)


def test_ingest_data_text_and_markdown(pipe):
    p, store, _ = pipe
    res = p.ingest_data(
        "Plain text body long enough to keep.", source="note-1", fmt="text", title="Note"
    )
    assert res.title == "Note" and store.get_source("note-1").source_type == "data"
    res2 = p.ingest_data("# MD Title\n\nBody here.", source="note-2", fmt="markdown")
    assert res2.title == "MD Title"


def test_ingest_data_html(pipe):
    p, store, _ = pipe
    res = p.ingest_data(
        "<html><head><title>H</title></head><body><p>Hypertext body.</p></body></html>",
        source="page-1",
        fmt="html",
    )
    assert res.title == "H"
    assert "Hypertext body" in store.neighbors("page-1", 0, 0, 0)[0].text


def test_ingest_data_bad_format(pipe):
    p, _, _ = pipe
    with pytest.raises(UnsupportedFormatError):
        p.ingest_data("x", source="s", fmt="pdf")


def test_ingest_url_scheme_rejected(pipe):
    p, _, _ = pipe
    with pytest.raises(SecurityError):
        p.ingest_url("file:///etc/passwd")


def test_ingest_url_mocked(pipe, monkeypatch):
    import minirag_mcp.ingest.pipeline as mod
    from minirag_mcp.ingest.parser import ParsedDoc

    p, store, _ = pipe
    monkeypatch.setattr(
        mod,
        "parse_url",
        lambda url: ParsedDoc(markdown="# Remote\n\nFetched body.", title="Remote"),
    )
    res = p.ingest_url("https://example.com/docs")
    assert res.source == "https://example.com/docs"
    assert store.get_source(res.source).source_type == "url"
