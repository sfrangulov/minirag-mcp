import pytest

from minirag_mcp.config import load_config
from minirag_mcp.ingest.parser import ParserError
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


def test_ingest_url_refuses_cloud_metadata(pipe):
    """The URL is usually picked by an LLM reading an indexed document — an
    unrestricted host turns that into an SSRF path."""
    p, _, _ = pipe
    with pytest.raises(SecurityError, match="ALLOW_PRIVATE_URLS"):
        p.ingest_url("http://169.254.169.254/latest/meta-data/")


def test_a_redirect_to_cloud_metadata_indexes_nothing(pipe, redirect_server):
    """End to end: the URL the caller gives passes the check, the response redirects
    to instance metadata, and the index stays empty. Checking only the supplied URL
    would have put the metadata response in it."""
    p, store, _ = pipe
    url = redirect_server.url("/redirect-to-metadata")
    with pytest.raises(ParserError, match="169.254.169.254"):
        p.ingest_url(url)
    assert store.chunk_count() == 0
    assert store.get_source(url) is None


def test_allow_private_urls_config_reaches_ingest_url(tmp_path, fake_embedder, monkeypatch):
    """ALLOW_PRIVATE_URLS=1 must reach parse_url as allow_private=True — otherwise a
    user who sets the flag still gets refused, while being told to set the flag they
    already set."""
    import minirag_mcp.ingest.pipeline as mod
    from minirag_mcp.ingest.parser import ParsedDoc

    calls: list[dict] = []

    def fake_parse_url(u, **kw):
        calls.append(kw)
        return ParsedDoc("# W\n\nWiki body.", "W")

    monkeypatch.setattr(mod, "parse_url", fake_parse_url)
    cfg = load_config({"BASE_DIR": str(tmp_path), "ALLOW_PRIVATE_URLS": "1"}, cwd=tmp_path)
    store = Store(tmp_path / "db-private", dim=fake_embedder.dim)
    res = Pipeline(store, fake_embedder, cfg).ingest_url("http://192.168.1.10/wiki")
    assert res.source == "http://192.168.1.10/wiki"
    assert calls == [{"allow_private": True}]


def test_allow_private_urls_off_reaches_ingest_url_as_false(pipe, monkeypatch, public_dns):
    """The other direction of the same plumbing: with the flag off (the default),
    parse_url must receive allow_private=False explicitly, not merely whatever a stub
    happened to default to."""
    import minirag_mcp.ingest.pipeline as mod
    from minirag_mcp.ingest.parser import ParsedDoc

    p, _, _ = pipe
    calls: list[dict] = []

    def fake_parse_url(u, **kw):
        calls.append(kw)
        return ParsedDoc("# Remote\n\nFetched body.", "Remote")

    monkeypatch.setattr(mod, "parse_url", fake_parse_url)
    res = p.ingest_url("https://example.com/docs")
    assert res.source == "https://example.com/docs"
    assert calls == [{"allow_private": False}]


def test_title_line_seeded_into_first_chunk_only(pipe):
    """The title must reach the embedding of chunk 0, so semantic search sees the
    filename too — but it must not be repeated in every chunk."""
    p, store, root = pipe
    f = root / "И-112_ЗПС_Хранение ТМЗ на складах.md"
    f.write_text("# 1. Общие положения\n\n" + "Абзац про хранение. " * 200, encoding="utf-8")
    res = p.ingest_file(f)
    assert res.title == "И-112 ЗПС Хранение ТМЗ на складах"
    chunks = store.all_chunks(str(f))
    assert len(chunks) > 1, "need several chunks to prove the prefix is chunk-0 only"
    assert chunks[0].text.startswith(f"# {res.title}\n\n")
    assert all(not c.text.startswith(f"# {res.title}") for c in chunks[1:])


def test_title_line_is_not_doubled_on_reingest(pipe):
    p, store, root = pipe
    f = root / "И-112_ЗПС_Хранение ТМЗ на складах.md"
    f.write_text("# 1. Общие положения\n\n" + "Абзац про хранение. " * 200, encoding="utf-8")
    res = p.ingest_file(f)
    first = store.all_chunks(str(f))[0].text
    assert first.startswith(f"# {res.title}\n\n")
    p.ingest_file(f)
    again = store.all_chunks(str(f))[0].text
    assert again == first
    assert again.count(f"# {res.title}") == 1


def test_title_line_not_added_when_the_chunk_already_starts_with_it(pipe):
    p, store, _ = pipe
    res = p.ingest_data("# MD Title\n\nBody here, long enough to keep.", source="n", fmt="markdown")
    assert res.title == "MD Title"
    text = store.all_chunks("n")[0].text
    assert text.startswith("# MD Title\n\n") and text.count("# MD Title") == 1


def test_title_seeding_does_not_change_chunk_ids_or_count(pipe):
    """The prefix is applied after chunking, so boundaries and ids must be untouched."""
    p, store, root = pipe
    body = "# 1. Общие положения\n\n" + "Абзац про хранение. " * 200
    named = root / "И-112_ЗПС_Хранение ТМЗ на складах.md"
    named.write_text(body, encoding="utf-8")
    generic = root / "doc.md"
    generic.write_text(body, encoding="utf-8")
    assert p.ingest_file(named).chunk_count == p.ingest_file(generic).chunk_count
    assert [c.chunk_index for c in store.all_chunks(str(named))] == [
        c.chunk_index for c in store.all_chunks(str(generic))
    ]


def test_no_title_seeded_when_a_data_source_has_no_title(pipe):
    """The source id identifies the document, it does not describe it: injecting it into
    the embedded text is non-semantic noise in the very vector seeding exists to help."""
    p, store, _ = pipe
    body = "plain body about storage, long enough to keep."
    res = p.ingest_data(body, source="notes-2026")
    assert res.title == "notes-2026"
    assert store.all_chunks("notes-2026")[0].text == body


def test_explicit_data_title_is_seeded(pipe):
    p, store, _ = pipe
    p.ingest_data("plain body, long enough to keep.", source="notes-2026", title="Storage Policy")
    assert store.all_chunks("notes-2026")[0].text.startswith("# Storage Policy\n\n")


def test_html_data_without_a_title_is_not_seeded_with_the_fallback(pipe):
    p, store, _ = pipe
    res = p.ingest_data(
        "<html><body><p>Body text long enough to keep around.</p></body></html>",
        source="page-1",
        fmt="html",
    )
    # the source id stands in, as it does for the other data formats — but only as
    # metadata; it must not reach the embedded text
    assert res.title == "page-1"
    text = store.all_chunks("page-1")[0].text
    assert "# Untitled" not in text and "# page-1" not in text


def test_url_title_is_seeded_only_when_the_page_has_one(pipe, monkeypatch, public_dns):
    """A raw URL is an address, not a title — "# https://example.com/a/b?x=1&y=2" in the
    embedded text buys nothing and costs relevance."""
    import minirag_mcp.ingest.pipeline as mod
    from minirag_mcp.ingest.parser import ParsedDoc

    p, store, _ = pipe
    url = "https://example.com/a/b?x=1&y=2"
    # what parse_url returns for a titleless page: the URL as a stand-in title
    monkeypatch.setattr(
        mod,
        "parse_url",
        lambda u, **kw: ParsedDoc(markdown="Fetched body text.", title=u, has_title=False),
    )
    p.ingest_url(url)
    assert store.all_chunks(url)[0].text == "Fetched body text."

    monkeypatch.setattr(
        mod,
        "parse_url",
        lambda u, **kw: ParsedDoc("Fetched body text.", title="Remote Page", has_title=True),
    )
    p.ingest_url(url)
    assert store.all_chunks(url)[0].text.startswith("# Remote Page\n\n")


def test_explicit_url_title_is_seeded_even_for_a_titleless_page(pipe, monkeypatch, public_dns):
    import minirag_mcp.ingest.pipeline as mod
    from minirag_mcp.ingest.parser import ParsedDoc

    p, store, _ = pipe
    url = "https://example.com/a/b"
    monkeypatch.setattr(
        mod,
        "parse_url",
        lambda u, **kw: ParsedDoc(markdown="Fetched body text.", title=u, has_title=False),
    )
    p.ingest_url(url, title="Storage Policy")
    assert store.all_chunks(url)[0].text.startswith("# Storage Policy\n\n")


def test_ingest_url_mocked(pipe, monkeypatch, public_dns):
    import minirag_mcp.ingest.pipeline as mod
    from minirag_mcp.ingest.parser import ParsedDoc

    p, store, _ = pipe
    monkeypatch.setattr(
        mod,
        "parse_url",
        lambda url, **kw: ParsedDoc(markdown="# Remote\n\nFetched body.", title="Remote"),
    )
    res = p.ingest_url("https://example.com/docs")
    assert res.source == "https://example.com/docs"
    assert store.get_source(res.source).source_type == "url"


def test_image_placeholders_are_stripped_but_never_empty_a_document(pipe):
    """The placeholders leave the index; a document made only of them still ingests."""
    p, store, root = pipe
    doc = root / "И-112 Хранение ТМЗ.md"
    png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    doc.write_text(f"# И-112\n\n![Схема обмена]({png})\n\nТело документа.\n", encoding="utf-8")
    p.ingest_file(doc)
    text = "\n\n".join(c.text for c in store.all_chunks(str(doc)))
    assert "base64" not in text
    assert "Схема обмена" in text  # alt text is content and stays

    only_image = root / "Скан договора.md"
    only_image.write_text(f"![]({png})\n", encoding="utf-8")
    assert p.ingest_file(only_image).chunk_count == 1


class RecordingEmbedder:
    """The shared fake, wrapped to record every batch handed to embed_documents."""

    def __init__(self, inner):
        self.inner = inner
        self.dim = inner.dim
        self.model_name = inner.model_name
        self.batches: list[list[str]] = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)

    @property
    def embedded_texts(self) -> int:
        return sum(len(b) for b in self.batches)


def _recording_pipeline(tmp_path, fake_embedder, name):
    emb = RecordingEmbedder(fake_embedder)
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    return Pipeline(Store(tmp_path / name, dim=emb.dim), emb, cfg), emb


def _capture_records(monkeypatch, pipeline):
    stored: dict[str, list] = {}
    real = pipeline.store.replace_source

    def spy(source, records):
        stored[source] = list(records)
        return real(source, records)

    monkeypatch.setattr(pipeline.store, "replace_source", spy)
    return stored


def test_unmerged_chunks_are_not_embedded_twice(tmp_path, fake_embedder, monkeypatch):
    """merge_blocks embeds every block to decide the merges; only what merging changed
    needs a second embedding. Chunk 0 counts as changed once the title is seeded."""
    p, emb = _recording_pipeline(tmp_path, fake_embedder, "db-reuse")
    stored = _capture_records(monkeypatch, p)
    a, b = "Абзац А. " * 100, "Абзац Б. " * 100  # 900 chars each: never merge with a peer
    code = "```\nx = 1\n```"  # a code block attaches to its predecessor unconditionally
    doc = tmp_path / "И-112 Хранение ТМЗ.md"
    doc.write_text(f"{a}\n\n{code}\n\n{b}", encoding="utf-8")
    res = p.ingest_file(doc)

    assert res.chunk_count == 2
    records = stored[str(doc)]
    blocks, rechunked = emb.batches
    assert len(blocks) == 3, "every block is embedded once, to decide the merges"
    # chunk 1 is block B untouched, so its block vector stands and it is not re-embedded
    assert records[1].text == b.strip()
    assert b.strip() not in rechunked
    assert emb.embedded_texts == 4 < 3 + res.chunk_count
    # every stored vector describes the text stored beside it — including chunk 0, whose
    # cached vector must have been dropped when merging and title seeding rewrote it
    for r in records:
        assert r.vector == fake_embedder.embed_documents([r.text])[0]
    assert records[0].text.startswith("# И-112 Хранение ТМЗ\n\n")


def test_reused_vectors_match_a_direct_embedding_of_the_stored_text(
    tmp_path, fake_embedder, monkeypatch
):
    """The whole index rests on this: a vector must describe the text it is stored with,
    whatever path it took to get there."""
    p, emb = _recording_pipeline(tmp_path, fake_embedder, "db-vectors")
    stored = _capture_records(monkeypatch, p)
    doc = tmp_path / "И-112 Хранение ТМЗ на складах.md"
    doc.write_text(
        "# И-112 Хранение ТМЗ на складах\n\n"
        + "\n\n".join(
            f"## Раздел {i}\n\nАбзац {i} про хранение товарно-материальных запасов, "
            f"достаточно длинный чтобы стать отдельным блоком."
            for i in range(1, 21)
        ),
        encoding="utf-8",
    )
    res = p.ingest_file(doc)
    records = stored[str(doc)]
    assert len(records) == res.chunk_count > 1
    for r in records:
        assert r.vector == fake_embedder.embed_documents([r.text])[0]
    blocks = len(emb.batches[0])
    assert emb.embedded_texts < blocks + res.chunk_count  # the old cost, embedding twice


def test_title_seeding_invalidates_the_cached_vector_of_chunk_zero(
    tmp_path, fake_embedder, monkeypatch
):
    """Chunk 0 can be an unmerged block, and seeding rewrites its text after merging
    handed back a vector for the text as it was. Keeping that vector would mis-embed the
    one chunk the title exists to reach."""
    p, emb = _recording_pipeline(tmp_path, fake_embedder, "db-seeded")
    stored = _capture_records(monkeypatch, p)
    doc = tmp_path / "И-112 Хранение ТМЗ.md"
    doc.write_text("Абзац А. " * 100 + "\n\n" + "Абзац Б. " * 100, encoding="utf-8")
    res = p.ingest_file(doc)

    assert res.chunk_count == 2, "two blocks too long to merge, so neither is altered"
    first, second = stored[str(doc)]
    assert first.text.startswith(f"# {res.title}\n\n")
    assert first.vector == fake_embedder.embed_documents([first.text])[0]
    assert first.vector != fake_embedder.embed_documents([emb.batches[0][0]])[0]
    assert second.vector == fake_embedder.embed_documents([second.text])[0]
    assert emb.embedded_texts == 3  # 2 blocks + the reseeded chunk 0, not 4
