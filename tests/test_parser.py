import pytest
from tests.pdf_builder import build_pdf

from conftest import SECRET_BODY
from minirag_mcp.ingest.parser import (
    MAX_REDIRECTS,
    SUPPORTED_EXTENSIONS,
    ParsedDoc,
    ParserError,
    _is_boilerplate_heading,
    _is_informative_stem,
    _strip_extension_suffix,
    extract_title,
    parse_file,
    parse_html,
    parse_url,
    strip_data_uri_images,
)


def test_supported_extensions_frozen():
    assert ".md" in SUPPORTED_EXTENSIONS and ".ipynb" in SUPPORTED_EXTENSIONS
    assert ".py" not in SUPPORTED_EXTENSIONS


def test_extract_title_precedence():
    md = "# Real Title\n\nBody"
    assert extract_title(md, "Explicit", "fallback") == "Explicit"
    assert extract_title(md, None, "fallback") == "Real Title"
    assert extract_title("no heading here", None, "fallback") == "fallback"


def test_parse_markdown_file(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# My Doc\n\nHello **world**.\n", encoding="utf-8")
    doc = parse_file(f)
    assert isinstance(doc, ParsedDoc)
    assert doc.title == "My Doc"  # markitdown returns title=None for md; H1 fallback
    assert "Hello" in doc.markdown


def test_parse_txt_file_title_falls_back_to_stem(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("just plain text without headings", encoding="utf-8")
    doc = parse_file(f)
    assert doc.title == "notes"


def test_parse_html_string_gets_title_tag():
    doc = parse_html("<html><head><title>Page T</title></head><body><p>Body text</p></body></html>")
    assert doc.title == "Page T"
    assert "Body text" in doc.markdown


def test_parse_html_explicit_title_wins():
    doc = parse_html("<html><body><p>x</p></body></html>", title="Given")
    assert doc.title == "Given"


def test_parse_file_failure_wrapped():
    # Test that ParserError is raised for non-existent files
    from pathlib import Path

    with pytest.raises(ParserError):
        parse_file(Path("/nonexistent/file.pdf"))


def test_parse_file_invalid_pdf(tmp_path):
    # NOTE: markitdown does NOT raise on invalid PDFs; it treats them as plaintext
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"not a real pdf")
    doc = parse_file(f)
    # Verify it gracefully falls back to treating it as plain text
    assert "not a real pdf" in doc.markdown
    assert doc.title == "broken"


def test_h1_inside_code_fence_is_not_a_title():
    md = "```\n# This is inside a code fence\n```\n\nActual body"
    assert extract_title(md, None, "fallback") == "fallback"


def test_real_h1_after_fenced_fake_wins():
    md = "```\n# Fenced Fake Title\n```\n\n# Real Title\n\nBody"
    assert extract_title(md, None, "fallback") == "Real Title"


def test_h1_after_tilde_fence():
    md = "~~~\n# fake\n~~~\n\n# Real\n\nBody"
    assert extract_title(md, None, "fallback") == "Real"


# --- filename as the document title (files only) -------------------------------------


def test_informative_stem_beats_boilerplate_h1(tmp_path):
    """The real-corpus failure: the filename carries the document code and subject,
    while the first H1 is boilerplate shared by every document in the set."""
    f = tmp_path / "И-112_ЗПС_Хранение ТМЗ на складах.md"
    f.write_text("# 1. Общие положения\n\nТекст документа.\n", encoding="utf-8")
    assert parse_file(f).title == "И-112 ЗПС Хранение ТМЗ на складах"


def test_generic_stem_falls_back_to_h1(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# A Real Heading\n\nBody text.\n", encoding="utf-8")
    assert parse_file(f).title == "A Real Heading"


def test_numeric_stem_falls_back_to_h1(tmp_path):
    f = tmp_path / "12345.md"
    f.write_text("# A Real Heading\n\nBody text.\n", encoding="utf-8")
    assert parse_file(f).title == "A Real Heading"


def test_converter_metadata_title_beats_the_stem(tmp_path):
    """markitdown supplies a real title for HTML; it outranks the filename."""
    f = tmp_path / "some-saved-page.html"
    f.write_text("<html><head><title>Page T</title></head><body><p>b</p></body></html>", "utf-8")
    assert parse_file(f).title == "Page T"


def test_generic_stem_without_h1_still_falls_back_to_the_stem(tmp_path):
    f = tmp_path / "readme.txt"
    f.write_text("plain body without any heading at all", encoding="utf-8")
    assert parse_file(f).title == "readme"


def test_is_informative_stem():
    assert _is_informative_stem("И-112_ЗПС_Хранение ТМЗ")
    assert _is_informative_stem("notes")
    assert not _is_informative_stem("abc")  # shorter than 4 chars
    assert not _is_informative_stem("12345")  # purely numeric
    assert not _is_informative_stem("2026-08-07")  # numeric once separators drop out
    assert not _is_informative_stem("Untitled")
    assert not _is_informative_stem("DOCUMENT")  # case-insensitive
    assert not _is_informative_stem("   ")


# Names a camera, a scanner, an office suite or a file manager hands out — they say
# nothing about the document and must never be mistaken for a deliberate name.
MACHINE_ISSUED_STEMS = [
    "Untitled-1",
    "New Document",
    "Copy of Report",
    "scan_001",
    "IMG_20260807_123456",
    "Копия документа (2)",
    "Document1",
    "DSC00042",
]


@pytest.mark.parametrize("stem", MACHINE_ISSUED_STEMS)
def test_machine_issued_stems_are_not_informative(stem):
    assert not _is_informative_stem(stem)


@pytest.mark.parametrize("stem", MACHINE_ISSUED_STEMS)
def test_machine_issued_stem_never_displaces_a_real_h1(tmp_path, stem):
    f = tmp_path / f"{stem}.md"
    f.write_text("# Real Document Heading\n\nBody.\n", encoding="utf-8")
    assert parse_file(f).title == "Real Document Heading"


@pytest.mark.parametrize(
    "name", ["notes.md", "intro.md", "CHANGELOG.md", "0001-use-lancedb.md", "guide.txt"]
)
def test_a_real_h1_outranks_the_filename(tmp_path, name):
    """A heading the author wrote is a better title than the file's name, whatever the
    name looks like — the filename only steps in when the heading is boilerplate."""
    f = tmp_path / name
    f.write_text("# Real Document Heading\n\nBody.\n", encoding="utf-8")
    assert parse_file(f).title == "Real Document Heading"


@pytest.mark.parametrize(
    "heading",
    ["**1. Общие положения**", "1. Общие положения", "Лист изменений", "IV. Introduction"],
)
def test_boilerplate_h1_yields_to_the_filename(tmp_path, heading):
    """The measured corpus problem: 76 documents titled "1 Общие положения" and 69
    titled "Лист изменений" — the section name is shared, the filename is not."""
    f = tmp_path / "И-112_ЗПС_Хранение ТМЗ на складах.md"
    f.write_text(f"# {heading}\n\nТекст документа.\n", encoding="utf-8")
    assert parse_file(f).title == "И-112 ЗПС Хранение ТМЗ на складах"


def test_boilerplate_h1_with_a_generic_filename_still_wins_over_nothing(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Общие положения\n\nТекст.\n", encoding="utf-8")
    assert parse_file(f).title == "Общие положения"


def test_is_boilerplate_heading():
    assert _is_boilerplate_heading("Общие положения")
    assert _is_boilerplate_heading("1.2.3 Общие положения")
    assert _is_boilerplate_heading("**Лист изменений**")
    assert _is_boilerplate_heading("  2) TABLE OF CONTENTS  ")
    assert _is_boilerplate_heading("III. Revision history")
    # whole-heading match only: a real title that merely mentions the words stays
    assert not _is_boilerplate_heading("Общие положения о премировании работников")
    assert not _is_boilerplate_heading("Introduction to LanceDB indexing")
    assert not _is_boilerplate_heading("И-112 ЗПС Хранение ТМЗ")


def test_image_only_h1_is_not_a_title(tmp_path):
    """markitdown renders a heading holding only a picture as an image placeholder;
    it names nothing, so the filename must win (5 documents in the measured corpus)."""
    f = tmp_path / "D-493 Передача данных.md"
    f.write_text("# ![](data:image/x-emf;base64...)\n\nТекст.\n", encoding="utf-8")
    assert parse_file(f).title == "D-493 Передача данных"


def test_leftover_extension_is_not_part_of_the_title(tmp_path):
    """`report.pdf.pdf` has the stem `report.pdf`; the extension is not part of a title."""
    assert _strip_extension_suffix("report.pdf") == "report"
    assert _strip_extension_suffix("И-112.DOCX") == "И-112"
    assert _strip_extension_suffix("v1.2 plan") == "v1.2 plan"  # not an extension
    f = tmp_path / "report.txt.txt"
    f.write_text("plain body without a heading", encoding="utf-8")
    assert parse_file(f).title == "report"


def test_a_name_that_only_contains_a_generic_word_stays_informative():
    """The rule rejects names made entirely of generic tokens, not every name that
    happens to use one — otherwise real titles would be thrown away."""
    assert _is_informative_stem("Document Management Policy")
    assert _is_informative_stem("Scanner maintenance log")
    assert _is_informative_stem("И-112_ЗПС_Хранение ТМЗ")


def test_stem_normalisation_keeps_meaningful_punctuation(tmp_path):
    f = tmp_path / "И-112_ЗПС__Хранение   ТМЗ.txt"
    f.write_text("body", encoding="utf-8")
    assert parse_file(f).title == "И-112 ЗПС Хранение ТМЗ"


def test_parse_html_and_url_titles_unaffected_by_stem_rule():
    # parse_html has no filename at all; its fallback stays "Untitled".
    assert parse_html("<html><body><p>no title tag</p></body></html>").title == "Untitled"


PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


def test_data_uri_image_keeps_its_alt_text():
    """Alt text is content the author wrote; the base64 blob around it is not."""
    md = f"Intro paragraph.\n\n![Схема обмена]({PNG})\n\nOutro."
    assert strip_data_uri_images(md) == "Intro paragraph.\n\nСхема обмена\n\nOutro."


def test_data_uri_image_without_alt_text_goes_entirely():
    md = f"Intro paragraph.\n\n![]({PNG})\n\nOutro."
    assert strip_data_uri_images(md) == "Intro paragraph.\n\n\n\nOutro."


def test_ordinary_image_links_are_left_alone():
    """A link to a file or an http URL is a reference, not an inlined picture."""
    md = "![diagram](images/diagram.png)\n\n![logo](https://example.com/logo.svg)"
    assert strip_data_uri_images(md) == md


def test_data_uri_inside_a_code_fence_is_content():
    """Inside a fence the URI is the subject of the document, not decoration."""
    md = f"Example:\n\n```markdown\n![]({PNG})\n```\n\nDone."
    assert strip_data_uri_images(md) == md


def test_a_document_that_is_only_a_placeholder_keeps_it():
    """Removing decoration must not remove the document."""
    md = f"![]({PNG})"
    assert strip_data_uri_images(md) == md
    assert strip_data_uri_images("") == ""


# --- the host rule survives a redirect (SSRF) ----------------------------------------


def test_redirect_to_cloud_metadata_is_refused(redirect_server):
    """The bypass this guard exists for: an allowed public host answering
    `302 -> http://<metadata>/`. Checking only the URL the caller supplied leaves the
    hop unexamined, and the metadata response reaches the converter."""
    with pytest.raises(ParserError) as exc:
        parse_url(redirect_server.url("/redirect-to-metadata"))
    msg = str(exc.value)
    assert "metadata.test" in msg  # names the blocked host
    assert "169.254.169.254" in msg and "link-local" in msg  # and why
    assert "redirected" in msg  # and that a redirect, not the given URL, was refused
    assert "ALLOW_PRIVATE_URLS" in msg  # and the way out
    # The strongest evidence: the request was never made. The stand-in metadata
    # endpoint is this same server, so a followed redirect would show up here.
    assert redirect_server.hits == [f"public.test:{redirect_server.port}/redirect-to-metadata"]


def test_the_refused_redirect_target_really_would_have_been_served(redirect_server):
    """Turning the host rule off fetches exactly what the refusal withheld — so the
    refusal is doing work, and the guard reads the setting in force at call time
    rather than whatever the cached converter was first built with."""
    url = redirect_server.url("/redirect-to-metadata")
    with pytest.raises(ParserError):
        parse_url(url)
    assert SECRET_BODY in parse_url(url, allow_private=True).markdown


def test_a_blocked_url_asked_for_directly_is_not_reported_as_a_redirect(redirect_server):
    """The same guard covers the first request, and the refusal says what happened —
    the caller asked for this host, nothing redirected them to it."""
    with pytest.raises(ParserError) as exc:
        parse_url(redirect_server.url("/latest/meta-data/", host="metadata.test"))
    msg = str(exc.value)
    assert "metadata.test" in msg and "link-local" in msg
    assert "redirected" not in msg
    assert redirect_server.hits == []  # refused before the socket was opened


def test_a_bare_authority_blocked_url_is_not_reported_as_a_redirect(redirect_server):
    """requests normalizes a path-less URL by appending "/" before the guard ever sees
    it (`http://host` -> `http://host/`), so a raw string compare between the checked
    URL and the one the caller gave would call that normalization a redirect. Nothing
    redirected here — the host itself is blocked on the very first request."""
    with pytest.raises(ParserError) as exc:
        parse_url(redirect_server.url("", host="metadata.test"))
    msg = str(exc.value)
    assert "metadata.test" in msg and "link-local" in msg
    assert "redirected" not in msg
    assert redirect_server.hits == []  # refused before the socket was opened


def test_a_redirect_chain_across_public_hosts_is_followed(redirect_server):
    """The guard refuses blocked hops, not redirects."""
    doc = parse_url(redirect_server.url("/chain/3"))
    assert "PUBLIC-BODY" in doc.markdown
    assert doc.title == "Public Page"


def test_redirects_are_capped(redirect_server):
    """requests would follow 30. Each hop is another chance to reach somewhere the
    guard has to refuse, and no document needs that many."""
    assert MAX_REDIRECTS == 5
    with pytest.raises(ParserError, match=f"Exceeded {MAX_REDIRECTS} redirects"):
        parse_url(redirect_server.url(f"/chain/{MAX_REDIRECTS + 4}"))


def test_every_entry_point_strips_placeholders(tmp_path):
    f = tmp_path / "И-112 Хранение ТМЗ.md"
    f.write_text(f"# И-112 Хранение ТМЗ\n\n![Схема]({PNG})\n\nТело документа.\n", encoding="utf-8")
    from_file = parse_file(f).markdown
    assert "base64" not in from_file and "Схема" in from_file
    from_html = parse_html(f'<h1>T</h1><p>body</p><img alt="Схема" src="{PNG}">').markdown
    assert "base64" not in from_html and from_html.endswith("Схема")


# --- images become documents when the [ocr] extra is present -------------------------


def test_supported_extensions_without_ocr(monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: False)
    exts = parser.supported_extensions()
    assert ".pdf" in exts and ".png" not in exts


def test_supported_extensions_with_ocr(monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    exts = parser.supported_extensions()
    assert ".png" in exts and ".webp" in exts


def test_parse_file_image_routes_through_ocr(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "recognized scan text")
    img = tmp_path / "Договор_поставки_2026.png"
    img.write_bytes(b"png bytes irrelevant, ocr is faked")
    doc = parser.parse_file(img, load_config({}, cwd=tmp_path))
    assert doc.markdown == "recognized scan text"
    assert doc.ocr_engine == "rapidocr"
    assert doc.title == "Договор поставки 2026"


def test_parse_file_image_without_ocr_raises_hint(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: False)
    img = tmp_path / "scan.png"
    img.write_bytes(b"...")
    with pytest.raises(parser.ParserError, match=r"minirag-mcp\[ocr\]"):
        parser.parse_file(img, load_config({}, cwd=tmp_path))


def test_parse_file_image_title_strips_leftover_extension(tmp_path, monkeypatch):
    """A non-informative image stem still goes through the same extension-stripping
    as every other file's fallback title — "scan.png.png" must not surface "scan.png"
    as the title, the exact leftover-extension case _EXTENSION_SUFFIX_RE exists for."""
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "text")
    img = tmp_path / "scan.png.png"
    img.write_bytes(b"...")
    doc = parser.parse_file(img, load_config({}, cwd=tmp_path))
    assert doc.title == "scan"


def test_parse_file_image_title_normalizes_underscores(tmp_path, monkeypatch):
    """A non-informative image stem's underscores must become spaces, matching every
    other file's fallback title (`_file_title`'s `_display_stem(stem) or stem`)."""
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "text")
    img = tmp_path / "IMG_20260807_123456.png"
    img.write_bytes(b"...")
    doc = parser.parse_file(img, load_config({}, cwd=tmp_path))
    assert doc.title == "IMG 20260807 123456"


# --- scanned PDFs fall back to OCR per page -------------------------------------------

LONG = "long enough text line to clear the default per-page threshold easily"


def _pdf(tmp_path, pages):
    p = tmp_path / "doc.pdf"
    p.write_bytes(build_pdf(pages))
    return p


def test_pdf_all_text_pages_skip_ocr(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_pdf", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not OCR"))
    )
    doc = parser.parse_file(_pdf(tmp_path, [LONG, LONG]), load_config({}, cwd=tmp_path))
    assert doc.ocr_engine == ""
    assert "long enough text" in doc.markdown


def test_pdf_scanned_pages_get_ocr(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_pdf", lambda path, config, pages: {i: f"OCR PAGE {i}" for i in pages}
    )
    doc = parser.parse_file(_pdf(tmp_path, [LONG, "", ""]), load_config({}, cwd=tmp_path))
    assert doc.ocr_engine == "rapidocr"
    # page order preserved: text page first, then the two OCR'd pages
    body = doc.markdown
    assert body.index("long enough") < body.index("OCR PAGE 1") < body.index("OCR PAGE 2")


def test_pdf_fully_scanned_without_ocr_raises_hint(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: False)
    with pytest.raises(parser.ParserError, match=r"scanned.*minirag-mcp\[ocr\]"):
        parser.parse_file(_pdf(tmp_path, ["", ""]), load_config({}, cwd=tmp_path))


def test_pdf_mixed_without_ocr_keeps_partial_text(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: False)
    doc = parser.parse_file(_pdf(tmp_path, [LONG, ""]), load_config({}, cwd=tmp_path))
    assert doc.ocr_engine == ""
    assert "long enough" in doc.markdown


def test_pdf_without_config_behaves_as_before(tmp_path):
    from minirag_mcp.ingest import parser

    doc = parser.parse_file(_pdf(tmp_path, ["", ""]))
    assert doc.markdown.strip() == ""  # legacy path: empty conversion, no error
