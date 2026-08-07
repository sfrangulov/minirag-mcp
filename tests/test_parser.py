import pytest

from minirag_mcp.ingest.parser import (
    SUPPORTED_EXTENSIONS,
    ParsedDoc,
    ParserError,
    _is_informative_stem,
    extract_title,
    parse_file,
    parse_html,
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
    assert not _is_informative_stem("read_me")  # separators ignored
    assert not _is_informative_stem("   ")


def test_stem_normalisation_keeps_meaningful_punctuation(tmp_path):
    f = tmp_path / "И-112_ЗПС__Хранение   ТМЗ.txt"
    f.write_text("body", encoding="utf-8")
    assert parse_file(f).title == "И-112 ЗПС Хранение ТМЗ"


def test_parse_html_and_url_titles_unaffected_by_stem_rule():
    # parse_html has no filename at all; its fallback stays "Untitled".
    assert parse_html("<html><body><p>no title tag</p></body></html>").title == "Untitled"
