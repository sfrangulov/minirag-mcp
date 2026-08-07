import pytest

from minirag_mcp.ingest.parser import (
    SUPPORTED_EXTENSIONS,
    ParsedDoc,
    ParserError,
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
