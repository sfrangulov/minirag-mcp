"""OCR module tests. Fast tests never import rapidocr; slow ones need the extra."""

from __future__ import annotations

from pathlib import Path

from tests.pdf_builder import build_pdf

from minirag_mcp import ocr


def _write(tmp_path: Path, pages: list[str]) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(build_pdf(pages))
    return p


def test_available_is_bool():
    assert isinstance(ocr.available(), bool)


def test_pdf_page_texts_counts_text_and_blank_pages(tmp_path):
    p = _write(tmp_path, ["first page with plenty of text", "", "third page"])
    texts = ocr.pdf_page_texts(p)
    assert len(texts) == 3
    assert "first page" in texts[0]
    assert texts[1].strip() == ""
    assert "third" in texts[2]
