"""Optional OCR engine (the [ocr] extra): scanned PDFs and images -> text.

The only module that touches rapidocr / pypdfium2 / rapid_orientation, and all
three are imported lazily inside functions: the core package must import and
run without the extra installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

ENGINE_RAPIDOCR = "rapidocr"

INSTALL_HINT = (
    "install the OCR extra: uv tool install 'minirag-mcp[ocr]' (or pip install 'minirag-mcp[ocr]')"
)


class OcrError(Exception):
    """OCR was needed and failed or is unavailable. Always loud, never a no-op."""


def available() -> bool:
    return all(importlib.util.find_spec(mod) is not None for mod in ("rapidocr", "pypdfium2"))


def pdf_page_texts(path: Path) -> list[str]:
    """Text-layer text per page, one pdfminer pass. "" for a page with none."""
    texts: list[str] = []
    for layout in extract_pages(str(path)):
        parts = [el.get_text() for el in layout if isinstance(el, LTTextContainer)]
        texts.append("".join(parts))
    return texts
