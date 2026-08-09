"""OCR module tests. Fast tests never import rapidocr; slow ones need the extra."""

from __future__ import annotations

from pathlib import Path

import pytest
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


def test_assemble_lines_reading_order():
    # Three boxes: two on one visual line (given out of x-order), one below.
    # Box format mirrors RapidOCR: 4 points, [x, y] each.
    boxes = [
        [[300, 10], [400, 10], [400, 30], [300, 30]],  # line 1, right
        [[10, 12], [200, 12], [200, 32], [10, 32]],  # line 1, left
        [[10, 100], [200, 100], [200, 120], [10, 120]],  # line 2
    ]
    txts = ["right", "left", "below"]
    assert ocr.assemble_lines(boxes, txts) == "left right\nbelow"


def test_assemble_lines_empty():
    assert ocr.assemble_lines([], []) == ""


def test_ocr_image_without_extra_raises_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    img = tmp_path / "scan.png"
    img.write_bytes(b"not really a png")
    from minirag_mcp.config import load_config

    cfg = load_config({}, cwd=tmp_path)
    with pytest.raises(ocr.OcrError, match=r"minirag-mcp\[ocr\]"):
        ocr.ocr_image(img, cfg)


def _break_import(monkeypatch, name: str) -> None:
    """Force `import <name>` to raise ImportError, simulating a broken
    install (e.g. ABI mismatch) even where the module happens to be
    resolvable by find_spec — or even actually importable in this test env.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(mod_name, *args, **kwargs):
        if mod_name == name:
            raise ImportError(f"simulated broken install of {name!r}")
        return real_import(mod_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_engine_broken_rapidocr_install_raises_ocr_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    _break_import(monkeypatch, "rapidocr")
    from minirag_mcp.config import load_config

    cfg = load_config({}, cwd=tmp_path)
    with pytest.raises(ocr.OcrError, match=r"failed to import.*minirag-mcp\[ocr\]"):
        ocr._engine(cfg)


def test_ocr_pdf_broken_pypdfium2_install_raises_ocr_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    _break_import(monkeypatch, "pypdfium2")
    from minirag_mcp.config import load_config

    cfg = load_config({}, cwd=tmp_path)
    with pytest.raises(ocr.OcrError, match=r"failed to import.*minirag-mcp\[ocr\]"):
        ocr.ocr_pdf(tmp_path / "missing.pdf", cfg, [0])


def test_ocr_image_broken_cv2_install_raises_ocr_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    _break_import(monkeypatch, "cv2")
    from minirag_mcp.config import load_config

    cfg = load_config({}, cwd=tmp_path)
    with pytest.raises(ocr.OcrError, match=r"failed to import.*minirag-mcp\[ocr\]"):
        ocr.ocr_image(tmp_path / "missing.png", cfg)


@pytest.mark.slow
def test_real_engine_reads_generated_cyrillic_page(tmp_path):
    pytest.importorskip("rapidocr")
    import cv2
    import numpy as np

    from minirag_mcp.config import load_config

    img = np.full((200, 900, 3), 255, dtype=np.uint8)
    # Latin marker text: cv2's built-in Hershey fonts cannot draw Cyrillic,
    # and this test pins the pipeline, not the language model (the eslav
    # model reads Latin too).
    cv2.putText(img, "INVOICE 12345 TOTAL 999", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 4)
    p = tmp_path / "scan.png"
    cv2.imwrite(str(p), img)
    cfg = load_config({}, cwd=tmp_path)
    text = ocr.ocr_image(p, cfg)
    assert "12345" in text and "999" in text


@pytest.mark.slow
def test_real_engine_fixes_rotated_page(tmp_path):
    """Pins the np.rot90 sign in _orient: a 90-degree page must still read."""
    pytest.importorskip("rapidocr")
    pytest.importorskip("rapid_orientation")
    import cv2
    import numpy as np

    from minirag_mcp.config import load_config

    img = np.full((300, 1200, 3), 255, dtype=np.uint8)
    for y in (80, 160, 240):
        cv2.putText(img, "ROTATION CHECK 777", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    rotated = np.rot90(img, k=1)
    p = tmp_path / "rot.png"
    cv2.imwrite(str(p), rotated)
    cfg = load_config({}, cwd=tmp_path)
    assert "777" in ocr.ocr_image(p, cfg)


@pytest.mark.slow
def test_real_engine_reads_rendered_pdf_page(tmp_path):
    """Pins the pypdfium2 render -> _orient -> _recognize path in ocr_pdf.

    The ocr_image tests above never touch pypdfium2, so a channel-order or
    shape mismatch in PdfBitmap.to_numpy() would go uncaught without this.

    The marker is repeated to span most of the page width: a single short
    line on an otherwise blank 612x792pt page (rendered at 300 dpi, ~2550px
    wide) falls under RapidOCR's default detection confidence threshold —
    confirmed by direct engine probing, not a bug in this module's render
    path (the identical pixels crop correctly when isolated). A wider line
    is closer to a real scanned page's text density and detects reliably.
    """
    pytest.importorskip("rapidocr")
    from minirag_mcp.config import load_config

    p = _write(tmp_path, ["OCR PDF CHECK 555 " * 5])
    cfg = load_config({}, cwd=tmp_path)
    texts = ocr.ocr_pdf(p, cfg, [0])
    assert "555" in texts[0]
