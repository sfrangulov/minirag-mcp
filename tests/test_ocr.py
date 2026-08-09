"""OCR module tests. Fast tests never import rapidocr; slow ones need the extra."""

from __future__ import annotations

from pathlib import Path

import pytest

from minirag_mcp import ocr
from pdf_builder import build_pdf


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
    assert ocr.assemble_lines(boxes, txts, Path("/docs/scan.png")) == "left right\nbelow"


def test_assemble_lines_empty():
    assert ocr.assemble_lines([], [], Path("/docs/scan.png")) == ""


def test_assemble_lines_mismatched_engine_output_raises_ocr_error():
    """One box short of its text is an engine-contract violation, and `strict=True`
    reports it as a bare ValueError with no file and no counts in it."""
    boxes = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
    with pytest.raises(ocr.OcrError) as e:
        ocr.assemble_lines(boxes, ["a", "b"], Path("/docs/scan.png"))
    assert "scan.png" in str(e.value)
    assert "1" in str(e.value) and "2" in str(e.value)


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


def test_orient_broken_numpy_install_raises_ocr_error(monkeypatch):
    monkeypatch.setattr(ocr.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ocr, "_orientation", [])
    _break_import(monkeypatch, "numpy")
    with pytest.raises(ocr.OcrError, match=r"failed to import.*minirag-mcp\[ocr\]"):
        ocr._orient(object(), Path("/docs/scan.png"))


def test_orient_broken_rapid_orientation_install_raises_ocr_error(monkeypatch):
    monkeypatch.setattr(ocr.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ocr, "_orientation", [])
    _break_import(monkeypatch, "rapid_orientation")
    with pytest.raises(ocr.OcrError, match=r"failed to import.*minirag-mcp\[ocr\]"):
        ocr._orient(object(), Path("/docs/scan.png"))


class _FakeCv2:
    """Just enough cv2 to run ocr_image's decode step without the extra installed.

    Pins only that every decoded frame reaches the recognizer, in order. That the
    decoder really returns every frame of a multi-page TIFF is the separate claim, and
    only the slow test below — real cv2, real file — can make it.
    """

    IMREAD_COLOR = 1

    def __init__(self, frames):
        self._frames = frames

    def imdecodemulti(self, buf, flags):
        return bool(self._frames), list(self._frames)


def test_ocr_image_recognizes_every_frame_of_a_multi_page_image(tmp_path, monkeypatch):
    """Multi-page TIFF is standard scanner and fax output. Reading frame 1 and calling
    the document done would index page 1 of a 20-page scan as the whole of it."""
    import sys

    from minirag_mcp.config import load_config

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2(["page-a", "page-b", "page-c"]))
    monkeypatch.setattr(ocr, "_recognize", lambda img, config, path: f"text of {img}")
    # Orientation is pinned by the slow rotation test; here it would only reject the
    # stand-in frames, and only on a machine that happens to have the extra installed.
    monkeypatch.setattr(ocr, "_orient", lambda img, path: img)
    img = tmp_path / "fax.tiff"
    img.write_bytes(b"pretend tiff bytes; the decoder is faked")
    text = ocr.ocr_image(img, load_config({}, cwd=tmp_path))
    assert text == "text of page-a\n\ntext of page-b\n\ntext of page-c"


def test_ocr_image_undecodable_bytes_still_raise(tmp_path, monkeypatch):
    import sys

    from minirag_mcp.config import load_config

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2([]))
    img = tmp_path / "broken.png"
    img.write_bytes(b"not an image")
    with pytest.raises(ocr.OcrError, match="could not decode"):
        ocr.ocr_image(img, load_config({}, cwd=tmp_path))


def test_ocr_image_unreadable_file_says_reading_failed(tmp_path, monkeypatch):
    """Reading the bytes and decoding them share one guard, so a missing file or a
    permission error was reported as an image that could not be decoded."""
    import sys

    from minirag_mcp.config import load_config

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2(["page-a"]))
    with pytest.raises(ocr.OcrError, match=r"could not read.*gone\.png"):
        ocr.ocr_image(tmp_path / "gone.png", load_config({}, cwd=tmp_path))


class _CrashingCv2:
    """A decoder that raises instead of returning `ok=False`. cv2 does both."""

    IMREAD_COLOR = 1

    def imdecodemulti(self, buf, flags):
        raise ValueError("cv2 error: unsupported depth")


def test_ocr_image_decoder_crash_raises_ocr_error(tmp_path, monkeypatch):
    import sys

    from minirag_mcp.config import load_config

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "cv2", _CrashingCv2())
    img = tmp_path / "corrupt.tiff"
    img.write_bytes(b"truncated scanner output")
    with pytest.raises(ocr.OcrError, match=r"corrupt\.tiff.*unsupported depth"):
        ocr.ocr_image(img, load_config({}, cwd=tmp_path))


class _FakeBitmap:
    def __init__(self, array):
        self._array = array

    def to_numpy(self):
        return self._array


class _FakePage:
    def __init__(self, error: Exception | None = None, bitmap=None):
        self._error = error
        self._bitmap = bitmap

    def render(self, scale):
        if self._error is not None:
            raise self._error
        if self._bitmap is None:
            raise AssertionError("this page was given neither an error nor a bitmap")
        return _FakeBitmap(self._bitmap)


class _FakeDoc:
    def __init__(self, pages):
        self._pages = list(pages)
        self.closed = False

    def __getitem__(self, index):
        return self._pages[index]

    def close(self):
        self.closed = True


class _FakePdfium:
    """Just enough pypdfium2 to drive ocr_pdf's open/lookup/render path without the extra."""

    def __init__(self, *, open_error: Exception | None = None, doc: _FakeDoc | None = None):
        self._open_error = open_error
        self._doc = doc

    def PdfDocument(self, path):
        if self._open_error is not None:
            raise self._open_error
        return self._doc


def _install_pdfium(monkeypatch, fake: _FakePdfium) -> None:
    import sys

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "pypdfium2", fake)


def _pdf(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.7 the renderer is faked; these bytes are never parsed")
    return p


def test_ocr_pdf_unopenable_file_raises_ocr_error(tmp_path, monkeypatch):
    """An encrypted or malformed PDF markitdown already handled: pypdfium2 raises its own
    exception type, which left the module unnamed, uncontextualised and not an OcrError."""
    from minirag_mcp.config import load_config

    _install_pdfium(monkeypatch, _FakePdfium(open_error=RuntimeError("password required")))
    p = _pdf(tmp_path, "encrypted.pdf")
    with pytest.raises(ocr.OcrError, match=r"encrypted\.pdf.*password required"):
        ocr.ocr_pdf(p, load_config({}, cwd=tmp_path), [0])


def test_ocr_pdf_page_the_renderer_does_not_have_raises_ocr_error(tmp_path, monkeypatch):
    """pdfminer counts the pages OCR routing asks for and pypdfium2 supplies them. When
    the two disagree on page count, `doc[index]` raised a bare IndexError."""
    from minirag_mcp.config import load_config

    doc = _FakeDoc([_FakePage()])
    _install_pdfium(monkeypatch, _FakePdfium(doc=doc))
    p = _pdf(tmp_path, "short.pdf")
    with pytest.raises(ocr.OcrError, match=r"short\.pdf.*list index out of range"):
        ocr.ocr_pdf(p, load_config({}, cwd=tmp_path), [3])
    assert doc.closed, "the document must still be closed when a page lookup fails"


def test_ocr_pdf_orientation_failure_raises_ocr_error(tmp_path, monkeypatch):
    """The same escape on the PDF path, which reaches `_orient` through the renderer."""
    from minirag_mcp.config import load_config

    cfg = load_config({}, cwd=tmp_path)
    doc = _FakeDoc([_FakePage(bitmap="rendered page")])
    _install_pdfium(monkeypatch, _FakePdfium(doc=doc))
    _install_orientation(monkeypatch, _crashing_orientation)
    p = _pdf(tmp_path, "tilted.pdf")
    with pytest.raises(ocr.OcrError, match=r"tilted\.pdf.*orientation model download failed"):
        ocr.ocr_pdf(p, cfg, [0])
    assert doc.closed


def test_ocr_pdf_unrenderable_page_raises_ocr_error(tmp_path, monkeypatch):
    from minirag_mcp.config import load_config

    doc = _FakeDoc([_FakePage(RuntimeError("bitmap allocation failed"))])
    _install_pdfium(monkeypatch, _FakePdfium(doc=doc))
    p = _pdf(tmp_path, "unrenderable.pdf")
    with pytest.raises(ocr.OcrError, match=r"unrenderable\.pdf.*bitmap allocation failed"):
        ocr.ocr_pdf(p, load_config({}, cwd=tmp_path), [0])
    assert doc.closed


class _CrashingOrientationModule:
    """rapid_orientation whose model init fails — an offline first use, a bad cache."""

    @staticmethod
    def RapidOrientation():
        raise RuntimeError("failed to download the orientation model")


def _install_orientation(monkeypatch, engine) -> None:
    """Make `_orient` believe the optional orientation package is installed.

    find_spec gates the whole step: without the extra it returns None and `_orient`
    hands the image straight back, so a fake engine would never be reached.
    """
    monkeypatch.setattr(ocr.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ocr, "_orientation", [engine])


def _crashing_orientation(img):
    raise RuntimeError("orientation model download failed")


def test_orient_engine_init_failure_raises_ocr_error_with_the_install_hint(monkeypatch):
    import sys

    monkeypatch.setattr(ocr.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(ocr, "_orientation", [])
    monkeypatch.setitem(sys.modules, "rapid_orientation", _CrashingOrientationModule())
    with pytest.raises(ocr.OcrError, match=r"orientation.*download.*minirag-mcp\[ocr\]"):
        ocr._orient(object(), Path("/docs/scan.png"))


def test_orient_unreadable_label_raises_ocr_error(monkeypatch):
    """The classifier's contract is an angle in a string; anything else is an engine
    fault, and `int()` reported it as a bare ValueError naming neither engine nor file."""
    _install_orientation(monkeypatch, lambda img: ("sideways", 0.01))
    with pytest.raises(ocr.OcrError, match=r"sideways.*scan\.png"):
        ocr._orient(object(), Path("/docs/scan.png"))


def test_ocr_image_orientation_failure_raises_ocr_error(tmp_path, monkeypatch):
    """A failing orientation engine left the module as a raw RuntimeError, which
    `parse_file` never converts: no file name, no install hint, no ParserError."""
    import sys

    from minirag_mcp.config import load_config

    cfg = load_config({}, cwd=tmp_path)
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2(["page-a"]))
    _install_orientation(monkeypatch, _crashing_orientation)
    img = tmp_path / "tilted.png"
    img.write_bytes(b"pretend png bytes; the decoder is faked")
    with pytest.raises(ocr.OcrError, match=r"tilted\.png.*orientation model download failed"):
        ocr.ocr_image(img, cfg)


def test_recognize_engine_failure_names_the_file(tmp_path, monkeypatch):
    from minirag_mcp.config import load_config

    def crash(img):
        raise RuntimeError("onnxruntime session failed")

    monkeypatch.setattr(ocr, "_engine", lambda config: crash)
    with pytest.raises(ocr.OcrError, match=r"scan\.png.*onnxruntime session failed"):
        ocr._recognize(object(), load_config({}, cwd=tmp_path), tmp_path / "scan.png")


@pytest.mark.slow
def test_real_engine_reads_both_pages_of_a_two_frame_tiff(tmp_path):
    """The claim the fake above cannot make: cv2 really hands back both frames of a
    two-page TIFF written the way a scanner writes one, and both reach the index."""
    pytest.importorskip("rapidocr")
    import cv2
    import numpy as np

    from minirag_mcp.config import load_config

    frames = []
    for marker in ("FIRST PAGE 111", "SECOND PAGE 222"):
        page = np.full((300, 1200, 3), 255, dtype=np.uint8)
        for y in (80, 160, 240):
            cv2.putText(page, marker, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
        frames.append(page)
    p = tmp_path / "scan.tiff"
    assert cv2.imwritemulti(str(p), frames), "failed to write a two-frame TIFF"
    text = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
    assert "111" in text and "222" in text


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
