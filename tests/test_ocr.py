"""OCR module tests. Fast tests run on a fake tesserocr; slow ones need the extra."""

from __future__ import annotations

import logging
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

from minirag_mcp import ocr, ocr_vision
from minirag_mcp.config import load_config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.store import Store
from pdf_builder import build_pdf


def _write(tmp_path: Path, pages: list[str]) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(build_pdf(pages))
    return p


def test_available_is_bool():
    assert isinstance(ocr.available(), bool)


def test_available_needs_tesserocr(monkeypatch):
    monkeypatch.setattr(
        ocr.importlib.util, "find_spec", lambda name: None if name == "tesserocr" else object()
    )
    assert ocr.available() is False


def test_pdf_page_texts_counts_text_and_blank_pages(tmp_path):
    p = _write(tmp_path, ["first page with plenty of text", "", "third page"])
    texts = ocr.pdf_page_texts(p)
    assert len(texts) == 3
    assert "first page" in texts[0]
    assert texts[1].strip() == ""
    assert "third" in texts[2]


def test_ocr_image_without_extra_raises_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    img = tmp_path / "scan.png"
    img.write_bytes(b"not really a png")
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


def test_engine_broken_tesserocr_install_raises_ocr_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "_engines", {})
    _break_import(monkeypatch, "tesserocr")
    cfg = load_config({}, cwd=tmp_path)
    with pytest.raises(ocr.OcrError, match=r"failed to import.*minirag-mcp\[ocr\]"):
        ocr._engine(cfg)


def test_ocr_pdf_broken_pypdfium2_install_raises_ocr_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    _break_import(monkeypatch, "pypdfium2")
    cfg = load_config({}, cwd=tmp_path)
    with pytest.raises(ocr.OcrError, match=r"failed to import.*minirag-mcp\[ocr\]"):
        ocr.ocr_pdf(tmp_path / "missing.pdf", cfg, [0])


def test_ocr_image_broken_pil_install_raises_ocr_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    _break_import(monkeypatch, "PIL")
    cfg = load_config({}, cwd=tmp_path)
    with pytest.raises(ocr.OcrError, match=r"failed to import.*minirag-mcp\[ocr\]"):
        ocr.ocr_image(tmp_path / "missing.png", cfg)


RIL = SimpleNamespace(BLOCK=0, PARA=1, TEXTLINE=2, WORD=3, SYMBOL=4)  # tesserocr's values


@dataclass
class _Line:
    box: tuple[int, int, int, int]
    text: str | None  # None: a line the iterator has no text for
    block: bool = False
    para: bool = False
    orientation: int = 0


@dataclass
class _Reading:
    """What the fake engine makes of one image, in the shapes tesserocr hands back."""

    text: str  # GetUTF8Text()
    confidence: int  # MeanTextConf(): an int, 0 with no words
    lines: list[_Line] = field(default_factory=list)


def _reading(text: str, confidence: int | None = None, *, orientation: int = 0) -> _Reading:
    """The page that reads as `text`: one block, a paragraph after each blank line, lines
    stacked 50 px apart. 91 confident, or 0 with no words, as Tesseract reports."""
    lines: list[_Line] = []
    para = True
    for row in text.splitlines():
        if not row.strip():
            para = True
            continue
        top = 10 + 50 * len(lines)
        box = (10, top, 10 + 20 * len(row), top + 30)
        lines.append(_Line(box, row, block=not lines, para=para, orientation=orientation))
        para = False
    if confidence is None:
        confidence = 91 if lines else 0
    return _Reading(text, confidence, lines)


class _FakeIterator:
    """A ResultIterator walked line by line. An empty page's iterator still exists: it
    has no box, no orientation worth reading, and raises for its text."""

    def __init__(self, lines: list[_Line]):
        self._lines = lines
        self._at = 0

    def _line(self, level: int) -> _Line | None:
        assert level == RIL.TEXTLINE, "the fake models iteration by line only"
        return self._lines[self._at] if self._lines else None

    def BoundingBox(self, level):
        line = self._line(level)
        return None if line is None else line.box

    def GetUTF8Text(self, level):
        line = self._line(level)
        if line is None or line.text is None:
            raise RuntimeError("No text returned")
        return line.text + "\n"

    def IsAtBeginningOf(self, level):
        if not self._lines:
            return False
        line = self._lines[self._at]
        return {RIL.BLOCK: line.block, RIL.PARA: line.para or line.block, RIL.TEXTLINE: True}[level]

    def Orientation(self):
        if not self._lines:
            return (0, 0, 2, 434553094144.0)  # what the real one returns on an empty page
        return (self._lines[self._at].orientation, 0, 2, 0.0)

    def Next(self, level):
        self._line(level)
        if self._at + 1 >= len(self._lines):
            return False
        self._at += 1
        return True


class _FakeTessApi:
    """One PyTessBaseAPI handle: it holds the last image it was given, as the real one
    does, so two callers interleaving on one handle would read each other's page."""

    def __init__(self, tess: _FakeTesserocr):
        self._tess = tess
        self._image = None
        self._read: _Reading | None = None
        self.calls: list[str] = []
        self.resolutions: list[int] = []
        self.images: list[Image.Image] = []

    def SetImage(self, image):
        self.calls.append("SetImage")
        self.images.append(image)
        # One flag for every handle, so a test cannot pass by two callers each
        # getting a handle of their own.
        if not self._tess.busy.acquire(blocking=False):
            raise AssertionError("SetImage while another recognition is in progress")
        self._read = None  # a new image clears the last one's results
        time.sleep(self._tess.hold)
        self._image = image

    def SetSourceResolution(self, dpi):
        self.calls.append("SetSourceResolution")
        self.resolutions.append(dpi)

    def GetUTF8Text(self):
        self.calls.append("GetUTF8Text")
        try:
            reply = self._tess.reply(self._image)
            self._read = reply if isinstance(reply, _Reading) else _reading(reply)
            return self._read.text
        finally:
            self._tess.busy.release()

    def _results(self) -> _Reading:
        if self._read is None:
            raise AssertionError("results read without a recognition of the current image")
        return self._read

    def MeanTextConf(self):
        self.calls.append("MeanTextConf")
        time.sleep(self._tess.hold)  # room for another caller's SetImage to clear it
        return self._results().confidence

    def GetIterator(self):
        self.calls.append("GetIterator")
        return _FakeIterator(self._results().lines)


class _FakeTesserocr:
    """Just enough tesserocr to drive ocr.py without the extra installed."""

    PSM = SimpleNamespace(AUTO_OSD=1)
    RIL = RIL

    def __init__(self):
        self.created: list[dict] = []
        self.apis: list[_FakeTessApi] = []
        self.texts: list[str | _Reading] = []
        self.reply = lambda image: self.texts.pop(0)
        self.init_error: Exception | None = None
        self.init_delay = 0.0
        self.hold = 0.0
        self.busy = threading.Lock()

    def PyTessBaseAPI(self, **kw):
        self.created.append(kw)
        time.sleep(self.init_delay)
        if self.init_error is not None:
            raise self.init_error
        api = _FakeTessApi(self)
        self.apis.append(api)
        return api

    @property
    def api(self) -> _FakeTessApi:
        assert len(self.apis) == 1, f"expected one engine, {len(self.apis)} were built"
        return self.apis[0]


@pytest.fixture
def fake_tess(monkeypatch, tmp_path):
    fake = _FakeTesserocr()
    monkeypatch.setitem(sys.modules, "tesserocr", fake)
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ensure_tessdata", lambda cache_dir, langs: tmp_path / "td")
    monkeypatch.setattr(ocr, "_engines", {})
    return fake


@pytest.fixture
def cfg(tmp_path):
    return load_config({}, cwd=tmp_path)


def test_the_engine_reads_with_auto_osd_and_the_configured_languages(fake_tess, tmp_path):
    ocr._engine(load_config({"RAG_OCR_LANG": "eslav"}, cwd=tmp_path))
    assert fake_tess.created == [{"path": str(tmp_path / "td"), "lang": "rus+eng", "psm": 1}]


def test_one_engine_per_language_set(fake_tess, tmp_path):
    both = load_config({}, cwd=tmp_path)
    assert ocr._engine(both) is ocr._engine(both)
    ocr._engine(load_config({"RAG_OCR_LANG": "eng"}, cwd=tmp_path))
    assert [kw["lang"] for kw in fake_tess.created] == ["rus+eng", "eng"]


def test_two_first_uses_at_once_build_one_engine(fake_tess, cfg):
    fake_tess.init_delay = 0.05
    start = threading.Barrier(2)
    engines = []

    def build():
        start.wait()
        engines.append(ocr._engine(cfg))

    threads = [threading.Thread(target=build) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(fake_tess.created) == 1
    assert engines[0] is engines[1]


def test_engine_init_failure_is_a_loud_ocr_error(fake_tess, cfg):
    fake_tess.init_error = RuntimeError("Failed to init API, possibly an invalid tessdata path")
    with pytest.raises(ocr.OcrError, match=r"failed to initialize.*invalid tessdata path"):
        ocr._engine(cfg)


def test_recognize_keeps_tesseracts_own_order(fake_tess, cfg, tmp_path):
    fake_tess.texts = ["Left 1\nLeft 2\n\nRight 1\nRight 2\n"]
    out = ocr._recognize(Image.new("L", (8, 8), 255), cfg, tmp_path / "p.png", dpi=None)
    assert out.text == "Left 1\nLeft 2\n\nRight 1\nRight 2"


def test_engine_failure_names_the_file(fake_tess, cfg, tmp_path):
    def abort(image):
        raise RuntimeError("tesseract aborted")

    fake_tess.reply = abort
    with pytest.raises(ocr.OcrError, match=r"scan\.png.*tesseract aborted"):
        ocr._recognize(Image.new("L", (8, 8), 255), cfg, tmp_path / "scan.png", dpi=None)


def test_two_recognitions_at_once_never_share_the_engine(fake_tess, cfg, tmp_path):
    """An MCP `ingest_file` during an in-process sync: both reach the one cached
    handle, which has no lock of its own. A page's confidence and lines are read off
    the handle after its text, so they belong in the same turn."""
    ocr._engine(cfg)
    fake_tess.hold = 0.05
    fake_tess.reply = lambda image: _reading(f"page {image.width}", image.width)
    start = threading.Barrier(2)
    results: dict[int, tuple] = {}
    errors: list[BaseException] = []

    def read(width: int) -> None:
        start.wait()
        try:
            page = Image.new("L", (width, 8), 255)
            got = ocr._recognize(page, cfg, tmp_path / f"{width}.png", dpi=None)
            results[width] = (got.text, got.confidence, [line.text for line in got.lines])
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=read, args=(width,)) for width in (10, 20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert results == {10: ("page 10", 0.1, ["page 10"]), 20: ("page 20", 0.2, ["page 20"])}
    assert len(fake_tess.created) == 1


def test_every_frame_of_a_multi_page_tiff_is_read(fake_tess, cfg, tmp_path):
    """Multi-page TIFF is standard scanner and fax output. Reading frame 1 and calling
    the document done would index page 1 of a 20-page scan as the whole of it."""
    frames = [Image.new("L", (width, 10), 255) for width in (100, 101, 102)]
    tiff = tmp_path / "fax.tiff"
    frames[0].save(tiff, save_all=True, append_images=frames[1:])
    fake_tess.reply = lambda image: f"w{image.width}"
    assert ocr.ocr_image(tiff, cfg) == ("w100\n\nw101\n\nw102", ocr.ENGINE_TESSERACT)


def test_an_image_file_keeps_its_recorded_dpi(fake_tess, cfg, tmp_path):
    fake_tess.reply = lambda image: "text"
    with_dpi = tmp_path / "200dpi.png"
    Image.new("L", (8, 8), 255).save(with_dpi, dpi=(200, 200))
    ocr.ocr_image(with_dpi, cfg)
    assert fake_tess.api.resolutions == [200]

    fake_tess.api.resolutions.clear()
    without = tmp_path / "no-dpi.png"
    Image.new("L", (8, 8), 255).save(without)
    ocr.ocr_image(without, cfg)
    assert fake_tess.api.resolutions == []

    implausible = tmp_path / "50dpi.png"
    Image.new("L", (8, 8), 255).save(implausible, dpi=(50, 50))
    ocr.ocr_image(implausible, cfg)
    assert fake_tess.api.resolutions == []


@pytest.mark.parametrize(
    "mode,suffix,white,ink",
    [
        ("I;16", ".png", 65535, 40 * 257),
        ("I", ".tiff", 65535, 40 * 257),
        ("I", ".tiff", 255, 40),
        ("F", ".tiff", 1.0, 40 / 255),
    ],
)
def test_a_high_depth_grayscale_frame_reaches_the_engine_as_8_bit(
    fake_tess, cfg, tmp_path, mode, suffix, white, ink
):
    """A 16-bit scan converted without scaling has every value above 255 clipped to
    white, so its text vanishes; a floating-point frame in the 0-1 range turns black."""
    fake_tess.reply = lambda image: "text"
    frame = Image.new(mode, (20, 20), white)
    frame.putpixel((5, 5), ink)
    path = tmp_path / f"scan{suffix}"
    frame.save(path)
    ocr.ocr_image(path, cfg)
    seen = fake_tess.api.images[0].convert("L")
    assert (seen.getpixel((0, 0)), seen.getpixel((5, 5))) == (255, 40)


def test_a_big_endian_16_bit_frame_is_scaled_too():
    """Pillow computes no extrema for I;16B, the mode of big-endian 16-bit data."""
    frame = Image.new("I;16B", (20, 20), 65535)
    frame.putpixel((5, 5), 40 * 257)
    seen = ocr._on_white(frame).convert("L")
    assert (seen.getpixel((0, 0)), seen.getpixel((5, 5))) == (255, 40)


@pytest.mark.parametrize("mode,background,ink", [("RGB", (0, 0, 0), (40, 40, 40)), ("L", 0, 40)])
def test_a_colour_key_transparency_is_read_on_white(
    fake_tess, cfg, tmp_path, mode, background, ink
):
    """PNG can mark one colour transparent without an alpha channel. Passed on as it is,
    a frame keyed on black reads as a black page."""
    fake_tess.reply = lambda image: "text"
    frame = Image.new(mode, (20, 20), background)
    frame.putpixel((5, 5), ink)
    png = tmp_path / "keyed.png"
    frame.save(png, transparency=background)
    ocr.ocr_image(png, cfg)
    seen = fake_tess.api.images[0].convert("L")
    assert (seen.getpixel((0, 0)), seen.getpixel((5, 5))) == (255, 40)


def test_an_image_with_transparency_is_read_on_white(fake_tess, cfg, tmp_path):
    """An exported diagram or a screenshot: black text on a transparent background
    reads as a black page when the alpha channel is simply dropped."""
    fake_tess.reply = lambda image: "text"
    rgba = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    rgba.putpixel((5, 5), (0, 0, 0, 255))
    png = tmp_path / "diagram.png"
    rgba.save(png)
    palette = Image.new("P", (20, 20), 0)
    palette.putpalette([0, 0, 0] * 256)
    palette.putpixel((5, 5), 1)
    gif = tmp_path / "diagram.gif"
    palette.save(gif, transparency=0)

    ocr.ocr_image(png, cfg)
    ocr.ocr_image(gif, cfg)
    for seen in fake_tess.api.images:
        assert seen.mode == "RGB"
        assert seen.getpixel((0, 0)) == (255, 255, 255)
        assert seen.getpixel((5, 5)) == (0, 0, 0)


def test_ocr_image_reads_a_non_ascii_path(fake_tess, cfg, tmp_path):
    fake_tess.reply = lambda image: f"{image.width}x{image.height}"
    img = tmp_path / "Скан договора.png"
    Image.new("L", (30, 12), 255).save(img)
    assert ocr.ocr_image(img, cfg) == ("30x12", ocr.ENGINE_TESSERACT)


def test_a_jpeg_is_not_recompressed_on_its_way_to_tesseract(fake_tess, cfg, tmp_path):
    """tesserocr serializes an image in the image's own format, so a frame still
    carrying "JPEG" would be re-encoded lossily before it is read."""
    fake_tess.reply = lambda image: "text"
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16), "white").save(photo)
    ocr.ocr_image(photo, cfg)
    assert [image.format for image in fake_tess.api.images] == [None]


def test_ocr_image_undecodable_bytes_raise(fake_tess, cfg, tmp_path):
    img = tmp_path / "broken.png"
    img.write_bytes(b"not an image")
    with pytest.raises(ocr.OcrError, match=r"could not read or decode.*broken\.png"):
        ocr.ocr_image(img, cfg)


def test_ocr_image_truncated_file_is_a_decode_error_not_an_engine_one(fake_tess, cfg, tmp_path):
    """The header opens and the pixels do not: a cut-off scan is a damaged file."""
    whole = tmp_path / "whole.png"
    Image.effect_noise((200, 200), 64).save(whole)
    img = tmp_path / "cut.png"
    img.write_bytes(whole.read_bytes()[:2000])
    with pytest.raises(ocr.OcrError, match=r"could not read or decode.*cut\.png"):
        ocr.ocr_image(img, cfg)


def test_ocr_image_unreadable_file_says_reading_failed(fake_tess, cfg, tmp_path):
    """Reading the bytes and decoding them share one guard, so a missing file or a
    permission error was reported as an image that could not be decoded."""
    with pytest.raises(ocr.OcrError, match=r"could not read.*gone\.png"):
        ocr.ocr_image(tmp_path / "gone.png", cfg)


class _FakeBitmap:
    def to_pil(self):
        return Image.new("RGB", (8, 8), "white")


class _FakePage:
    def __init__(self, error: Exception | None = None):
        self._error = error
        self.scales: list[float] = []

    def render(self, scale):
        self.scales.append(scale)
        if self._error is not None:
            raise self._error
        return _FakeBitmap()


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
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "pypdfium2", fake)


def _pdf(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.7 the renderer is faked; these bytes are never parsed")
    return p


def test_pdf_pages_render_at_300_dpi_and_say_so(fake_tess, monkeypatch, cfg, tmp_path):
    page = _FakePage()
    _install_pdfium(monkeypatch, _FakePdfium(doc=_FakeDoc([page])))
    fake_tess.texts = ["page text"]
    assert ocr.ocr_pdf(_pdf(tmp_path, "scan.pdf"), cfg, [0]) == (
        {0: "page text"},
        ocr.ENGINE_TESSERACT,
    )
    assert page.scales == [300 / 72] and fake_tess.api.resolutions == [300]
    assert fake_tess.api.calls[:3] == ["SetImage", "SetSourceResolution", "GetUTF8Text"]


def test_ocr_pdf_unopenable_file_raises_ocr_error(tmp_path, monkeypatch, cfg):
    """An encrypted or malformed PDF markitdown already handled: pypdfium2 raises its own
    exception type, which left the module unnamed, uncontextualised and not an OcrError."""
    _install_pdfium(monkeypatch, _FakePdfium(open_error=RuntimeError("password required")))
    p = _pdf(tmp_path, "encrypted.pdf")
    with pytest.raises(ocr.OcrError, match=r"encrypted\.pdf.*password required"):
        ocr.ocr_pdf(p, cfg, [0])


def test_ocr_pdf_page_the_renderer_does_not_have_raises_ocr_error(tmp_path, monkeypatch, cfg):
    """pdfminer counts the pages OCR routing asks for and pypdfium2 supplies them. When
    the two disagree on page count, `doc[index]` raised a bare IndexError."""
    doc = _FakeDoc([_FakePage()])
    _install_pdfium(monkeypatch, _FakePdfium(doc=doc))
    p = _pdf(tmp_path, "short.pdf")
    with pytest.raises(ocr.OcrError, match=r"short\.pdf.*list index out of range"):
        ocr.ocr_pdf(p, cfg, [3])
    assert doc.closed, "the document must still be closed when a page lookup fails"


def test_ocr_pdf_unrenderable_page_raises_ocr_error(tmp_path, monkeypatch, cfg):
    doc = _FakeDoc([_FakePage(RuntimeError("bitmap allocation failed"))])
    _install_pdfium(monkeypatch, _FakePdfium(doc=doc))
    p = _pdf(tmp_path, "unrenderable.pdf")
    with pytest.raises(ocr.OcrError, match=r"unrenderable\.pdf.*bitmap allocation failed"):
        ocr.ocr_pdf(p, cfg, [0])
    assert doc.closed


# --- what Tesseract reports besides the text --------------------------------------------


def _image(size: tuple[int, int] = (400, 200)) -> Image.Image:
    return Image.new("L", size, 255)


def test_recognize_reports_confidence_lines_paragraph_starts_and_turns(fake_tess, cfg, tmp_path):
    """GetUTF8Text leaves a blank line before every paragraph, not only before a block,
    so a paragraph start is what a line has to carry for the text to be rebuilt."""
    fake_tess.texts = [
        _Reading(
            "Title\n\nFirst line\nSecond line\n\nNew paragraph\n",
            70,
            [
                _Line((10, 10, 200, 40), "Title", block=True, para=True, orientation=3),
                _Line((10, 80, 300, 110), "First line", block=True, para=True, orientation=3),
                _Line((10, 120, 300, 150), "Second line", orientation=3),
                _Line((10, 160, 300, 190), "New paragraph", para=True, orientation=3),
            ],
        )
    ]
    got = ocr._recognize(_image(), cfg, tmp_path / "p.png", dpi=None)
    assert got.text == "Title\n\nFirst line\nSecond line\n\nNew paragraph"
    assert got.confidence == 0.7
    assert got.lines == (
        ocr.Line((10, 10, 200, 40), "Title", starts_paragraph=True),
        ocr.Line((10, 80, 300, 110), "First line", starts_paragraph=True),
        ocr.Line((10, 120, 300, 150), "Second line"),
        ocr.Line((10, 160, 300, 190), "New paragraph", starts_paragraph=True),
    )
    assert got.turns == 3


def test_turns_are_the_orientation_most_blocks_report(fake_tess, cfg, tmp_path):
    fake_tess.texts = [
        _Reading(
            "a\nb\nc\nd\n",
            80,
            [
                _Line((0, 0, 9, 9), "a", block=True, orientation=1),
                _Line((0, 10, 9, 19), "b", orientation=0),
                _Line((0, 20, 9, 29), "c", block=True, orientation=3),
                _Line((0, 30, 9, 39), "d", block=True, orientation=3),
            ],
        )
    ]
    assert ocr._recognize(_image(), cfg, tmp_path / "p.png", dpi=None).turns == 3


def test_an_empty_page_has_no_lines_and_no_confidence(fake_tess, cfg, tmp_path):
    """The real iterator of an empty page exists and raises "No text returned"."""
    fake_tess.texts = [""]
    got = ocr._recognize(_image(), cfg, tmp_path / "p.png", dpi=None)
    assert (got.text, got.confidence, got.lines, got.turns) == ("", 0.0, (), 0)


def test_a_line_the_iterator_has_no_text_for_is_left_out(fake_tess, cfg, tmp_path):
    fake_tess.texts = [
        _Reading(
            "kept\nalso kept\n",
            88,
            [
                _Line((0, 0, 90, 30), "kept", block=True, para=True),
                _Line((0, 40, 90, 70), None, block=True, para=True, orientation=2),
                _Line((0, 80, 90, 110), "also kept", block=True, para=True),
            ],
        )
    ]
    got = ocr._recognize(_image(), cfg, tmp_path / "p.png", dpi=None)
    assert [line.text for line in got.lines] == ["kept", "also kept"]
    assert got.turns == 0


def test_a_line_without_text_hands_its_block_and_paragraph_start_on(fake_tess, cfg, tmp_path):
    """Its block's orientation still counts, and the blank line before its paragraph
    still comes, at the next line that has text."""
    fake_tess.texts = [
        _Reading(
            "body\n\nnext paragraph\n",
            80,
            [
                _Line((0, 0, 90, 30), None, block=True, para=True, orientation=3),
                _Line((0, 40, 90, 70), "body", orientation=3),
                _Line((0, 80, 90, 110), None, para=True, orientation=3),
                _Line((0, 120, 90, 150), "next paragraph", orientation=3),
            ],
        )
    ]
    got = ocr._recognize(_image(), cfg, tmp_path / "p.png", dpi=None)
    assert got.lines == (
        ocr.Line((0, 40, 90, 70), "body", starts_paragraph=True),
        ocr.Line((0, 120, 90, 150), "next paragraph", starts_paragraph=True),
    )
    assert got.turns == 3


# --- Apple Vision re-reads the pages Tesseract is unsure of ------------------------------


class _FakeVision:
    """`ocr_vision.read_lines`, answering with lines in the page's own frame as it does."""

    def __init__(self):
        self.calls: list[tuple[tuple[int, int], int, list[str]]] = []
        self.lines: list[ocr.Line] = []
        self.errors: list[Exception | None] = []

    def read_lines(self, image, turns, languages):
        self.calls.append((image.size, turns, languages))
        if self.errors and (error := self.errors.pop(0)) is not None:
            raise error
        return list(self.lines)


@pytest.fixture
def vision(monkeypatch):
    fake = _FakeVision()
    monkeypatch.setattr(ocr_vision, "available", lambda: True)
    monkeypatch.setattr(ocr_vision, "read_lines", fake.read_lines)
    return fake


_UNSURE = _Reading(
    "Dogovor postavki\nnomer 7OO\n",
    70,
    [
        _Line((10, 10, 300, 40), "Dogovor postavki", block=True, para=True),
        _Line((10, 60, 300, 90), "nomer 7OO"),
    ],
)
_VISION_READS = [
    ocr.Line((9, 8, 302, 42), "Договор поставки"),
    ocr.Line((8, 58, 305, 92), "nomer 700"),
]


def _read_one_page(how: str, monkeypatch, cfg, tmp_path, size=(400, 120)) -> tuple[str, str]:
    if how == "image":
        path = tmp_path / "scan.png"
        _image(size).save(path)
        return ocr.ocr_image(path, cfg)
    _install_pdfium(monkeypatch, _FakePdfium(doc=_FakeDoc([_FakePage()])))
    texts, engine = ocr.ocr_pdf(_pdf(tmp_path, "scan.pdf"), cfg, [0])
    return texts[0], engine


@pytest.mark.parametrize("how", ["image", "pdf"])
def test_a_page_under_the_threshold_is_spliced_with_visions_reading(
    fake_tess, vision, monkeypatch, cfg, tmp_path, how
):
    fake_tess.texts = [_UNSURE]
    vision.lines = _VISION_READS
    text, engine = _read_one_page(how, monkeypatch, cfg, tmp_path)
    assert text == "Договор поставки\nnomer 700"
    assert engine == ocr.ENGINE_TESSERACT_VISION == "tesseract+vision"
    size = (400, 120) if how == "image" else (8, 8)
    assert vision.calls == [(size, 0, ["ru-RU", "en-US"])]


@pytest.mark.parametrize("how", ["image", "pdf"])
@pytest.mark.parametrize("confidence", [85, 90])
def test_a_page_at_or_above_the_threshold_keeps_tesseracts_own_text(
    fake_tess, vision, monkeypatch, cfg, tmp_path, how, confidence
):
    fake_tess.texts = [_reading("  Dogovor   postavki\nnomer 7OO\n\n", confidence)]
    vision.lines = _VISION_READS
    text, engine = _read_one_page(how, monkeypatch, cfg, tmp_path)
    assert (text, engine) == ("Dogovor   postavki\nnomer 7OO", ocr.ENGINE_TESSERACT)
    assert vision.calls == []


@pytest.mark.parametrize("how", ["image", "pdf"])
def test_without_vision_an_unsure_page_keeps_tesseracts_text(
    fake_tess, vision, monkeypatch, cfg, tmp_path, how
):
    monkeypatch.setattr(ocr_vision, "available", lambda: False)
    fake_tess.texts = [_UNSURE]
    vision.lines = _VISION_READS
    text, engine = _read_one_page(how, monkeypatch, cfg, tmp_path)
    assert (text, engine) == ("Dogovor postavki\nnomer 7OO", ocr.ENGINE_TESSERACT)
    assert vision.calls == []


def test_a_page_tesseract_found_no_words_on_is_read_by_vision(
    fake_tess, vision, monkeypatch, cfg, tmp_path
):
    fake_tess.texts = [""]
    vision.lines = [ocr.Line((50, 50, 150, 80), "Печать")]
    assert _read_one_page("image", monkeypatch, cfg, tmp_path) == (
        "Печать",
        ocr.ENGINE_TESSERACT_VISION,
    )


@pytest.mark.parametrize("how", ["image", "pdf"])
@pytest.mark.parametrize(
    "tesseract", ["", "  Dogovor   postavki\nnomer 7OO\n\n"], ids=["blank", "unsure"]
)
def test_a_page_vision_finds_nothing_on_keeps_tesseracts_text_and_name(
    fake_tess, vision, monkeypatch, cfg, tmp_path, how, tesseract
):
    """A blank page (the back of a duplex scan) is routine: Vision reads it too, adds
    nothing, and the document must read and be named as it is on any other platform."""
    fake_tess.texts = [_reading(tesseract, 0 if not tesseract else 70)]
    vision.lines = []
    assert _read_one_page(how, monkeypatch, cfg, tmp_path) == (
        tesseract.strip(),
        ocr.ENGINE_TESSERACT,
    )
    assert len(vision.calls) == 1


@pytest.mark.parametrize(
    "setting,confidence,reread", [("0", 0, False), ("95", 90, True), ("60", 70, False)]
)
def test_the_threshold_is_the_configured_one_and_zero_turns_the_reader_off(
    fake_tess, vision, monkeypatch, tmp_path, setting, confidence, reread
):
    cfg = load_config({"RAG_OCR_SECOND_READER_BELOW": setting}, cwd=tmp_path)
    fake_tess.texts = [_reading("" if confidence == 0 else "Dogovor", confidence)]
    vision.lines = [ocr.Line((10, 10, 100, 40), "Договор")]
    _, engine = _read_one_page("image", monkeypatch, cfg, tmp_path)
    assert (bool(vision.calls), engine == ocr.ENGINE_TESSERACT_VISION) == (reread, reread)


def test_a_failing_second_reader_keeps_tesseracts_text_and_says_so(
    fake_tess, vision, fake_embedder, caplog, tmp_path
):
    """The file is never lost to the second reader; the warning is what tells the owner
    that a page they would have had re-read was not."""
    fake_tess.texts = [_UNSURE]
    vision.errors = [RuntimeError("Vision refused the page")]
    scan = tmp_path / "invoice.png"
    _image().save(scan)
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    store = Store(tmp_path / ".minirag" / "v2", dim=fake_embedder.dim)
    with caplog.at_level(logging.WARNING, logger="minirag_mcp.ocr"):
        result = Pipeline(store, fake_embedder, cfg).ingest_file(scan)
    assert result.chunk_count >= 1
    assert store.get_source(str(scan)).ocr_engine == ocr.ENGINE_TESSERACT
    rows = store._table.search().where(f"source = '{scan}'").limit(10_000).to_list()
    assert "nomer 7OO" in " ".join(r["text"] for r in rows)
    ours = [r for r in caplog.records if r.name == "minirag_mcp.ocr"]
    [warning] = [r for r in ours if r.levelno == logging.WARNING]
    assert "invoice.png" in warning.getMessage() and "page 0" in warning.getMessage()
    assert "Vision refused the page" in warning.getMessage()


def test_one_spliced_page_marks_the_document_and_a_failed_one_does_not(
    fake_tess, vision, monkeypatch, cfg, tmp_path
):
    sure = _reading("Page one reads clean", 93)
    _install_pdfium(monkeypatch, _FakePdfium(doc=_FakeDoc([_FakePage() for _ in range(3)])))
    fake_tess.texts = [sure, _UNSURE, _UNSURE]
    vision.lines = _VISION_READS
    vision.errors = [RuntimeError("Vision refused the page"), None]
    texts, engine = ocr.ocr_pdf(_pdf(tmp_path, "scan.pdf"), cfg, [0, 1, 2])
    assert texts == {
        0: "Page one reads clean",
        1: "Dogovor postavki\nnomer 7OO",
        2: "Договор поставки\nnomer 700",
    }
    assert engine == ocr.ENGINE_TESSERACT_VISION
    assert len(vision.calls) == 2

    fake_tess.texts = [sure, _UNSURE]
    vision.errors = [RuntimeError("Vision refused the page")]
    _, engine = ocr.ocr_pdf(_pdf(tmp_path, "scan.pdf"), cfg, [0, 1])
    assert engine == ocr.ENGINE_TESSERACT


def test_a_turned_page_is_spliced_upright(fake_tess, vision, monkeypatch, cfg, tmp_path):
    """Both readers give boxes in the frame of the page as scanned, here turned a
    quarter. Rows and left-to-right only mean anything upright: on the sideways page,
    Vision's tall line boxes reach across into the next line and sort by the wrong axis."""
    w, h = 300, 1000  # the scanned page; upright it is 1000 x 300

    def scanned(box):  # an upright box on the page as scanned, one turn short of upright
        x0, y0, x1, y1 = box
        return (w - y1, x0, w - y0, x1)

    upright_tess = [(100, 100, 900, 140), (100, 150, 900, 190), (100, 200, 900, 240)]
    fake_tess.texts = [
        _Reading(
            "one\ntwo\nthree\n",
            60,
            [
                _Line(scanned(box), text, block=i == 0, para=i == 0, orientation=1)
                for i, (box, text) in enumerate(
                    zip(upright_tess, ["one", "two", "three"], strict=True)
                )
            ],
        )
    ]
    upright_vision = [(100, 95, 900, 141), (100, 138, 900, 202), (100, 195, 900, 247)]
    vision.lines = [
        ocr.Line(scanned(box), text)
        for box, text in zip(upright_vision, ["ONE", "TWO", "THREE"], strict=True)
    ]
    path = tmp_path / "sideways.png"
    _image((w, h)).save(path)
    assert ocr.ocr_image(path, cfg) == ("ONE\nTWO\nTHREE", ocr.ENGINE_TESSERACT_VISION)
    assert vision.calls == [((w, h), 1, ["ru-RU", "en-US"])]


# --- splice: Tesseract's lines and order, Vision's reading of them ----------------------


def _tess(box, text, para=False):
    return ocr.Line(box, text, starts_paragraph=para)


def test_splice_keeps_a_line_vision_does_not_overlap():
    tess = [_tess((0, 0, 100, 20), "alpha", True), _tess((0, 30, 100, 50), "beta")]
    vision = [ocr.Line((0, 29, 100, 51), "BETA")]
    assert ocr.splice(tess, vision) == "alpha\nBETA"


def test_splice_appends_what_only_vision_found_after_the_page_in_visions_own_order():
    tess = [_tess((0, 0, 100, 20), "alpha", True)]
    vision = [
        ocr.Line((0, 0, 100, 21), "ALPHA"),
        ocr.Line((0, 200, 90, 220), "below"),
        ocr.Line((150, 100, 300, 120), "right"),
        ocr.Line((0, 102, 140, 118), "left"),
    ]
    assert ocr.splice(tess, vision) == "ALPHA\n\nbelow\nright\nleft"


def test_splice_does_not_interleave_the_columns_vision_read_one_after_the_other():
    tess = [_tess((0, 0, 100, 20), "alpha", True)]
    vision = [
        ocr.Line((0, 100, 90, 120), "L1"),
        ocr.Line((0, 140, 90, 160), "L2"),
        ocr.Line((200, 100, 290, 120), "R1"),
        ocr.Line((200, 140, 290, 160), "R2"),
    ]
    assert ocr.splice(tess, vision) == "alpha\n\nL1\nL2\nR1\nR2"


def test_splice_keeps_every_tesseract_line_in_a_region_vision_read_nothing_of():
    tess = [_tess((0, y, 100, y + 20), f"t{y}", y == 0) for y in (0, 40, 600, 640, 680)]
    vision = [ocr.Line((0, 39, 100, 61), "V40")]
    assert ocr.splice(tess, vision) == "t0\nV40\nt600\nt640\nt680"


def test_splice_emits_one_vision_line_over_two_tesseract_lines_once():
    tess = [_tess((0, 0, 100, 20), "top half", True), _tess((0, 25, 100, 45), "bottom half")]
    vision = [ocr.Line((0, 2, 100, 43), "one line")]
    assert ocr.splice(tess, vision) == "one line"


def test_splice_replaces_a_line_by_every_vision_line_over_it_left_to_right():
    tess = [_tess((0, 0, 300, 20), "a wide row", True), _tess((0, 40, 300, 60), "next")]
    vision = [ocr.Line((160, 0, 300, 20), "right cell"), ocr.Line((0, 0, 150, 20), "left cell")]
    assert ocr.splice(tess, vision) == "left cell\nright cell\nnext"


def test_splice_keeps_tesseracts_paragraph_breaks():
    tess = [
        _tess((0, 0, 100, 20), "a1", True),
        _tess((0, 30, 100, 50), "a2"),
        _tess((0, 70, 100, 90), "b1", True),
        _tess((0, 100, 100, 120), "b2"),
    ]
    vision = [ocr.Line((0, y0 - 1, 100, y1 + 1), f"V{y0}") for y0, y1 in [(0, 20), (30, 50)]]
    vision += [ocr.Line((0, 70, 100, 90), "B1")]
    assert ocr.splice(tess, vision) == "V0\nV30\n\nB1\nb2"


def test_splice_carries_a_paragraph_break_past_a_line_vision_merged_away():
    tess = [
        _tess((0, 0, 100, 20), "a1", True),
        _tess((0, 25, 100, 45), "b1", True),
        _tess((0, 60, 100, 80), "b2"),
    ]
    vision = [ocr.Line((0, 2, 100, 43), "a1 b1"), ocr.Line((0, 60, 100, 80), "B2")]
    assert ocr.splice(tess, vision) == "a1 b1\n\nB2"


@pytest.mark.parametrize("top,same", [(9, True), (10, False), (0, True)])
def test_lines_overlap_by_more_than_half_the_smaller_height(top, same):
    """Vision's boxes are taller than the glyphs: a neighbour's box reaching a little
    into a line is not that line."""
    tess = [_tess((0, 0, 100, 20), "tess", True)]
    vision = [ocr.Line((50, top, 300, top + 40), "vision")]
    assert ocr.splice(tess, vision) == ("vision" if same else "tess\n\nvision")


def test_splice_of_nothing_is_empty():
    assert ocr.splice([], []) == ""


def _tilted(k: int) -> tuple[int, int, int, int]:
    """Line `k` of a page tilted about 2 degrees, at a 60 px pitch: at 300 dpi its
    full-width box is 130 px tall, and reaches more than halfway into each neighbour's."""
    return (0, 60 * k, 2400, 60 * k + 130)


def test_splice_keeps_a_line_only_tesseract_read_on_a_tilted_page():
    tess = [_tess(_tilted(k), f"t{k}", k == 0) for k in range(6)]
    vision = [ocr.Line(_tilted(k), f"V{k}") for k in range(6) if k != 3]
    assert ocr.splice(tess, vision) == "V0\nV1\nV2\nt3\nV4\nV5"


def test_a_vision_line_no_tesseract_line_is_clearly_read_by_follows_the_page():
    """Tesseract missed line 2 and Vision line 1: Vision's line 2 covers half of each
    Tesseract line beside it, and reads neither."""
    tess = [_tess(_tilted(k), f"t{k}", k == 0) for k in (0, 1, 3)]
    vision = [ocr.Line(_tilted(k), f"V{k}") for k in (0, 2)]
    assert ocr.splice(tess, vision) == "V0\nt1\nt3\n\nV2"


def test_splice_loses_and_repeats_no_line_from_flat_pages_to_tilted_ones():
    """Boxes from 0.6 to 3.2 pitches tall (flat to about 3 degrees), each line boxed a few
    pixels apart by the two readers, and a third of the lines unread by Vision: every line
    comes out once, in order, as Vision read it where it did."""
    rng = random.Random(0)
    for _ in range(300):
        pitch = rng.randint(40, 70)
        height = round(pitch * rng.uniform(0.6, 3.2))
        tess, vision, expected = [], [], []
        for k in range(rng.randint(2, 12)):
            box = (rng.randint(0, 40), k * pitch, 2400 - rng.randint(0, 40), k * pitch + height)
            tess.append(_tess(box, f"t{k}", k == 0))
            if rng.random() < 2 / 3:
                seen = tuple(edge + rng.randint(-3, 3) for edge in box)
                vision.append(ocr.Line(seen, f"V{k}"))
                expected.append(f"V{k}")
            else:
                expected.append(f"t{k}")
        assert ocr.splice(tess, vision) == "\n".join(expected), (pitch, height)


def _font(size: int = 44) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _page(lines: list[str], size: tuple[int, int], *, x: int = 150) -> Image.Image:
    page = Image.new("L", size, 255)
    draw = ImageDraw.Draw(page)
    for i, line in enumerate(lines):
        draw.text((x, 150 + 100 * i), line, fill=0, font=_font())
    return page


@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
def test_real_engine_reads_both_pages_of_a_two_frame_tiff(tmp_path):
    """The claim the fake above cannot make: Pillow really hands back both frames of a
    two-page TIFF written the way a scanner writes one, and both reach the index."""
    frames = [_page([marker] * 3, (1600, 500)) for marker in ("FIRST PAGE 111", "SECOND PAGE 222")]
    p = tmp_path / "scan.tiff"
    frames[0].save(p, save_all=True, append_images=frames[1:], dpi=(300, 300))
    text, _ = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
    assert "111" in text and "222" in text
    assert text.index("111") < text.index("222")


@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
def test_real_engine_reads_a_generated_page(tmp_path):
    p = tmp_path / "scan.png"
    _page(["INVOICE 12345 TOTAL 999"], (1600, 400)).save(p, dpi=(300, 300))
    text, _ = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
    assert "12345" in text and "999" in text


@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
@pytest.mark.parametrize("angle", [90, 180])
def test_real_engine_reads_a_rotated_page(tmp_path, angle):
    """Orientation is Tesseract's own OSD, which needs enough text to decide: a slip of
    three short lines turned 180 degrees reads as noise."""
    lines = [
        f"The quick brown fox number {700 + i} jumps over the lazy dog again" for i in range(12)
    ]
    p = tmp_path / "rotated.png"
    _page(lines, (2480, 1400)).rotate(angle, expand=True).save(p, dpi=(300, 300))
    text, _ = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
    assert "700" in text and "711" in text


@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
def test_real_engine_reads_rendered_pdf_page(tmp_path):
    """Pins the pypdfium2 render -> to_pil -> _recognize path in ocr_pdf, which the
    image-file tests never touch."""
    p = _write(tmp_path, ["OCR PDF CHECK 555 " * 5])
    texts, _ = ocr.ocr_pdf(p, load_config({}, cwd=tmp_path), [0])
    assert "555" in texts[0]


@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
def test_real_engine_keeps_the_columns_of_a_two_column_page_apart(tmp_path):
    page = Image.new("L", (2480, 1400), 255)
    draw = ImageDraw.Draw(page)
    for i in range(10):
        y = 150 + 100 * i
        draw.text((150, y), f"Left column line {100 + i} reads here", fill=0, font=_font())
        draw.text((1340, y), f"Right side entry {500 + i} goes on", fill=0, font=_font())
    p = tmp_path / "two-columns.png"
    page.save(p, dpi=(300, 300))
    text, _ = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
    assert all(f"{100 + i}" in text and f"{500 + i}" in text for i in range(10))
    assert text.index("109") < text.index("500")


@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_real_engine_reports_the_turns_that_make_a_page_upright(tmp_path, quarter_turns):
    """The page below is turned counter-clockwise; turning it back takes the rest of a
    full turn. Its line boxes stay in the frame of the page as handed over."""
    lines = [
        f"The quick brown fox number {700 + i} jumps over the lazy dog again" for i in range(12)
    ]
    page = _page(lines, (2480, 1400)).rotate(90 * quarter_turns, expand=True)
    got = ocr._recognize(page, load_config({}, cwd=tmp_path), tmp_path / "p.png", dpi=300)
    assert got.turns == (4 - quarter_turns) % 4
    assert got.confidence > 0.85
    assert len(got.lines) == 12 and got.lines[0].starts_paragraph
    assert "700" in got.lines[0].text and "711" in got.lines[-1].text
    wide = [x1 - x0 > y1 - y0 for x0, y0, x1, y1 in (line.box for line in got.lines)]
    assert wide == [quarter_turns % 2 == 0] * 12


@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
def test_real_engine_reports_an_empty_page_as_no_lines_and_no_confidence(tmp_path):
    page = Image.new("L", (800, 600), 255)
    got = ocr._recognize(page, load_config({}, cwd=tmp_path), tmp_path / "p.png", dpi=300)
    assert (got.text, got.confidence, got.lines, got.turns) == ("", 0.0, (), 0)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("rus+eng", "rus+eng"),
        ("eng", "eng"),
        (" ENG + rus ", "eng+rus"),
        ("eslav", "rus+eng"),
        ("en", "eng"),
    ],
)
def test_tesseract_languages(value, expected):
    assert ocr.tesseract_languages(value) == expected


@pytest.mark.parametrize("value", ["", "ch", "deu", "rus+", "rus+rus", "osd"])
def test_tesseract_languages_refuses(value):
    with pytest.raises(ValueError, match=r"rus.*eng.*eslav"):
        ocr.tesseract_languages(value)


class _TessdataServer:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.hits: list[str] = []


@pytest.fixture
def tessdata_server(monkeypatch):
    import hashlib
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = _TessdataServer()
    for name in ("eng", "osd", "rus"):
        state.files[f"/{ocr.TESSDATA_COMMIT}/{name}.traineddata"] = f"{name} data".encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.hits.append(self.path)
            body = state.files.get(self.path)
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    monkeypatch.setattr(
        ocr, "_TESSDATA_URL", f"http://127.0.0.1:{port}/{{commit}}/{{name}}.traineddata"
    )
    monkeypatch.setattr(
        ocr,
        "TESSDATA_SHA256",
        {n: hashlib.sha256(f"{n} data".encode()).hexdigest() for n in ("eng", "osd", "rus")},
    )
    yield state
    server.shutdown()
    server.server_close()


def test_tessdata_is_fetched_once_from_the_pinned_commit(tmp_path, tessdata_server):
    d = ocr.ensure_tessdata(tmp_path, "rus+eng")
    assert d == ocr.tessdata_dir(tmp_path) and d.parent == tmp_path / "ocr"
    assert sorted(p.name for p in d.iterdir()) == [
        "eng.traineddata",
        "osd.traineddata",
        "rus.traineddata",
    ]
    assert len(tessdata_server.hits) == 3
    assert all(h.startswith(f"/{ocr.TESSDATA_COMMIT}/") for h in tessdata_server.hits)
    ocr.ensure_tessdata(tmp_path, "rus+eng")
    assert len(tessdata_server.hits) == 3


def test_only_the_languages_asked_for_and_osd_are_fetched(tmp_path, tessdata_server):
    ocr.ensure_tessdata(tmp_path, "eng")
    assert sorted(h.rsplit("/", 1)[1] for h in tessdata_server.hits) == [
        "eng.traineddata",
        "osd.traineddata",
    ]


def test_a_download_with_the_wrong_digest_is_discarded(tmp_path, tessdata_server):
    tessdata_server.files[f"/{ocr.TESSDATA_COMMIT}/rus.traineddata"] = b"tampered"
    with pytest.raises(ocr.OcrError, match=r"rus\.traineddata.*sha256"):
        ocr.ensure_tessdata(tmp_path, "rus")
    leftovers = [
        p.name for p in ocr.tessdata_dir(tmp_path).iterdir() if p.name.startswith(("rus", ".rus"))
    ]
    assert leftovers == []


def test_a_cached_file_with_the_wrong_digest_is_refused(tmp_path, tessdata_server):
    d = ocr.tessdata_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "osd.traineddata").write_bytes(b"x")
    with pytest.raises(ocr.OcrError, match="delete it to download it again"):
        ocr.ensure_tessdata(tmp_path, "eng")


def test_an_unreachable_host_is_a_loud_ocr_error(tmp_path, monkeypatch, dead_url):
    monkeypatch.setattr(ocr, "_TESSDATA_URL", dead_url + "/{commit}/{name}.traineddata")
    with pytest.raises(
        ocr.OcrError, match=r"could not download (eng|osd)\.traineddata .*127\.0\.0\.1"
    ):
        ocr.ensure_tessdata(tmp_path, "eng")
