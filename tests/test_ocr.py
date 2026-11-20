"""OCR module tests. Fast tests run on a fake tesserocr; slow ones need the extra."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

from minirag_mcp import ocr
from minirag_mcp.config import load_config
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


class _FakeTessApi:
    """One PyTessBaseAPI handle: it holds the last image it was given, as the real one
    does, so two callers interleaving on one handle would read each other's page."""

    def __init__(self, tess: _FakeTesserocr):
        self._tess = tess
        self._image = None
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
        time.sleep(self._tess.hold)
        self._image = image

    def SetSourceResolution(self, dpi):
        self.calls.append("SetSourceResolution")
        self.resolutions.append(dpi)

    def GetUTF8Text(self):
        self.calls.append("GetUTF8Text")
        try:
            return self._tess.reply(self._image)
        finally:
            self._tess.busy.release()


class _FakeTesserocr:
    """Just enough tesserocr to drive ocr.py without the extra installed."""

    PSM = SimpleNamespace(AUTO_OSD=1)

    def __init__(self):
        self.created: list[dict] = []
        self.apis: list[_FakeTessApi] = []
        self.texts: list[str] = []
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
    assert out == "Left 1\nLeft 2\n\nRight 1\nRight 2"


def test_engine_failure_names_the_file(fake_tess, cfg, tmp_path):
    def abort(image):
        raise RuntimeError("tesseract aborted")

    fake_tess.reply = abort
    with pytest.raises(ocr.OcrError, match=r"scan\.png.*tesseract aborted"):
        ocr._recognize(Image.new("L", (8, 8), 255), cfg, tmp_path / "scan.png", dpi=None)


def test_two_recognitions_at_once_never_share_the_engine(fake_tess, cfg, tmp_path):
    """An MCP `ingest_file` during an in-process sync: both reach the one cached
    handle, which has no lock of its own."""
    ocr._engine(cfg)
    fake_tess.hold = 0.05
    fake_tess.reply = lambda image: f"page {image.width}"
    start = threading.Barrier(2)
    results: dict[int, str] = {}
    errors: list[BaseException] = []

    def read(width: int) -> None:
        start.wait()
        try:
            page = Image.new("L", (width, 8), 255)
            results[width] = ocr._recognize(page, cfg, tmp_path / f"{width}.png", dpi=None)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=read, args=(width,)) for width in (10, 20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert results == {10: "page 10", 20: "page 20"}
    assert len(fake_tess.created) == 1


def test_every_frame_of_a_multi_page_tiff_is_read(fake_tess, cfg, tmp_path):
    """Multi-page TIFF is standard scanner and fax output. Reading frame 1 and calling
    the document done would index page 1 of a 20-page scan as the whole of it."""
    frames = [Image.new("L", (width, 10), 255) for width in (100, 101, 102)]
    tiff = tmp_path / "fax.tiff"
    frames[0].save(tiff, save_all=True, append_images=frames[1:])
    fake_tess.reply = lambda image: f"w{image.width}"
    assert ocr.ocr_image(tiff, cfg) == "w100\n\nw101\n\nw102"


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
    assert ocr.ocr_image(img, cfg) == "30x12"


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
    assert ocr.ocr_pdf(_pdf(tmp_path, "scan.pdf"), cfg, [0]) == {0: "page text"}
    assert page.scales == [300 / 72] and fake_tess.api.resolutions == [300]
    assert fake_tess.api.calls == ["SetImage", "SetSourceResolution", "GetUTF8Text"]


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
    text = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
    assert "111" in text and "222" in text
    assert text.index("111") < text.index("222")


@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
def test_real_engine_reads_a_generated_page(tmp_path):
    p = tmp_path / "scan.png"
    _page(["INVOICE 12345 TOTAL 999"], (1600, 400)).save(p, dpi=(300, 300))
    text = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
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
    text = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
    assert "700" in text and "711" in text


@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
def test_real_engine_reads_rendered_pdf_page(tmp_path):
    """Pins the pypdfium2 render -> to_pil -> _recognize path in ocr_pdf, which the
    image-file tests never touch."""
    p = _write(tmp_path, ["OCR PDF CHECK 555 " * 5])
    texts = ocr.ocr_pdf(p, load_config({}, cwd=tmp_path), [0])
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
    text = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
    assert all(f"{100 + i}" in text and f"{500 + i}" in text for i in range(10))
    assert text.index("109") < text.index("500")


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
