"""Apple Vision as the second reader. The geometry runs everywhere; real Vision on macOS."""

from __future__ import annotations

import contextlib
import io
import sys
import types

import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from minirag_mcp import ocr, ocr_vision
from minirag_mcp.config import load_config

# conftest replaces `read_lines` in every fast test; the fake-Vision tests below need the real one.
_READ_LINES = ocr_vision.read_lines


@pytest.mark.parametrize(
    "platform,missing,expected",
    [
        ("darwin", None, True),
        ("darwin", "Vision", False),
        ("darwin", "Quartz", False),
        ("darwin", "objc", False),
        ("linux", None, False),
        ("win32", None, False),
    ],
)
def test_available_needs_macos_and_the_bindings(monkeypatch, platform, missing, expected):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(
        ocr_vision.importlib.util, "find_spec", lambda name: None if name == missing else object()
    )
    assert ocr_vision.available() is expected


@pytest.mark.parametrize(
    "ocr_lang,expected",
    [
        ("rus+eng", ["ru-RU", "en-US"]),
        ("rus", ["ru-RU"]),
        ("eng", ["en-US"]),
        ("eng+rus", ["en-US", "ru-RU"]),
    ],
)
def test_languages_are_tesseracts_in_visions_spelling(ocr_lang, expected):
    assert ocr_vision.languages(ocr_lang) == expected


# A 1000-pixel page read in two bands, each reaching 75 pixels past the midline.
TOP, BOTTOM = (0.0, 575.0), (425.0, 1000.0)


def test_the_bands_are_two_halves_reaching_15_percent_of_a_half_past_the_midline():
    assert ocr_vision._bands(1000) == [TOP, BOTTOM]


def _kept(seen: list[ocr_vision._Seen]) -> list[str]:
    return [line.text for line in ocr_vision._deduplicated(seen)]


def test_two_overlapping_lines_of_one_band_are_both_kept():
    """Vision's boxes are taller than the glyphs, so neighbouring lines of one band
    overlap. Dropping one of them deleted whole lines of real scans."""
    seen = [
        ocr_vision._Seen(TOP, (0, 100, 500, 160), "first line"),
        ocr_vision._Seen(TOP, (0, 120, 500, 180), "second line"),
    ]
    assert _kept(seen) == ["first line", "second line"]


@pytest.mark.parametrize(
    "top_box,bottom_box,kept",
    [
        ((0, 540, 500, 570), (0, 538, 500, 571), "bottom"),  # near the top band's edge
        ((0, 430, 500, 460), (0, 431, 500, 462), "top"),  # near the bottom band's edge
    ],
)
def test_a_line_read_in_both_bands_is_kept_once_from_the_band_it_sits_deeper_in(
    top_box, bottom_box, kept
):
    seen = [ocr_vision._Seen(TOP, top_box, "top"), ocr_vision._Seen(BOTTOM, bottom_box, "bottom")]
    assert _kept(seen) == [kept]


def test_lines_of_the_two_bands_that_barely_touch_are_both_kept():
    seen = [
        ocr_vision._Seen(TOP, (0, 500, 500, 540), "upper"),
        ocr_vision._Seen(BOTTOM, (0, 525, 500, 565), "lower"),
    ]
    assert _kept(seen) == ["upper", "lower"]


def test_a_box_normalised_to_its_band_from_the_bottom_left_lands_in_page_pixels():
    box = ocr_vision._pixel_box((0.1, 0.0, 0.5, 0.1), BOTTOM, 2000)
    assert box == pytest.approx((200, 942.5, 1200, 1000))
    box = ocr_vision._pixel_box((0.0, 0.9, 1.0, 0.1), TOP, 2000)
    assert box == pytest.approx((0, 0, 2000, 57.5))


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_a_box_turned_upright_and_back_lands_on_itself(turns):
    """Pinned to Pillow's own rotation, which turns the page Vision reads."""
    size = (300, 200)
    box = (40, 30, 120, 70)
    page = Image.new("L", size, 255)
    ImageDraw.Draw(page).rectangle((40, 30, 119, 69), fill=0)
    inked = ImageOps.invert(page.rotate(90 * turns, expand=True)).getbbox()
    assert ocr._upright_box(box, turns, size) == inked
    assert ocr_vision._page_box(inked, turns, size) == box


# --- read_lines against a fake Vision that frames boxes as the real one does ----------


class _FakeVision:
    """Stands in for pyobjc's objc, Quartz and Vision. Each request answers from `script`
    (one list of (text, page box) per band request, in order), the way Vision does: it
    sees the image it was handed, and reports a box normalised to the region it was asked
    to read, from that region's bottom left."""

    def __init__(self, script, page_size):
        self.script, self.page_size = script, page_size
        self.received: tuple[int, int] | None = None

    def install(self, monkeypatch):
        fake = self

        class Request:
            @classmethod
            def alloc(cls):
                return cls()

            def init(self):
                return self

            def setRecognitionLevel_(self, level):
                pass

            def setRecognitionLanguages_(self, languages):
                pass

            def setUsesLanguageCorrection_(self, flag):
                pass

            def setRegionOfInterest_(self, region):
                self.region = region

            def results(self):
                return self.answers

        class Handler:
            @classmethod
            def alloc(cls):
                return cls()

            def initWithData_options_(self, data, options):
                fake.received = Image.open(io.BytesIO(data)).size
                return self

            def performRequests_error_(self, requests, error):
                width, height = fake.received
                for request, lines in zip(requests, fake.script, strict=True):
                    request.answers = [
                        _observation(text, box, request.region, fake.page_size, (width, height))
                        for text, box in lines
                    ]
                return True, None

        pairs = {
            "objc": {"autorelease_pool": contextlib.nullcontext},
            "Quartz": {"CGRectMake": lambda x, y, w, h: _rect(x, y, w, h)},
            "Vision": {
                "VNImageRequestHandler": Handler,
                "VNRecognizeTextRequest": Request,
                "VNRequestTextRecognitionLevelAccurate": 1,
            },
        }
        for name, members in pairs.items():
            module = types.ModuleType(name)
            module.__dict__.update(members)
            monkeypatch.setitem(sys.modules, name, module)


def _rect(x, y, w, h):
    return types.SimpleNamespace(
        origin=types.SimpleNamespace(x=x, y=y), size=types.SimpleNamespace(width=w, height=h)
    )


def _observation(text, page_box, region, page_size, seen_size):
    """`page_box` (upright page pixels, from the top left) as Vision reports it."""
    sx, sy = seen_size[0] / page_size[0], seen_size[1] / page_size[1]
    x0, y0, x1, y1 = (page_box[0] * sx, page_box[1] * sy, page_box[2] * sx, page_box[3] * sy)
    band_top = (1 - region.origin.y - region.size.height) * seen_size[1]
    band_height = region.size.height * seen_size[1]
    box = _rect(
        x0 / seen_size[0],
        1 - (y1 - band_top) / band_height,
        (x1 - x0) / seen_size[0],
        (y1 - y0) / band_height,
    )
    candidate = types.SimpleNamespace(string=lambda: text)
    return types.SimpleNamespace(topCandidates_=lambda n: [candidate], boundingBox=lambda: box)


def _read(monkeypatch, script, page_size, turns=0, scanned_size=None):
    fake = _FakeVision(script, page_size)
    fake.install(monkeypatch)
    page = Image.new("L", scanned_size or page_size, 255)
    return fake, _READ_LINES(page, turns, ["en-US"])


def test_vision_reads_a_copy_three_quarters_the_size_and_its_boxes_land_in_full_pixels(
    monkeypatch,
):
    box = (400, 1000, 1600, 1100)
    fake, lines = _read(monkeypatch, [[("alpha", box)], []], (2000, 2800))
    assert fake.received == (1500, 2100)
    [line] = lines
    assert line.box == pytest.approx(box, abs=2)


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_a_turned_page_is_read_scaled_and_its_boxes_come_back_in_the_scanned_frame(
    monkeypatch, turns
):
    scanned = (2000, 2800)
    upright = scanned[::-1] if turns % 2 else scanned
    box = (400, 1000, 1600, 1100)  # on the upright page
    fake, lines = _read(monkeypatch, [[], [("alpha", box)]], upright, turns, scanned)
    assert fake.received == (round(upright[0] * 0.75), round(upright[1] * 0.75))
    [line] = lines
    assert line.box == pytest.approx(ocr_vision._page_box(box, turns, scanned), abs=2)


def test_lines_come_back_in_the_order_vision_gave_them_band_by_band(monkeypatch):
    """Vision reads a two-column page column by column: that order is its reading order."""
    script = [
        [
            ("L1", (100, 100, 900, 160)),
            ("L2", (100, 300, 900, 360)),
            ("R1", (1100, 100, 1900, 160)),
            ("R2", (1100, 300, 1900, 360)),
        ],
        [("L3", (100, 2000, 900, 2060)), ("R3", (1100, 2000, 1900, 2060))],
    ]
    _, lines = _read(monkeypatch, script, (2000, 2800))
    assert [line.text for line in lines] == ["L1", "L2", "R1", "R2", "L3", "R3"]


def test_a_band_that_reads_nothing_costs_the_page_only_what_that_band_alone_read(monkeypatch):
    script = [[("top", (100, 100, 900, 160))], []]
    _, lines = _read(monkeypatch, script, (2000, 2800))
    assert [line.text for line in lines] == ["top"]
    tess = [
        ocr.Line((100, 100, 900, 160), "top", starts_paragraph=True),
        ocr.Line((100, 2000, 900, 2060), "bottom"),
    ]
    assert ocr.splice(tess, lines) == "top\nbottom"


# --- real Vision -----------------------------------------------------------------------

needs_vision = pytest.mark.skipif(
    not ocr_vision.available(), reason="needs macOS and the [ocr] extra"
)
_ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"


@pytest.mark.slow
@needs_vision
@pytest.mark.parametrize("quarter_turns", [0, 1, 3])
def test_real_vision_reads_every_line_once_in_the_frame_of_the_page_as_scanned(quarter_turns):
    """Thirty lines from top to bottom: five of them in both bands."""
    font = ImageFont.truetype(_ARIAL, 44)
    page = Image.new("RGB", (2480, 3508), "white")
    draw = ImageDraw.Draw(page)
    drawn = {}
    for i in range(30):
        text = f"Пункт {i + 1}. Поставка номер {700 + i}"
        draw.text((150, 150 + 105 * i), text, fill="black", font=font)
        drawn[str(700 + i)] = draw.textbbox((150, 150 + 105 * i), text, font=font)
    scanned = page.rotate(90 * quarter_turns, expand=True)
    lines = ocr_vision.read_lines(scanned, (4 - quarter_turns) % 4, ["ru-RU", "en-US"])
    for number, box in drawn.items():
        [line] = [line for line in lines if number in line.text]
        x0, y0, x1, y1 = ocr._upright_box(box, quarter_turns, page.size)
        u0, v0, u1, v1 = line.box
        assert abs((u0 + u1) - (x0 + x1)) / 2 < 25 and abs((v0 + v1) - (y0 + y1)) / 2 < 25


_KNOWN = [
    "Поставщик обязуется передать Покупателю товар",
    "в количестве и ассортименте согласно спецификации",
    "Оплата производится в течение десяти банковских дней",
    "с момента подписания сторонами товарной накладной",
    "Стороны несут ответственность за неисполнение",
    "обязательств по настоящему договору поставки",
]


def _degraded_page() -> Image.Image:
    font = ImageFont.truetype(_ARIAL, 30)
    page = Image.new("L", (2480, 1200), 255)
    draw = ImageDraw.Draw(page)
    for i, line in enumerate(_KNOWN):
        draw.text((150, 150 + 120 * i), line, fill=0, font=font)
    small = page.resize((page.width // 3, page.height // 3), Image.Resampling.BILINEAR)
    return small.resize(page.size, Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(2))


def _words(text: str) -> set[str]:
    return {word.strip(".,").lower() for word in text.split()}


@pytest.mark.slow
@needs_vision
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
def test_real_vision_recovers_words_tesseract_lost_on_a_degraded_page(tmp_path):
    path = tmp_path / "degraded.png"
    _degraded_page().save(path, dpi=(300, 300))
    cfg = load_config({}, cwd=tmp_path)
    known = _words(" ".join(_KNOWN))

    tesseract = ocr._recognize(Image.open(path), cfg, path, dpi=300)
    assert tesseract.confidence < ocr.SECOND_READER_BELOW
    lost = known - _words(tesseract.text)
    assert lost, "the page is not degraded enough to tell the readers apart"

    text, engine = ocr.ocr_image(path, cfg)
    assert engine == ocr.ENGINE_TESSERACT_VISION
    assert lost <= _words(text)


def _rss_mb() -> float:
    import resource

    # ru_maxrss is bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


@pytest.mark.slow
@needs_vision
@pytest.mark.skipif(sys.platform != "darwin", reason="the raw-CGImage leak is macOS's")
def test_real_vision_over_a_hundred_pages_does_not_grow_the_process():
    """Vision kept ~16-20 MB per page read from a raw CGImage; read from PNG bytes it is freed."""
    font = ImageFont.truetype(_ARIAL, 44)
    page = Image.new("L", (2480, 3508), 255)
    draw = ImageDraw.Draw(page)
    for i in range(30):
        draw.text((150, 150 + 105 * i), f"Line {i + 1} delivery number {700 + i}", font=font)
    ocr_vision.read_lines(page, 0, ["en-US"])  # the model loads once, outside the measure
    before = _rss_mb()
    for _ in range(100):
        ocr_vision.read_lines(page, 0, ["en-US"])
    assert _rss_mb() - before < 100
