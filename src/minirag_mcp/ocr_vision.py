"""Apple Vision, the second reader of the pages Tesseract is unsure of (macOS, [ocr] extra).

The only module that touches pyobjc, and every import of it is lazy inside a function:
the package must import and run on every platform without it.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from PIL import Image

    from minirag_mcp.ocr import Box, Line

_MODULES = ("objc", "Quartz", "Vision")
_LANGUAGES = {"rus": "ru-RU", "eng": "en-US"}  # ru-RU exists at the accurate level only
# Read whole, a page loses its small text to Vision's own scaling, so it is read as two
# halves, each reaching 15% of a half past the midline: a line cut by one edge is whole
# in the other band.
_BANDS = ((0.0, 0.575), (0.425, 1.0))
# Vision reads the page at this share of its pixels (225 dpi for a 300-dpi render). Measured
# against full size on real scans: fewer bands lose lines and the character error rate falls
# no further; below it small text starts to go. Tesseract keeps the full-resolution page.
_VISION_SCALE = 0.75


class _Seen(NamedTuple):
    """A line one band read: the band's top and bottom and the line's box, in pixels of
    the upright page."""

    band: tuple[float, float]
    box: Box
    text: str


def supported() -> bool:
    """Whether this platform has Apple Vision at all."""
    return sys.platform == "darwin"


def available() -> bool:
    return supported() and all(importlib.util.find_spec(mod) is not None for mod in _MODULES)


def languages(ocr_lang: str) -> list[str]:
    """`RAG_OCR_LANG`'s Tesseract languages, as Vision names them."""
    return [_LANGUAGES[name] for name in ocr_lang.split("+")]


def read_lines(image: Image.Image, turns: int, languages: list[str]) -> list[Line]:
    """The lines Vision reads on `image` turned upright by `turns` counter-clockwise
    quarter-turns, boxed in the frame of `image` as given: the frame Tesseract's are in.

    Every call makes its own handler and requests, so calls share nothing across sync's
    worker processes or the server's threads. The first call in a process loads the model.
    """
    import objc
    import Quartz
    import Vision
    from PIL import Image

    from minirag_mcp.ocr import Line

    upright = image.rotate(90 * turns, expand=True) if turns else image
    full_width, full_height = upright.size
    upright = upright.resize(
        (max(1, round(full_width * _VISION_SCALE)), max(1, round(full_height * _VISION_SCALE))),
        Image.Resampling.LANCZOS,
    )
    width, height = upright.size
    across, down = full_width / width, full_height / height
    # A PNG, not a CGImage over the raw pixels: Vision keeps a copy of the region it read
    # out of such a CGImage, about 20 MB per band of an A4 page at 300 dpi, and never
    # frees it (measured with pyobjc 13). One it decodes from a PNG is freed.
    encoded = io.BytesIO()
    upright.save(encoded, format="PNG", compress_level=1)
    bands = _bands(height)
    seen: list[_Seen] = []
    with objc.autorelease_pool():
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
            encoded.getvalue(), None
        )
        requests = [_request(Vision, Quartz, band, height, languages) for band in bands]
        ok, error = handler.performRequests_error_(requests, None)
        if not ok:
            raise RuntimeError(f"Vision could not read the page: {error}")
        for band, request in zip(bands, requests, strict=True):
            for observation in request.results() or ():
                candidates = observation.topCandidates_(1)
                text = str(candidates[0].string()).strip() if candidates else ""
                if not text:
                    continue
                found = observation.boundingBox()
                normalised = (found.origin.x, found.origin.y, found.size.width, found.size.height)
                x0, y0, x1, y1 = _pixel_box(normalised, band, width)
                seen.append(
                    _Seen(
                        (band[0] * down, band[1] * down),
                        (x0 * across, y0 * down, x1 * across, y1 * down),
                        text,
                    )
                )
    return [Line(_page_box(line.box, turns, image.size), line.text) for line in _deduplicated(seen)]


def _request(Vision, Quartz, band: tuple[float, float], height: int, languages: list[str]):
    top, bottom = band
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setRecognitionLanguages_(languages)
    request.setUsesLanguageCorrection_(True)
    # Normalised to the page and measured from its bottom left, as Vision measures.
    region = Quartz.CGRectMake(0, 1 - bottom / height, 1, (bottom - top) / height)
    request.setRegionOfInterest_(region)
    return request


def _bands(height: float) -> list[tuple[float, float]]:
    """The top and bottom of each band of a page `height` pixels high."""
    return [(top * height, bottom * height) for top, bottom in _BANDS]


def _pixel_box(
    normalised: tuple[float, float, float, float], band: tuple[float, float], width: float
) -> Box:
    """A box Vision gives as x, y, width and height, normalised to the band it read and
    measured from the band's bottom left, in pixels of the page from its top left."""
    x, y, w, h = normalised
    top, bottom = band
    span = bottom - top
    upper = top + (1 - y - h) * span
    return (x * width, upper, (x + w) * width, upper + h * span)


def _deduplicated(seen: list[_Seen]) -> list[_Seen]:
    """`seen` without the second reading of the lines both bands read.

    A line is a kept line's reading by the other band when their boxes share more than
    half the smaller one; the reading further from its band's edge is the one kept. Lines
    of one band are never compared: Vision's boxes are taller than the glyphs, so the
    neighbouring lines of one band overlap too, and dropping them deleted whole lines.
    """

    def depth(line: _Seen) -> float:
        middle = (line.box[1] + line.box[3]) / 2
        return min(middle - line.band[0], line.band[1] - middle)

    kept: list[int] = []
    for i in sorted(range(len(seen)), key=lambda i: depth(seen[i]), reverse=True):
        line = seen[i]
        if not any(seen[k].band != line.band and _shared(line.box, seen[k].box) for k in kept):
            kept.append(i)
    return [seen[i] for i in sorted(kept)]


def _shared(a: Box, b: Box) -> bool:
    across = min(a[2], b[2]) - max(a[0], b[0])
    down = min(a[3], b[3]) - max(a[1], b[1])
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return across > 0 and down > 0 and across * down > smaller / 2


def _page_box(box: Box, turns: int, size: tuple[int, int]) -> Box:
    """A box on the page turned upright by `turns` counter-clockwise quarter-turns, back
    on the page of `size` as it was given."""
    width, height = size
    u0, v0, u1, v1 = box
    if turns == 1:  # (u, v) -> (width - v, u)
        return (width - v1, u0, width - v0, u1)
    if turns == 2:  # (u, v) -> (width - u, height - v)
        return (width - u1, height - v1, width - u0, height - v0)
    if turns == 3:  # (u, v) -> (v, height - u)
        return (v0, height - u1, v1, height - u0)
    return box
