"""Optional OCR engine (the [ocr] extra): scanned PDFs and images -> text.

The only module that touches tesserocr / pypdfium2, and both are imported lazily
inside functions: the core package must import and run without the extra installed.
Apple Vision, the second reader on macOS, is `ocr_vision`, under the same rule.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from minirag_mcp import ocr_vision

if TYPE_CHECKING:
    from PIL import Image
    from tesserocr import PyTessBaseAPI

    from minirag_mcp.config import Config

logger = logging.getLogger(__name__)

ENGINE_TESSERACT = "tesseract"
# A document Apple Vision re-read at least one page of.
ENGINE_TESSERACT_VISION = "tesseract+vision"
DPI = 300
# A page Tesseract reads with a lower mean word confidence is read again by Apple Vision
# where it exists. RAG_OCR_SECOND_READER_BELOW overrides it, in percent.
SECOND_READER_BELOW = 0.85
# A Vision line that shares a line with several Tesseract lines reads each one it covers
# this share of in height. Over the two halves of a line Tesseract split it covers about
# 90% of each; on a page tilted about 2 degrees a full-width line's box at 300 dpi is over
# two line pitches tall, and covers about half of each neighbour's.
_SPLICE_COVER = 0.75

Box = tuple[float, float, float, float]  # left, top, right, bottom; pixels from the top left

INSTALL_HINT = (
    "install the OCR extra: uv tool install 'minirag-mcp[ocr]' (or pip install 'minirag-mcp[ocr]')"
)


class OcrError(Exception):
    """OCR was needed and failed or is unavailable. Always loud, never a no-op."""


@dataclass(frozen=True)
class Line:
    """One line of a page as a reader saw it. `starts_paragraph` marks where Tesseract's
    own text leaves a blank line: before every paragraph, not only every block."""

    box: Box
    text: str
    starts_paragraph: bool = False


@dataclass(frozen=True)
class Recognized:
    """A page as Tesseract read it, its boxes in the frame of the image it was given."""

    text: str
    confidence: float  # mean word confidence, 0 to 1; 0 when no word was found
    lines: tuple[Line, ...]
    turns: int  # counter-clockwise quarter-turns that make the page upright


def available() -> bool:
    return all(importlib.util.find_spec(mod) is not None for mod in ("tesserocr", "pypdfium2"))


def pdf_page_texts(path: Path) -> list[str]:
    """Text-layer text per page, one pdfminer pass. "" for a page with none."""
    texts: list[str] = []
    for layout in extract_pages(str(path)):
        parts = [el.get_text() for el in layout if isinstance(el, LTTextContainer)]
        texts.append("".join(parts))
    return texts


# A handle holds its language models (tens of MB each); one per language set and data
# directory, each with the lock its callers share.
_engines: dict[tuple[str, str], tuple[PyTessBaseAPI, threading.Lock]] = {}
_engines_lock = threading.Lock()


def _require_available() -> None:
    if not available():
        raise OcrError(f"this file needs OCR, which is not installed; {INSTALL_HINT}")


def _import_error_as_ocr_error(e: ImportError) -> OcrError:
    """find_spec (available()) proves a module is locatable, not importable —
    a broken install (e.g. ABI mismatch) must surface as loud OcrError, not a
    raw ImportError.
    """
    return OcrError(
        "an OCR dependency is present but failed to import — a broken or partial "
        f"install: {e}; repair it with a reinstall — {INSTALL_HINT}"
    )


def _engine(config: Config) -> tuple[PyTessBaseAPI, threading.Lock]:
    """The cached Tesseract handle for `config.ocr_lang`; its data downloads on first use."""
    key = (config.ocr_lang, str(tessdata_dir(config.cache_dir)))
    with _engines_lock:
        if key not in _engines:
            try:
                import tesserocr
            except ImportError as e:
                raise _import_error_as_ocr_error(e) from e

            data = ensure_tessdata(config.cache_dir, config.ocr_lang)
            try:
                api = tesserocr.PyTessBaseAPI(
                    path=str(data), lang=config.ocr_lang, psm=tesserocr.PSM.AUTO_OSD
                )
            except Exception as e:
                raise OcrError(f"OCR engine failed to initialize: {e}") from e
            _engines[key] = (api, threading.Lock())
        return _engines[key]


def _recognize(image: Image.Image, config: Config, path: Path, *, dpi: int | None) -> Recognized:
    """The page in Tesseract's own reading order: block by block, so the columns of a
    two-column page stay apart.

    With `dpi` None no resolution is set here. tesserocr hands a frame without a format
    to Tesseract as BMP, and Pillow writes 96 dpi into every BMP it saves.
    """
    api, lock = _engine(config)
    # The handle is not thread-safe, and SetImage resets the source resolution and the
    # last page's results, so everything read off it happens in one turn. Never call
    # DetectOrientationScript on this handle: on one initialised without `osd` as its
    # language it aborts the process (std::out_of_range, observed 2026-11-19); AUTO_OSD
    # orients the page instead.
    with lock:
        try:
            api.SetImage(image)
            if dpi is not None:
                api.SetSourceResolution(dpi)
            text = api.GetUTF8Text()
            lines, orientations = _layout(api)
            confidence = api.MeanTextConf() / 100 if lines else 0.0
        except Exception as e:
            raise OcrError(f"OCR engine failed on {path}: {e}") from e
    turns = Counter(orientations).most_common(1)[0][0] if orientations else 0
    return Recognized(text.strip(), confidence, tuple(lines), turns)


def _layout(api: PyTessBaseAPI) -> tuple[list[Line], list[int]]:
    """Tesseract's lines, and the orientation of each block that holds one.

    An empty page still has an iterator, and it raises for its text ("No text returned"),
    as it may for a line with no text in it. Such a line hands the start of its block and
    paragraph on to the next line that has text.
    """
    from tesserocr import RIL

    lines: list[Line] = []
    orientations: list[int] = []
    iterator = api.GetIterator()
    if iterator is None:
        return lines, orientations
    block = paragraph = False
    while True:
        block = block or bool(iterator.IsAtBeginningOf(RIL.BLOCK))
        paragraph = paragraph or bool(iterator.IsAtBeginningOf(RIL.PARA))
        box = iterator.BoundingBox(RIL.TEXTLINE)
        try:
            text = iterator.GetUTF8Text(RIL.TEXTLINE).strip()
        except RuntimeError:
            text = ""
        if box is not None and text:
            if block:
                orientations.append(int(iterator.Orientation()[0]))
            lines.append(Line(tuple(box), text, paragraph))
            block = paragraph = False
        if not iterator.Next(RIL.TEXTLINE):
            return lines, orientations


def _read(
    image: Image.Image, config: Config, path: Path, *, dpi: int | None, page: int
) -> tuple[str, bool]:
    """The page's text, and whether Apple Vision's reading went into it.

    A page Tesseract is unsure of is read again by Vision where Vision exists, and the
    two readings are spliced. A page Vision fails on keeps Tesseract's text: the file
    never fails because of the second reader, and the warning is how that shows.
    """
    recognized = _recognize(image, config, path, dpi=dpi)
    if recognized.confidence >= config.ocr_second_reader_below / 100 or not ocr_vision.available():
        return recognized.text, False
    try:
        languages = ocr_vision.languages(config.ocr_lang)
        seen = ocr_vision.read_lines(image, recognized.turns, languages)
    except Exception as e:
        logger.warning(
            "ocr: Apple Vision failed on page %d of %s, which keeps Tesseract's text: %s",
            page,
            path,
            e,
        )
        return recognized.text, False
    if not seen:
        return recognized.text, False

    def upright(line: Line) -> Line:
        return replace(line, box=_upright_box(line.box, recognized.turns, image.size))

    return splice([upright(line) for line in recognized.lines], [upright(s) for s in seen]), True


def splice(tess_lines: Sequence[Line], vision_lines: Sequence[Line]) -> str:
    """Tesseract's lines in Tesseract's order, each as Vision read it.

    A Tesseract line is replaced by the Vision lines that read it and are not used yet,
    left to right, so a Vision line across two Tesseract lines comes out once; one no
    Vision line reads is kept. A Vision line that reads none but overlaps some comes out
    after the one of them it covers most of; those that overlap none, text Tesseract
    never found, follow the page as a paragraph of their own. Either way, several in one
    place come out in the order Vision gave them. Both readings are boxed on the upright
    page: left to right means nothing on a sideways one.
    """
    reads = [_read_by(seen.box, tess_lines) for seen in vision_lines]
    beside: dict[int, list[str]] = {}
    rest: list[str] = []
    for seen, lines in zip(vision_lines, reads, strict=True):
        if lines:
            continue
        anchor = _beside(seen.box, tess_lines)
        if anchor is None:
            rest.append(seen.text)
        else:
            beside.setdefault(anchor, []).append(seen.text)
    used: set[int] = set()
    out: list[str] = []
    paragraph = False
    for t, line in enumerate(tess_lines):
        readings = [i for i, lines in enumerate(reads) if t in lines]
        fresh = sorted(set(readings) - used, key=lambda i: vision_lines[i].box[0])
        texts = [vision_lines[i].text for i in fresh] if readings else [line.text]
        texts += beside.get(t, [])
        used.update(readings)
        paragraph = paragraph or line.starts_paragraph
        if texts:
            if paragraph and out:
                out.append("")
            paragraph = False
            out.extend(texts)
    if rest and out:
        out.append("")
    out.extend(rest)
    return "\n".join(out)


def _read_by(box: Box, tess_lines: Sequence[Line]) -> set[int]:
    """The Tesseract lines a Vision line boxed `box` reads: the only one it shares a line
    with, or of several, each it covers `_SPLICE_COVER` of. On a tilted page a box reaches
    over its neighbours, and a line Vision missed must not go out as a neighbour's reading."""
    shared = [t for t, line in enumerate(tess_lines) if _same_line(line.box, box)]
    if len(shared) == 1:
        return set(shared)
    return {t for t in shared if _covered(tess_lines[t].box, box) >= _SPLICE_COVER}


def _beside(box: Box, tess_lines: Sequence[Line]) -> int | None:
    """The Tesseract line a Vision line boxed `box` that reads none goes out after: of
    those it overlaps, the one it covers most of, the first of equals; None if it overlaps
    none. A table row Tesseract read as two cell lines and Vision as one line across both
    covers under `_SPLICE_COVER` of each, and belongs beside that row."""
    overlapped = [t for t, line in enumerate(tess_lines) if _overlaps(line.box, box)]
    return max(overlapped, key=lambda t: _covered(tess_lines[t].box, box), default=None)


def _covered(a: Box, b: Box) -> float:
    """The share of `a`'s height `b` covers; `a` is one `b` overlaps, so it has a height."""
    return (min(a[3], b[3]) - max(a[1], b[1])) / (a[3] - a[1])


def _overlaps(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _same_line(a: Box, b: Box) -> bool:
    """Whether two boxes hold one line: they meet across, and down they share more than
    half the shorter one's height. Vision's boxes are taller than the glyphs, so a
    neighbour's box reaching a little into a line is not that line."""
    across = min(a[2], b[2]) - max(a[0], b[0])
    down = min(a[3], b[3]) - max(a[1], b[1])
    return across > 0 and down > min(a[3] - a[1], b[3] - b[1]) / 2


def _upright_box(box: Box, turns: int, size: tuple[int, int]) -> Box:
    """`box` on a page of `size` once the page is turned `turns` quarter-turns
    counter-clockwise, as `Image.rotate(90 * turns, expand=True)` turns it."""
    width, height = size
    x0, y0, x1, y1 = box
    if turns == 1:
        return (y0, width - x1, y1, width - x0)
    if turns == 2:
        return (width - x1, height - y1, width - x0, height - y0)
    if turns == 3:
        return (height - y1, x0, height - y0, x1)
    return box


def ocr_pdf(path: Path, config: Config, pages: Sequence[int]) -> tuple[dict[int, str], str]:
    """OCR the given zero-based pages of a PDF, rendered at `DPI`: the text of each, and
    the engine that read them."""
    _require_available()
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise _import_error_as_ocr_error(e) from e

    out: dict[int, str] = {}
    spliced = False
    try:
        doc = pdfium.PdfDocument(str(path))
    except Exception as e:
        raise OcrError(f"could not open {path} for OCR: {e}") from e
    try:
        for index in pages:
            try:
                page = doc[index]
            except Exception as e:
                # The pages to OCR are chosen from pdfminer's page list; the renderer
                # opening the same file is a second parser and can count differently.
                raise OcrError(f"{path} has no page {index} to render for OCR: {e}") from e
            try:
                image = page.render(scale=DPI / 72).to_pil()
            except Exception as e:
                raise OcrError(f"could not render page {index} of {path} for OCR: {e}") from e
            out[index], second = _read(image, config, path, dpi=DPI, page=index)
            spliced = spliced or second
            logger.info("ocr: %s page %d -> %d chars", path, index, len(out[index]))
    finally:
        doc.close()
    return out, ENGINE_TESSERACT_VISION if spliced else ENGINE_TESSERACT


def ocr_image(path: Path, config: Config) -> tuple[str, str]:
    """The text of every frame of an image file, and the engine that read them."""
    _require_available()
    texts = []
    frames = 0
    spliced = False
    for frame, dpi in _decoded_frames(path):
        text, second = _read(frame, config, path, dpi=dpi, page=frames)
        frames += 1
        spliced = spliced or second
        if text:
            texts.append(text)
    logger.info("ocr: %s -> %d frame(s)", path, frames)
    return "\n\n".join(texts), ENGINE_TESSERACT_VISION if spliced else ENGINE_TESSERACT


def _decoded_frames(path: Path) -> Iterator[tuple[Image.Image, int | None]]:
    """Every frame of an image file, decoded, with the dpi it records.

    Every frame, not the first: multi-page TIFF is ordinary scanner and fax output, and
    a 20-page scan would otherwise index as page 1 and report success.
    """
    try:
        from PIL import Image, ImageSequence
    except ImportError as e:
        raise _import_error_as_ocr_error(e) from e

    try:
        with Image.open(path) as image:
            for frame in ImageSequence.Iterator(image):
                yield _on_white(frame), _recorded_dpi(frame)
    except Exception as e:
        # Opening reads only the header; the pixels decode in `_on_white`. Both, and the
        # read itself (missing file, no permission), must leave this module as OcrError.
        raise OcrError(f"could not read or decode image {path}: {e}") from e


def _on_white(frame: Image.Image) -> Image.Image:
    from PIL import Image

    frame = _eight_bit(frame)
    # A copy, even of a frame that needs no conversion: tesserocr hands Tesseract an
    # image re-encoded in its own format, which recompresses a JPEG or WebP frame
    # lossily. A copy has no format and goes as a lossless BMP. `transparency` is a
    # colour key, PNG's alternative to an alpha channel.
    if frame.mode in ("1", "L", "RGB") and "transparency" not in frame.info:
        return frame.copy()
    # Dropping the alpha channel of black text on a transparent background (an exported
    # diagram, a screenshot) leaves a black page.
    page = Image.new("RGBA", frame.size, (255, 255, 255, 255))
    page.alpha_composite(frame.convert("RGBA"))
    return page.convert("RGB")


def _eight_bit(frame: Image.Image) -> Image.Image:
    """A 16-bit, 32-bit or floating-point grayscale frame as 8-bit `L`; others unchanged.

    Pillow's own conversion clips every value above 255 to white, so a 16-bit scan would
    arrive as an empty page. The scale is read from the frame's brightest value, because
    `I` and `F` do not record their bit depth: up to 1 is a fraction of white, up to 255
    is 8-bit already, anything brighter is 16-bit.
    """
    if not frame.mode.startswith(("I", "F")):
        return frame
    # Through `F`, which each of these modes converts to: Pillow has no extrema for I;16B.
    values = frame.convert("F")
    _, brightest = values.getextrema()
    scale = 255 if brightest <= 1 else 1 if brightest <= 255 else 1 / 257
    return values.point(lambda v: v * scale).convert("L")


def _recorded_dpi(frame: Image.Image) -> int | None:
    """The frame's own dpi, or None below Tesseract's credible minimum of 70."""
    try:
        dpi = round(float(frame.info["dpi"][0]))
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None
    return dpi if dpi >= 70 else None


TESSDATA_COMMIT = "87416418657359cb625c412a48b6e1d6d41c29bd"  # tessdata_fast main, 2024-08-01
TESSDATA_SHA256 = {
    "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
    "osd": "9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff",
    "rus": "e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225",
}
_TESSDATA_URL = (
    "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{commit}/{name}.traineddata"
)
DEFAULT_LANGUAGES = "rus+eng"
LANGUAGE_ALIASES = {"eslav": "rus+eng", "en": "eng"}


def tesseract_languages(value: str) -> str:
    text = value.strip().lower()
    if text in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[text]
    parts = [part.strip() for part in text.split("+")]
    known = [name for name in TESSDATA_SHA256 if name != "osd"]
    if any(part not in known for part in parts) or len(set(parts)) != len(parts):
        raise ValueError(
            f"{value!r} is not a language set OCR can read: use rus, eng or both joined "
            "by '+' (rus+eng), or the 1.x names eslav, en"
        )
    return "+".join(parts)


def tessdata_dir(cache_dir: Path) -> Path:
    return cache_dir / "ocr" / f"tessdata_fast-{TESSDATA_COMMIT[:12]}"


# sha256 of each cached traineddata file this process has verified, keyed by what would
# change if the file did, so repeated OCR jobs do not rehash ~10 MB each time.
_tessdata_digests: dict[tuple[str, int, int], str] = {}


def _file_sha256(path: Path) -> str:
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _tessdata_digests:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        _tessdata_digests[key] = digest.hexdigest()
    return _tessdata_digests[key]


def _download_tessdata(name: str, target: Path) -> None:
    url = _TESSDATA_URL.format(commit=TESSDATA_COMMIT, name=name)
    expected = TESSDATA_SHA256[name]
    # Several sync workers may fetch at once; the replace is what keeps any of them
    # from reading a partial file.
    tmp = tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{name}.", delete=False)
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            digest = hashlib.sha256()
            try:
                with requests.get(url, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    for block in resp.iter_content(1 << 20):
                        digest.update(block)
                        tmp.write(block)
            except requests.RequestException as e:
                raise OcrError(
                    f"could not download {name}.traineddata from {url}: {e}; "
                    "OCR needs it once, then works offline"
                ) from e
        if digest.hexdigest() != expected:
            raise OcrError(
                f"{url} has sha256 {digest.hexdigest()}, expected {expected}; "
                "the download was discarded"
            )
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def ensure_tessdata(cache_dir: Path, languages: str) -> Path:
    directory = tessdata_dir(cache_dir)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for name in [*languages.split("+"), "osd"]:
            path = directory / f"{name}.traineddata"
            if path.exists():
                actual = _file_sha256(path)
                if actual != TESSDATA_SHA256[name]:
                    raise OcrError(
                        f"{path} has sha256 {actual}, expected {TESSDATA_SHA256[name]}; "
                        "delete it to download it again"
                    )
            else:
                _download_tessdata(name, path)
    except OSError as e:
        raise OcrError(f"could not prepare Tesseract data in {directory}: {e}") from e
    return directory
