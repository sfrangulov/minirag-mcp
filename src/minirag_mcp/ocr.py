"""Optional OCR engine (the [ocr] extra): scanned PDFs and images -> text.

The only module that touches rapidocr / pypdfium2 / rapid_orientation, and all
three are imported lazily inside functions: the core package must import and
run without the extra installed.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

if TYPE_CHECKING:
    from minirag_mcp.config import Config

logger = logging.getLogger(__name__)

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


_engines: dict[tuple[str, str], object] = {}
# pypdfium2 scale for 300 dpi; recognition crops keep this resolution because
# the engine params below raise its silent 2000 px input cap.
_RENDER_SCALE = 300 / 72


def assemble_lines(boxes: list, txts: list) -> str:
    """Reading order for flat OCR line boxes: rows by y-overlap, then x.

    RapidOCR returns detection quads in image order, not reading order. Two
    boxes share a row when their vertical centers sit within half the median
    box height of each other — a threshold that tolerates slight skew without
    merging adjacent lines.
    """
    if not txts:
        return ""
    items = []
    for box, txt in zip(boxes, txts, strict=True):
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append(((min(ys) + max(ys)) / 2, min(xs), max(ys) - min(ys), txt))
    items.sort()
    heights = sorted(h for _, _, h, _ in items)
    row_tol = max(heights[len(heights) // 2] / 2, 1.0)
    rows: list[list[tuple[float, str]]] = []
    row_y = None
    for cy, x, _, txt in items:
        if row_y is None or cy - row_y > row_tol:
            rows.append([])
            row_y = cy
        rows[-1].append((x, txt))
    return "\n".join(" ".join(t for _, t in sorted(row)) for row in rows)


def _require_available() -> None:
    if not available():
        raise OcrError(f"this file needs OCR, which is not installed; {INSTALL_HINT}")


def _import_error_as_ocr_error(e: ImportError) -> OcrError:
    """find_spec (available()) proves a module is locatable, not importable —
    a broken install (e.g. ABI mismatch) must surface as loud OcrError, not a
    raw ImportError.
    """
    return OcrError(f"OCR dependencies are installed but failed to import: {e}; {INSTALL_HINT}")


def _engine(config: Config):
    """One cached RapidOCR per (lang, cache_dir); models download on first use."""
    key = (config.ocr_lang, str(config.cache_dir))
    engine = _engines.get(key)
    if engine is None:
        try:
            from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        except ImportError as e:
            raise _import_error_as_ocr_error(e) from e

        try:
            lang = LangRec(config.ocr_lang)
        except ValueError as e:
            valid = ", ".join(m.value for m in LangRec)
            raise OcrError(
                f"RAG_OCR_LANG={config.ocr_lang!r} is not a rapidocr language; one of: {valid}"
            ) from e
        try:
            engine = RapidOCR(
                params={
                    "Global.model_root_dir": str(config.cache_dir / "ocr"),
                    # Raised from the default 2000: recognition crops are cut
                    # from this image, and 300-dpi pages must not be resampled.
                    "Global.max_side_len": 4000,
                    # Detection alone runs on a bounded copy — without this it
                    # would process the full 300-dpi bitmap at ~3x the cost.
                    "Det.limit_type": "max",
                    "Det.limit_side_len": 1600,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.lang_type": lang,
                    "Rec.model_type": ModelType.MOBILE,
                }
            )
        except Exception as e:  # model download / init failure must be loud
            raise OcrError(f"OCR engine failed to initialize: {e}") from e
        _engines[key] = engine
    return engine


def _orient(img):
    """Rotate a sideways/upside-down page upright, when the classifier says so.

    rapid-orientation returns only a label ('0'/'90'/'180'/'270'), no
    confidence, so the only gate is a non-zero label. Optional: skipped
    silently when the package is missing (it ships with the extra).
    """
    if importlib.util.find_spec("rapid_orientation") is None:
        return img
    try:
        import numpy as np
    except ImportError as e:
        raise _import_error_as_ocr_error(e) from e

    engine = _orientation_engine()
    label, _elapse = engine(img)
    quarter_turns = int(label) // 90
    if quarter_turns:
        # np.rot90 is counterclockwise; the slow rotation test pins this sign.
        img = np.rot90(img, k=quarter_turns).copy()
    return img


_orientation: list[object] = []


def _orientation_engine():
    if not _orientation:
        try:
            from rapid_orientation import RapidOrientation
        except ImportError as e:
            raise _import_error_as_ocr_error(e) from e

        _orientation.append(RapidOrientation())
    return _orientation[0]


def _recognize(img, config: Config) -> str:
    result = _engine(config)(img)
    if result.txts is None:
        return ""
    return assemble_lines(list(result.boxes), list(result.txts))


def ocr_pdf(path: Path, config: Config, pages: Sequence[int]) -> dict[int, str]:
    """OCR the given zero-based pages of a PDF, rendered at 300 dpi."""
    _require_available()
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise _import_error_as_ocr_error(e) from e

    out: dict[int, str] = {}
    doc = pdfium.PdfDocument(str(path))
    try:
        for index in pages:
            bitmap = doc[index].render(scale=_RENDER_SCALE)
            img = _orient(bitmap.to_numpy())
            out[index] = _recognize(img, config)
            logger.info("ocr: %s page %d -> %d chars", path, index, len(out[index]))
    finally:
        doc.close()
    return out


def ocr_image(path: Path, config: Config) -> str:
    _require_available()
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise _import_error_as_ocr_error(e) from e

    # np.fromfile + imdecodemulti instead of cv2.imread / cv2.imreadmulti: the imread
    # family cannot open non-ASCII paths on Windows, and this corpus's filenames are
    # Russian. imdecodemulti over imdecode because imdecode returns the first frame
    # only, and multi-page TIFF is ordinary scanner and fax output here — a 20-page
    # scan would index as page 1 and report success.
    data = np.fromfile(str(path), dtype=np.uint8)
    ok, frames = cv2.imdecodemulti(data, cv2.IMREAD_COLOR)
    if not ok or not frames:
        raise OcrError(f"could not decode image {path}")
    logger.info("ocr: %s -> %d frame(s)", path, len(frames))
    return "\n\n".join(text for f in frames if (text := _recognize(_orient(f), config)))
