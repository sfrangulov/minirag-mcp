"""Optional OCR engine (the [ocr] extra): scanned PDFs and images -> text.

The only module that touches rapidocr / pypdfium2 / rapid_orientation, and all
three are imported lazily inside functions: the core package must import and
run without the extra installed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import requests
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


def assemble_lines(boxes: list, txts: list, source: Path | str) -> str:
    """Reading order for flat OCR line boxes: rows by y-overlap, then x.

    RapidOCR returns detection quads in image order, not reading order. Two
    boxes share a row when their vertical centers sit within half the median
    box height of each other — a threshold that tolerates slight skew without
    merging adjacent lines.

    `source` is carried only to name the file if the engine breaks its own
    contract and hands back one list longer than the other.
    """
    if not txts:
        return ""
    try:
        pairs = list(zip(boxes, txts, strict=True))
    except ValueError as e:
        raise OcrError(
            f"OCR of {source} returned {len(boxes)} box(es) for {len(txts)} text(s)"
        ) from e
    items = []
    for box, txt in pairs:
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
    return OcrError(
        "an OCR dependency is present but failed to import — a broken or partial "
        f"install: {e}; repair it with a reinstall — {INSTALL_HINT}"
    )


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


def _orient(img, path: Path):
    """Rotate a sideways/upside-down page upright, when the classifier says so.

    rapid-orientation returns only a label ('0'/'90'/'180'/'270'), no
    confidence, so the only gate is a non-zero label. Optional: skipped
    silently when the package is missing (it ships with the extra).

    `path` is carried only to name the file in a failure.
    """
    if importlib.util.find_spec("rapid_orientation") is None:
        return img
    try:
        import numpy as np
    except ImportError as e:
        raise _import_error_as_ocr_error(e) from e

    engine = _orientation_engine()
    try:
        label, _elapse = engine(img)
    except Exception as e:
        raise OcrError(f"orientation detection failed on {path}: {e}") from e
    try:
        quarter_turns = int(label) // 90
    except (TypeError, ValueError) as e:
        raise OcrError(
            f"the orientation classifier returned {label!r}, not an angle, for {path}"
        ) from e
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

        try:
            _orientation.append(RapidOrientation())
        except Exception as e:  # model download / init failure must be loud
            raise OcrError(
                f"the OCR orientation model failed to initialize: {e}; {INSTALL_HINT}"
            ) from e
    return _orientation[0]


def _recognize(img, config: Config, path: Path) -> str:
    engine = _engine(config)
    try:
        result = engine(img)
    except Exception as e:
        raise OcrError(f"OCR engine failed on {path}: {e}") from e
    if result.txts is None:
        return ""
    return assemble_lines(list(result.boxes), list(result.txts), path)


def ocr_pdf(path: Path, config: Config, pages: Sequence[int]) -> dict[int, str]:
    """OCR the given zero-based pages of a PDF, rendered at 300 dpi."""
    _require_available()
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise _import_error_as_ocr_error(e) from e

    out: dict[int, str] = {}
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
                bitmap = page.render(scale=_RENDER_SCALE).to_numpy()
            except Exception as e:
                raise OcrError(f"could not render page {index} of {path} for OCR: {e}") from e
            out[index] = _recognize(_orient(bitmap, path), config, path)
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
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        ok, frames = cv2.imdecodemulti(data, cv2.IMREAD_COLOR)
    except Exception as e:
        # Reading and decoding share one guard because both must leave this module as
        # OcrError; the message covers both because an OSError from the read (missing
        # file, no permission) is not a statement about the image's encoding. cv2
        # reports a corrupt file either way: `ok=False` for some, its own exception
        # type for others.
        raise OcrError(f"could not read or decode image {path}: {e}") from e
    if not ok or not frames:
        raise OcrError(f"could not decode image {path}")
    logger.info("ocr: %s -> %d frame(s)", path, len(frames))
    return "\n\n".join(text for f in frames if (text := _recognize(_orient(f, path), config, path)))


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
