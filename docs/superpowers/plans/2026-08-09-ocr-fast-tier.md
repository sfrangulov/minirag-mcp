# OCR Fast Tier (`[ocr]`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scanned PDFs and standalone image files become searchable through an
optional RapidOCR-based extra, with loud failures where OCR is missing.

**Architecture:** A new `ocr` module owns everything raster: per-page PDF text
counts (pdfminer), page rendering (pypdfium2), orientation fix
(rapid-orientation), recognition (RapidOCR), reading-order assembly. The
parser routes empty/near-empty PDF pages and image files through it; the
store records which engine produced a chunk's text. Everything else in the
pipeline (chunk → embed → store) is untouched.

**Tech Stack:** rapidocr (onnxruntime), pypdfium2, rapid-orientation,
pdfminer.six (already present via markitdown), LanceDB (existing store).

**Beads issue:** minirag-mcp-286. Spec:
`docs/superpowers/specs/2026-08-09-ocr-design.md`.

## Global Constraints

- Python >= 3.11; everything runs through `uv` (`uv run pytest -q`,
  `uv run ruff check src tests`, `uv run ruff format src tests`).
- The fast test suite (`uv run pytest -q`) MUST stay green **without** the
  `ocr` extra installed — OCR tests either monkeypatch/fake the engine or
  carry `pytest.importorskip`. Real-engine tests are `@pytest.mark.slow`.
- Project trap rule: every new test for a behavior change must be run
  against the unfixed code first and observed RED. Steps below encode this.
- English everywhere: code, comments, commit messages.
- Comments state constraints code cannot show — never narrate.
- No corporate documents or their fragments in the repo. Test fixtures are
  synthetic, generated in-test (no binary fixtures committed).
- New env vars: `RAG_OCR_LANG` (default `eslav`),
  `RAG_OCR_MIN_CHARS_PER_PAGE` (default `25`, `0` disables the PDF
  OCR-routing check).
- OCR model files go to `<cache_dir>/ocr/` via rapidocr's
  `Global.model_root_dir` param — never into site-packages.
- Line length 100 (ruff).

## File Structure

- Create `src/minirag_mcp/ocr.py` — the only file that imports rapidocr /
  pypdfium2 / rapid_orientation (all imports lazy, inside functions).
- Create `tests/test_ocr.py` — unit tests (fake engine) + slow real-engine
  tests; `tests/pdf_builder.py` — tiny hand-rolled PDF byte builder.
- Modify `pyproject.toml` — `[project.optional-dependencies] ocr = [...]`.
- Modify `src/minirag_mcp/config.py` — two fields.
- Modify `src/minirag_mcp/ingest/parser.py` — image branch, scanned-PDF
  branch, `supported_extensions()`, `ParsedDoc.ocr_engine`.
- Modify `src/minirag_mcp/ingest/scanner.py` — use `supported_extensions()`.
- Modify `src/minirag_mcp/ingest/pipeline.py` — pass config to parser,
  carry `ocr_engine` into records.
- Modify `src/minirag_mcp/store.py` — `ocr_engine` column + migration +
  `SourceInfo.ocr_engine`.
- Modify `src/minirag_mcp/server.py`, `src/minirag_mcp/cli/__init__.py` —
  surface the marker in `list_files`.

---

### Task 1: Config fields and the `ocr` extra

**Files:**
- Modify: `pyproject.toml` (after the `[project.scripts]` block)
- Modify: `src/minirag_mcp/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.ocr_lang: str` (default `"eslav"`),
  `Config.ocr_min_chars_per_page: int` (default `25`). Later tasks read
  both from the `Config` object they already receive.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_config.py`;
  follow the file's existing style of calling `load_config({})`)

```python
def test_ocr_defaults():
    cfg = load_config({}, cwd=Path("/tmp"))
    assert cfg.ocr_lang == "eslav"
    assert cfg.ocr_min_chars_per_page == 25


def test_ocr_lang_override():
    cfg = load_config({"RAG_OCR_LANG": "cyrillic"}, cwd=Path("/tmp"))
    assert cfg.ocr_lang == "cyrillic"


def test_ocr_min_chars_zero_allowed_and_negative_rejected():
    cfg = load_config({"RAG_OCR_MIN_CHARS_PER_PAGE": "0"}, cwd=Path("/tmp"))
    assert cfg.ocr_min_chars_per_page == 0
    with pytest.raises(ConfigError):
        load_config({"RAG_OCR_MIN_CHARS_PER_PAGE": "-1"}, cwd=Path("/tmp"))


def test_ocr_lang_blank_rejected():
    with pytest.raises(ConfigError):
        load_config({"RAG_OCR_LANG": "  "}, cwd=Path("/tmp"))
```

Match the existing imports at the top of `tests/test_config.py`; add
`pytest` / `ConfigError` imports only if not already present.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_config.py -q -k ocr`
Expected: FAIL — `Config.__init__() got an unexpected keyword argument` /
`AttributeError: ocr_lang`.

- [ ] **Step 3: Implement.** In `config.py` add to the `Config` dataclass
  (after `allow_private_urls: bool`):

```python
    ocr_lang: str = "eslav"
    ocr_min_chars_per_page: int = 25
```

In `load_config`, before the final `return Config(...)`:

```python
    ocr_lang = env.get("RAG_OCR_LANG", "eslav").strip()
    if not ocr_lang:
        raise ConfigError("RAG_OCR_LANG must be a non-empty language code, e.g. 'eslav'")
```

and add to the `Config(...)` call:

```python
        ocr_lang=ocr_lang,
        ocr_min_chars_per_page=_int(env, "RAG_OCR_MIN_CHARS_PER_PAGE", 25, 0),
```

In `pyproject.toml`, after `[project.scripts]`:

```toml
[project.optional-dependencies]
ocr = [
    "rapidocr>=3.9",
    "onnxruntime>=1.19",
    "pypdfium2>=5",
    "rapid-orientation>=0.0.11",
]
```

- [ ] **Step 4: Verify green + lock**

Run: `uv run pytest tests/test_config.py -q && uv lock && uv run ruff check src tests`
Expected: PASS; `uv.lock` updated (extras resolved, nothing installed by
default).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/minirag_mcp/config.py tests/test_config.py
git commit -m "feat(config): add RAG_OCR_LANG and RAG_OCR_MIN_CHARS_PER_PAGE, declare the [ocr] extra"
```

---

### Task 2: `ocr.available()` and per-page PDF text extraction

**Files:**
- Create: `src/minirag_mcp/ocr.py`
- Create: `tests/pdf_builder.py`
- Create: `tests/test_ocr.py`

**Interfaces:**
- Produces:
  - `ocr.available() -> bool` — True iff `rapidocr` and `pypdfium2` are
    importable.
  - `ocr.pdf_page_texts(path: Path) -> list[str]` — extracted text-layer
    text per page, `""` for a page with none (pdfminer, one pass).
  - `ocr.ENGINE_RAPIDOCR = "rapidocr"` — the marker value stored on chunks.
  - `tests/pdf_builder.py: build_pdf(pages: list[str]) -> bytes` — a
    minimal valid PDF; `""` produces a content-free page (a stand-in for a
    scanned page: zero extractable chars).

- [ ] **Step 1: Write `tests/pdf_builder.py`** (a test helper, not a test —
  hand-rolled so the fast suite needs no PDF-writing dependency)

```python
"""Minimal PDF byte builder for tests: text pages and blank (scan-like) pages."""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: list[str]) -> bytes:
    """A valid single-font PDF. Each entry is the page's text; "" means no text.

    Cyrillic cannot be encoded in the standard-font byte strings this builder
    writes, so tests use ASCII markers; the OCR routing logic under test only
    counts characters.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # object numbers are 1-based

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    kids_placeholder = add(b"")  # pages node, patched below
    for text in pages:
        if text:
            stream = f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET".encode()
            content = add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
            body = (
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (kids_placeholder, font, content)
            )
        else:
            body = (
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] >>" % kids_placeholder
            )
        page_ids.append(add(body))
    kids = b" ".join(b"%d 0 R" % p for p in page_ids)
    objects[kids_placeholder - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        kids,
        len(page_ids),
    )
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % kids_placeholder)

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += (
        b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, catalog, xref_at)
    )
    return bytes(out)
```

- [ ] **Step 2: Write the failing tests** (`tests/test_ocr.py`)

```python
"""OCR module tests. Fast tests never import rapidocr; slow ones need the extra."""

from __future__ import annotations

from pathlib import Path

from minirag_mcp import ocr
from tests.pdf_builder import build_pdf


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
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_ocr.py -q`
Expected: FAIL — `ModuleNotFoundError: minirag_mcp.ocr`.

- [ ] **Step 4: Implement `src/minirag_mcp/ocr.py`**

```python
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

INSTALL_HINT = "install the OCR extra: uv tool install 'minirag-mcp[ocr]' (or pip install 'minirag-mcp[ocr]')"


class OcrError(Exception):
    """OCR was needed and failed or is unavailable. Always loud, never a no-op."""


def available() -> bool:
    return all(
        importlib.util.find_spec(mod) is not None for mod in ("rapidocr", "pypdfium2")
    )


def pdf_page_texts(path: Path) -> list[str]:
    """Text-layer text per page, one pdfminer pass. "" for a page with none."""
    texts: list[str] = []
    for layout in extract_pages(str(path)):
        parts = [el.get_text() for el in layout if isinstance(el, LTTextContainer)]
        texts.append("".join(parts))
    return texts
```

- [ ] **Step 5: Verify green**

Run: `uv run pytest tests/test_ocr.py -q && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/minirag_mcp/ocr.py tests/test_ocr.py tests/pdf_builder.py
git commit -m "feat(ocr): availability probe and per-page PDF text extraction"
```

---

### Task 3: OCR engine core — render, orient, recognize, assemble

**Files:**
- Modify: `src/minirag_mcp/ocr.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Consumes: `Config.ocr_lang`, `Config.cache_dir` (Task 1).
- Produces:
  - `ocr.assemble_lines(boxes: list, txts: list) -> str` — reading-order
    text from RapidOCR boxes; pure function, unit-tested.
  - `ocr.ocr_pdf(path: Path, config: Config, pages: Sequence[int]) -> dict[int, str]`
    — OCR of the given zero-based page indices.
  - `ocr.ocr_image(path: Path, config: Config) -> str`
  - Both raise `OcrError` (with `INSTALL_HINT` inside the message) when the
    extra is missing, and wrap engine/model-download failures in `OcrError`.

- [ ] **Step 1: Write the failing unit tests** (append to `tests/test_ocr.py`)

```python
def test_assemble_lines_reading_order():
    # Three boxes: two on one visual line (given out of x-order), one below.
    # Box format mirrors RapidOCR: 4 points, [x, y] each.
    boxes = [
        [[300, 10], [400, 10], [400, 30], [300, 30]],   # line 1, right
        [[10, 12], [200, 12], [200, 32], [10, 32]],     # line 1, left
        [[10, 100], [200, 100], [200, 120], [10, 120]], # line 2
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
```

Add `import pytest` to the test file's imports.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_ocr.py -q`
Expected: FAIL — `AttributeError: assemble_lines` / `ocr_image`.

- [ ] **Step 3: Implement** (append to `src/minirag_mcp/ocr.py`)

```python
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minirag_mcp.config import Config

logger = logging.getLogger(__name__)

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


def _engine(config: Config):
    """One cached RapidOCR per (lang, cache_dir); models download on first use."""
    key = (config.ocr_lang, str(config.cache_dir))
    engine = _engines.get(key)
    if engine is None:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

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
        except OcrError:
            raise
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
    import numpy as np
    from rapid_orientation import RapidOrientation

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
        from rapid_orientation import RapidOrientation

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
    import pypdfium2 as pdfium

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
    import cv2
    import numpy as np

    # np.fromfile + imdecode instead of cv2.imread: imread cannot open
    # non-ASCII paths on Windows, and this corpus's filenames are Russian.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise OcrError(f"could not decode image {path}")
    return _recognize(_orient(img), config)
```

Note: `_require_available()` runs before the lazy imports, so the
`monkeypatch.setattr(ocr, "available", ...)` test exercises the loud error
without the extra present.

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/test_ocr.py -q && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Write the slow real-engine tests** (append to
  `tests/test_ocr.py`)

```python
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
        cv2.putText(
            img, "ROTATION CHECK 777", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3
        )
    rotated = np.rot90(img, k=1)
    p = tmp_path / "rot.png"
    cv2.imwrite(str(p), rotated)
    cfg = load_config({}, cwd=tmp_path)
    assert "777" in ocr.ocr_image(p, cfg)
```

- [ ] **Step 6: Run the slow tests with the extra installed**

Run: `uv run --extra ocr pytest tests/test_ocr.py -m slow -q`
Expected: PASS (first run downloads models into
`~/Library/Caches/minirag-mcp/models/ocr` — check that path exists after).
If `test_real_engine_fixes_rotated_page` fails while the unrotated test
passes, the rot90 sign in `_orient` is inverted — change
`k=quarter_turns` to `k=-quarter_turns` and re-run; keep whichever passes.

- [ ] **Step 7: Verify the fast suite is still green WITHOUT the extra**

Run: `uv run pytest -q`
Expected: PASS, slow tests deselected.

- [ ] **Step 8: Commit**

```bash
git add src/minirag_mcp/ocr.py tests/test_ocr.py
git commit -m "feat(ocr): RapidOCR engine with orientation fix and reading-order assembly"
```

---

### Task 4: Image files become documents; scanner honors the OCR gate

**Files:**
- Modify: `src/minirag_mcp/ingest/parser.py`
- Modify: `src/minirag_mcp/ingest/scanner.py:11` (import) and `:76`
  (extension check)
- Test: `tests/test_parser.py`, `tests/test_scanner.py`

**Interfaces:**
- Consumes: `ocr.available()`, `ocr.ocr_image(path, config)`,
  `ocr.ENGINE_RAPIDOCR`, `ocr.OcrError` (Task 3).
- Produces:
  - `parser.IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"})`
  - `parser.supported_extensions() -> frozenset[str]` — the scan whitelist;
    includes images only when `ocr.available()`.
  - `ParsedDoc.ocr_engine: str = ""` — non-empty when OCR produced (part of)
    the markdown.
  - `parser.parse_file(path: Path, config: Config | None = None) -> ParsedDoc`
    — the new optional parameter; existing single-argument callers keep
    working with unchanged behavior.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_parser.py`)

```python
def test_supported_extensions_without_ocr(monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: False)
    exts = parser.supported_extensions()
    assert ".pdf" in exts and ".png" not in exts


def test_supported_extensions_with_ocr(monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    exts = parser.supported_extensions()
    assert ".png" in exts and ".webp" in exts


def test_parse_file_image_routes_through_ocr(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "recognized scan text")
    img = tmp_path / "Договор_поставки_2026.png"
    img.write_bytes(b"png bytes irrelevant, ocr is faked")
    doc = parser.parse_file(img, load_config({}, cwd=tmp_path))
    assert doc.markdown == "recognized scan text"
    assert doc.ocr_engine == "rapidocr"
    assert doc.title == "Договор поставки 2026"


def test_parse_file_image_without_ocr_raises_hint(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: False)
    img = tmp_path / "scan.png"
    img.write_bytes(b"...")
    with pytest.raises(parser.ParserError, match=r"minirag-mcp\[ocr\]"):
        parser.parse_file(img, load_config({}, cwd=tmp_path))
```

And in `tests/test_scanner.py`:

```python
def test_scan_roots_sees_images_only_with_ocr(tmp_path, monkeypatch):
    from minirag_mcp import ocr

    (tmp_path / "doc.md").write_text("# hi")
    (tmp_path / "scan.png").write_bytes(b"...")

    monkeypatch.setattr(ocr, "available", lambda: False)
    names = {e.path.name for e in scan_roots([tmp_path])}
    assert names == {"doc.md"}

    monkeypatch.setattr(ocr, "available", lambda: True)
    names = {e.path.name for e in scan_roots([tmp_path])}
    assert names == {"doc.md", "scan.png"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_parser.py tests/test_scanner.py -q`
Expected: FAIL — `AttributeError: supported_extensions` etc.

- [ ] **Step 3: Implement.** In `parser.py`:

Add after the `SUPPORTED_EXTENSIONS` block (`parser.py:33`):

```python
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"})


def supported_extensions() -> frozenset[str]:
    """The scan whitelist. Images are documents only when OCR can read them —
    without it they would ingest as nothing, and most images in a docs tree
    are illustrations, so their absence is silence, not an error."""
    if ocr.available():
        return SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS
    return SUPPORTED_EXTENSIONS
```

Add `from minirag_mcp import ocr` to the imports and
`from minirag_mcp.config import Config` under `TYPE_CHECKING` (parser must
not import config at runtime — check for cycles; if `config` imports
nothing from `ingest`, a plain import is fine and preferred).

Extend `ParsedDoc`:

```python
@dataclass(frozen=True)
class ParsedDoc:
    markdown: str
    title: str
    has_title: bool = True
    # Non-empty when OCR produced some or all of `markdown` (engine name).
    ocr_engine: str = ""
```

Change `parse_file` signature and add the image branch at its top:

```python
def parse_file(path: Path, config: Config | None = None) -> ParsedDoc:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        if not ocr.available():
            raise ParserError(
                f"{path} is an image, which needs OCR; {ocr.INSTALL_HINT}"
            )
        if config is None:
            raise ParserError(f"OCR for {path} needs a Config (internal error)")
        try:
            text = ocr.ocr_image(path, config)
        except ocr.OcrError as e:
            raise ParserError(str(e)) from e
        stem = _strip_extension_suffix(path.stem)
        title = _display_stem(stem) if _is_informative_stem(stem) else path.stem
        return ParsedDoc(
            markdown=text, title=title, ocr_engine=ocr.ENGINE_RAPIDOCR
        )
    # ... existing markitdown flow unchanged below
```

In `scanner.py` change the import from `SUPPORTED_EXTENSIONS` to
`supported_extensions` and in `scan_roots` hoist one call before the walk:

```python
    exts = supported_extensions()
    ...
                if p.suffix.lower() not in exts:
                    continue
```

Also update `parser.py`'s `_EXTENSION_SUFFIX_RE` — it is built from
`SUPPORTED_EXTENSIONS` at import time; extend it to
`SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS` so `"scan.png.png"` degrades the
same way PDFs do.

- [ ] **Step 4: Run to verify green**

Run: `uv run pytest tests/test_parser.py tests/test_scanner.py -q && uv run pytest -q`
Expected: PASS (full fast suite — pipeline's `ingest_file` still calls
`parse_file(path)` with one argument; that keeps working until Task 7).

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/ingest/parser.py src/minirag_mcp/ingest/scanner.py tests/test_parser.py tests/test_scanner.py
git commit -m "feat(ingest): image files are documents when the [ocr] extra is present"
```

---

### Task 5: Scanned-PDF fallback in the parser

**Files:**
- Modify: `src/minirag_mcp/ingest/parser.py` (inside `parse_file`, after the
  markitdown conversion)
- Test: `tests/test_parser.py`

**Interfaces:**
- Consumes: `ocr.pdf_page_texts`, `ocr.ocr_pdf`, `Config.ocr_min_chars_per_page`
  (Tasks 1–3), `tests.pdf_builder.build_pdf` (Task 2).
- Produces: `parse_file` PDF behavior:
  - all pages have text → unchanged markitdown result, `ocr_engine == ""`;
  - some/all pages below the threshold and OCR available → page-ordered
    merge of text-layer pages and OCR'd pages, `ocr_engine == "rapidocr"`;
  - ALL pages below the threshold and OCR missing → `ParserError` naming
    `minirag-mcp[ocr]`;
  - mixed pages and OCR missing → markitdown result as-is (partial text
    beats refusal; unchanged current behavior).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_parser.py`;
  `build_pdf` imported from `tests.pdf_builder`)

```python
LONG = "long enough text line to clear the default per-page threshold easily"


def _pdf(tmp_path, pages):
    p = tmp_path / "doc.pdf"
    p.write_bytes(build_pdf(pages))
    return p


def test_pdf_all_text_pages_skip_ocr(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_pdf", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not OCR"))
    )
    doc = parser.parse_file(_pdf(tmp_path, [LONG, LONG]), load_config({}, cwd=tmp_path))
    assert doc.ocr_engine == ""
    assert "long enough text" in doc.markdown


def test_pdf_scanned_pages_get_ocr(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_pdf", lambda path, config, pages: {i: f"OCR PAGE {i}" for i in pages}
    )
    doc = parser.parse_file(
        _pdf(tmp_path, [LONG, "", ""]), load_config({}, cwd=tmp_path)
    )
    assert doc.ocr_engine == "rapidocr"
    # page order preserved: text page first, then the two OCR'd pages
    body = doc.markdown
    assert body.index("long enough") < body.index("OCR PAGE 1") < body.index("OCR PAGE 2")


def test_pdf_fully_scanned_without_ocr_raises_hint(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: False)
    with pytest.raises(parser.ParserError, match=r"scanned.*minirag-mcp\[ocr\]"):
        parser.parse_file(_pdf(tmp_path, ["", ""]), load_config({}, cwd=tmp_path))


def test_pdf_mixed_without_ocr_keeps_partial_text(tmp_path, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.config import load_config
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: False)
    doc = parser.parse_file(_pdf(tmp_path, [LONG, ""]), load_config({}, cwd=tmp_path))
    assert doc.ocr_engine == ""
    assert "long enough" in doc.markdown


def test_pdf_without_config_behaves_as_before(tmp_path):
    from minirag_mcp.ingest import parser

    doc = parser.parse_file(_pdf(tmp_path, ["", ""]))
    assert doc.markdown.strip() == ""  # legacy path: empty conversion, no error
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/test_parser.py -q -k pdf`
Expected: the three OCR-routing tests FAIL (parse_file never OCRs today);
`test_pdf_without_config_behaves_as_before` PASSES already — it documents
the frozen legacy path, run it against current code to confirm.

- [ ] **Step 3: Implement.** In `parse_file` the existing tail is
  (`parser.py:374-380`, after the image branch added in Task 4):

```python
    try:
        result = _md().convert(str(path))
    except Exception as e:  # markitdown may raise lib-specific errors too
        raise ParserError(f"Failed to convert {path}: {e}") from e
    markdown = _converted_markdown(result)
    return ParsedDoc(markdown=markdown, title=_file_title(markdown, result.title, path.stem))
```

Keep the try/except; replace everything from `markdown = ...` down with:

```python
    markdown = _converted_markdown(result)
    ocr_engine = ""
    if config is not None and path.suffix.lower() == ".pdf":
        markdown, ocr_engine = _pdf_with_ocr_fallback(path, markdown, config)
    return ParsedDoc(
        markdown=markdown,
        title=_file_title(markdown, result.title, path.stem),
        ocr_engine=ocr_engine,
    )
```

and add the helper:

```python
def _pdf_with_ocr_fallback(path: Path, markdown: str, config) -> tuple[str, str]:
    """Route near-empty PDF pages through OCR; keep text-layer pages as they are.

    Per-page, not whole-document: a text cover page over scanned pages must
    not mask them. Threshold 0 disables the check entirely.
    """
    threshold = config.ocr_min_chars_per_page
    if threshold <= 0:
        return markdown, ""
    try:
        page_texts = ocr.pdf_page_texts(path)
    except Exception:
        # pdfminer failing on an exotic PDF must not break the markitdown
        # result that already succeeded.
        return markdown, ""
    needs = [i for i, text in enumerate(page_texts) if len(text.strip()) < threshold]
    if not needs:
        return markdown, ""
    if not ocr.available():
        if len(needs) == len(page_texts):
            raise ParserError(
                f"{path} looks like a scanned PDF (no text layer); {ocr.INSTALL_HINT}"
            )
        return markdown, ""  # partial text beats refusal
    try:
        recognized = ocr.ocr_pdf(path, config, needs)
    except ocr.OcrError as e:
        raise ParserError(str(e)) from e
    parts = [
        recognized.get(i, "") if i in recognized else text
        for i, text in enumerate(page_texts)
    ]
    merged = "\n\n".join(p.strip() for p in parts if p.strip())
    return merged, ocr.ENGINE_RAPIDOCR
```

Note the merge deliberately rebuilds the whole document from per-page
texts when any page was OCR'd: markitdown's markdown has no page
boundaries to splice into.

- [ ] **Step 4: Run to verify green**

Run: `uv run pytest tests/test_parser.py -q && uv run pytest -q && uv run ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/ingest/parser.py tests/test_parser.py
git commit -m "feat(ingest): route near-empty PDF pages through OCR, fail loudly when it is missing"
```

---

### Task 6: The `ocr_engine` marker in the store and listings

**Files:**
- Modify: `src/minirag_mcp/store.py` — `ChunkRecord` (`:33`), create-table
  schema (`:201`), `_ADDED_COLUMNS`, `_META_COLS` (`:78`), `SourceInfo`
  (`:66`), `list_sources` (`:370`), `get_source` (`:392`)
- Modify: `src/minirag_mcp/ingest/scanner.py` — `FileState` (`:163`),
  `compute_states` (`:182`)
- Modify: `src/minirag_mcp/server.py:306-317`, `src/minirag_mcp/cli/__init__.py:359-371`
- Test: `tests/test_store.py`, `tests/test_scanner.py`

**Interfaces:**
- Produces:
  - `ChunkRecord.ocr_engine: str = ""` — set by the pipeline (Task 7).
  - `SourceInfo.ocr_engine: str = ""` — first non-empty engine among the
    source's chunks.
  - `FileState.ocr_engine: str = ""`.
  - `list_files` payloads gain `"ocrEngine"`; the CLI human line shows
    ` [ocr:rapidocr]` when set.

- [ ] **Step 1: Write the failing tests.** In `tests/test_store.py` (follow
  the file's existing fixture style for constructing a `Store` and records —
  reuse its record-builder helper if one exists, adding
  `ocr_engine="rapidocr"` to one record):

```python
def test_ocr_engine_round_trips_and_aggregates(store_factory, record_factory):
    store = store_factory()
    recs = [
        record_factory(id="s#0", source="s", chunk_index=0, ocr_engine="rapidocr"),
        record_factory(id="s#1", source="s", chunk_index=1),
    ]
    store.replace_source("s", recs)
    info = store.get_source("s")
    assert info.ocr_engine == "rapidocr"
    assert [s.ocr_engine for s in store.list_sources()] == ["rapidocr"]


def test_old_table_without_ocr_column_migrates(store_factory):
    # Simulate a pre-OCR index: build a store, drop the column via to_arrow
    # round-trip is heavy — instead assert the migration map covers it.
    from minirag_mcp.store import _ADDED_COLUMNS

    assert _ADDED_COLUMNS["ocr_engine"] == "''"
```

(Adapt fixture names to the file's actual helpers — `tests/test_store.py`
already constructs stores and `ChunkRecord`s; copy its established pattern
rather than inventing new fixtures.)

In `tests/test_scanner.py`, extend the existing `compute_states` test data
with an indexed source whose `SourceInfo` has `ocr_engine="rapidocr"` and
assert the resulting `FileState.ocr_engine == "rapidocr"`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_store.py tests/test_scanner.py -q`
Expected: FAIL — unexpected keyword `ocr_engine`.

- [ ] **Step 3: Implement.**
  - `ChunkRecord`: add `ocr_engine: str = ""` after `parent_id`.
  - Create-table schema: add `pa.field("ocr_engine", pa.string())` after
    `parent_id`.
  - `_ADDED_COLUMNS`: add `"ocr_engine": "''"` (the existing
    `_ensure_scheme_columns` migration picks it up on open).
  - `_META_COLS`: append `"ocr_engine"`.
  - `SourceInfo`: add `ocr_engine: str = ""`.
  - `list_sources`: track `engines[src] = engines.get(src) or row.get("ocr_engine", "")`
    alongside the existing per-source dicts and pass
    `ocr_engine=engines[src] or ""` to `SourceInfo`.
  - `get_source`: `ocr_engine=next((r.get("ocr_engine", "") for r in rows if r.get("ocr_engine")), "")`.
  - `FileState`: add `ocr_engine: str = ""`; in `compute_states`, thread
    `prior.ocr_engine` into the two branches that build a `FileState` from a
    `prior` (`ingested`/`stale`), and into the data/url tail loop.
  - `server.py` `list_files` dict: add `"ocrEngine": s.ocr_engine`.
  - `cli/__init__.py` payload: add `"ocrEngine": s.ocr_engine`; human line:

```python
    human = "\n".join(
        f"[{s.state}] {s.source} ({s.chunk_count} chunks)"
        + (f" [ocr:{s.ocr_engine}]" if s.ocr_engine else "")
        for s in states
    )
```

- [ ] **Step 4: Run to verify green**

Run: `uv run pytest -q && uv run ruff check src tests`
Expected: PASS (row `.get` defaults keep un-migrated read-only tables
working, mirroring `_scheme_of`).

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/store.py src/minirag_mcp/ingest/scanner.py src/minirag_mcp/server.py src/minirag_mcp/cli/__init__.py tests/test_store.py tests/test_scanner.py
git commit -m "feat(store): record and surface which OCR engine produced a source's text"
```

---

### Task 7: Pipeline integration and end-to-end sync

**Files:**
- Modify: `src/minirag_mcp/ingest/pipeline.py` — `ingest_file` (`:124-147`),
  `_chunk_and_store` (`:72-122`)
- Test: `tests/test_pipeline.py`, `tests/test_sync.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `ingest_file` passes `self.config` to `parse_file`, validates
  extensions against `supported_extensions()`, and stores
  `doc.ocr_engine` on every `ChunkRecord` of the source.

- [ ] **Step 1: Write the failing tests.** In `tests/test_pipeline.py`
  (reuse the file's existing pipeline fixture — it already builds a
  `Pipeline` with a fake embedder and a real store):

```python
def test_ingest_file_carries_ocr_engine(tmp_path, pipeline, monkeypatch):
    from minirag_mcp import ocr
    from minirag_mcp.ingest import parser

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "scanned contract text")
    img = tmp_path / "scan.png"
    img.write_bytes(b"...")
    result = pipeline.ingest_file(img)
    assert result.chunk_count >= 1
    info = pipeline.store.get_source(str(img))
    assert info.ocr_engine == "rapidocr"


def test_ingest_file_image_without_ocr_is_unsupported(tmp_path, pipeline, monkeypatch):
    from minirag_mcp import ocr

    monkeypatch.setattr(ocr, "available", lambda: False)
    img = tmp_path / "scan.png"
    img.write_bytes(b"...")
    with pytest.raises(UnsupportedFormatError):
        pipeline.ingest_file(img)
```

In `tests/test_sync.py`, an end-to-end: a root with one markdown file and
one image, fake OCR, `run_sync` → both ingested; re-run → both unchanged
(`skipped == 2`). Follow the file's existing `run_sync` test arrangement.

```python
def test_sync_ingests_images_via_ocr(tmp_path, pipeline_factory, monkeypatch):
    from minirag_mcp import ocr

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "scanned text body")
    (tmp_path / "note.md").write_text("# plain note\n\nbody")
    (tmp_path / "scan.jpg").write_bytes(b"...")
    pipeline, store = pipeline_factory(tmp_path)
    counts, errors = run_sync(pipeline, store, [tmp_path], max_file_size=10_000_000)
    assert errors == []
    assert counts["ingested"] == 2
    counts2, _ = run_sync(pipeline, store, [tmp_path], max_file_size=10_000_000)
    assert counts2["skipped"] == 2 and counts2["ingested"] == 0
```

(Adapt fixture names to what `tests/test_sync.py` actually provides.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_pipeline.py tests/test_sync.py -q`
Expected: FAIL — `ingest_file` rejects `.png` (`UnsupportedFormatError`)
in the OCR-enabled tests because it still checks the static set.

- [ ] **Step 3: Implement.** In `pipeline.py`:
  - Import `supported_extensions` instead of `SUPPORTED_EXTENSIONS` (keep
    `find_title`, `parse_file`, `parse_html` imports).
  - `ingest_file`: replace the extension check body with

```python
        exts = supported_extensions()
        if path.suffix.lower() not in exts:
            raise UnsupportedFormatError(
                f"Unsupported file extension {path.suffix!r}; supported: "
                + ", ".join(sorted(exts))
            )
```

  and change the parse call to `doc = parse_file(path, self.config)`, then
  pass `ocr_engine=doc.ocr_engine` into `_chunk_and_store`.
  - `_chunk_and_store`: add keyword-only parameter `ocr_engine: str = ""`
    and set `ocr_engine=ocr_engine` on each `ChunkRecord`.

- [ ] **Step 4: Run the full gate**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: all green.

- [ ] **Step 5: Run the slow suite with the extra, as release-grade proof**

Run: `uv run --extra ocr pytest -m slow -q`
Expected: PASS (embedding model + OCR models download on first run).

- [ ] **Step 6: Commit**

```bash
git add src/minirag_mcp/ingest/pipeline.py tests/test_pipeline.py tests/test_sync.py
git commit -m "feat(ingest): wire OCR through the pipeline end to end"
```

---

### Task 8: Close out

- [ ] **Step 1: Manual smoke on the real private corpus** (assets location:
  `bd memories ocr-test-assets`): run
  `uv run --extra ocr minirag-mcp sync --base-dir <folder with the test PDF> --db-path /tmp/ocr-smoke-db`
  then `minirag-mcp list-files` and a search for a term you can see on a
  scan. Expect the PDF ingested with `[ocr:rapidocr]` and findable.
- [ ] **Step 2:** `bd close minirag-mcp-286` with a reason naming the
  measured smoke result.
- [ ] **Step 3:** `git pull --rebase && git push && git status` — must be
  "up to date with origin".
