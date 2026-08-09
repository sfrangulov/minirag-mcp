# OCR for scanned documents — design

Date: 2026-08-09. Status: draft, pending review.

## Problem

A scanned (image-only) PDF passes through markitdown/pdfminer and produces an
empty document: `_chunk_and_store` raises `EmptyDocumentError`, sync counts the
file as failed, and the document is absent from the corpus. Standalone image
files (`.png`, `.jpg`, …) are not in `SUPPORTED_EXTENSIONS` at all, so a scan
stored as a picture is invisible to `scan_roots` in the first place. The target
corpus is Russian-language office documents (УПД, invoices, acts, приказы) of
middling scan quality — печати, подписи, tables.

## Constraints

- **Local only.** No cloud OCR in any form, including opt-in. A locally hosted
  inference server (ollama / llama.cpp / vLLM) counts as local.
- **`uvx minirag-mcp` must keep working** with no system binaries and no
  compilation; OCR ships as optional extras.
- **MIT-distributable dependency chain**, model weights included. Weights under
  OpenRAIL (revenue caps), CC-BY-NC, or custom licenses with termination
  clauses are excluded even as extras.
- **Cyrillic recognition quality is the primary criterion.** Reference
  hardware for the heavy tier: MacBook Pro M5, 64 GB unified memory; CUDA
  boxes are secondary.

## Design: two tiers plus a remote mode

### Extras

```toml
[project.optional-dependencies]
ocr     = ["rapidocr>=3.9", "onnxruntime>=1.19", "pypdfium2>=5"]
ocr-max = ["torch>=2.2", "transformers", "accelerate"]
```

Plain dependencies, no index tricks: pip extras cannot select wheel indexes, so
GPU wheel choice is delegated to the environment (`UV_TORCH_BACKEND` for uv
users; documented in README). On macOS arm64 the default torch wheel (111 MB)
already includes MPS. Never declare `onnxruntime-gpu` in any extra — it shares
the `onnxruntime` import name and the loser of that conflict is undefined
(docling-serve #434 shipped a CUDA image silently running CPU).

### Tier selection

`RAG_OCR_ENGINE = auto | rapidocr | vlm` (default `auto`: `vlm` when its
dependencies or endpoint are configured and available, else `rapidocr`, else
OCR is off). Every resolution is logged once, loudly, with the reason.

### `[ocr]` — fast tier (RapidOCR)

- Engine: RapidOCR on onnxruntime — the runtime fastembed already ships, so
  the marginal weight is ~160 MB (opencv dominates). Apache-2.0 end to end,
  pip-only wheels on macOS/Linux/Windows.
- Models: PP-OCRv5, `lang_type` from `RAG_OCR_LANG` (default `eslav`;
  the stock ch/en model silently drops Cyrillic, so the default must be
  explicitly Cyrillic-capable). `model_type=mobile` — verified this session
  that no server-size Cyrillic recognition model exists anywhere (HF 404,
  RapidOCR model list, PP-OCRv6 has zero Cyrillic), so mobile is the ceiling
  of this tier by construction.
- Model files download into `<base_dir>/ocr-models/`, not site-packages
  (uvx environments are cache-scoped and may be read-only). Download failure
  is a loud per-file error, never a silent empty index.
- CoreML execution provider auto-enabled on macOS (`use_coreml`; verified the
  default onnxruntime wheel carries it). Speedup is measured before being
  advertised.
- PDF rendering to images via pypdfium2 (Apache/BSD; poppler-free;
  0.08 s/page at 300 dpi measured).

### `[ocr-max]` — accuracy tier (VLM)

- Default model: **Qwen3-VL-8B-Instruct** (`RAG_OCR_MODEL` to override).
  Apache-2.0 code and weights (verified via HF API); Cyrillic named in the
  official OCR capability docs; 0.642 % CER on the only public real-Russian
  office-document eval (ЦРПТ, 169 pages). The Instruct checkpoint, not
  Thinking — measured on this corpus, the Thinking variant burns its whole
  token budget on reasoning and returns empty content.
- **In-process is the primary path** — no external tools to install or run:
  - macOS arm64: `mlx-vlm` under a `sys_platform == "darwin" and
    platform_machine == "arm64"` marker (docling's pattern), loading a
    community MLX quant of the model (4–8 bit builds of
    Qwen3-VL-8B-Instruct exist on HF; exact build pinned at
    implementation time after a smoke test).
  - CUDA/CPU: torch + transformers, device from `RAG_OCR_DEVICE =
    auto | cuda | mps | cpu` (auto resolves cuda → mps → cpu; an
    explicitly requested unavailable device is a hard error — docling's
    `decide_device` pattern).
- **Local server as optional escape hatch, not a requirement**:
  `RAG_OCR_ENDPOINT` + `RAG_OCR_MODEL` speak the OpenAI-compatible chat
  API of a local ollama/llama.cpp/vLLM using the existing `requests`
  dependency (core package, no extra needed). For users who already run
  such a server; nothing in the design depends on it. First use
  health-checks the endpoint and fails loudly if unreachable.
- Quality ceiling for 64 GB machines: Qwen3-VL-30B-A3B-Instruct (MoE, 3B
  active; Apache-2.0 verified) — a config value, not a code change.
- Endpoint-mode caveat that cost this design a debugging session (applies
  when the escape hatch points at ollama): its default context window
  (4096) is smaller than one page's visual tokens (~4100 for a 2000×2200
  scan). The endpoint mode must request `num_ctx >= 16384` and treat
  `done_reason: "length"` as an error, not a result. Also use the
  `-instruct` model tag — the bare tag is a thinking checkpoint that burns
  the whole token budget on reasoning and returns empty content.
- Runner-up, documented for the record: dots.ocr (3B) beats Qwen3-VL-8B on
  the Cyrillic tier of GlotOCR (Acc@5 78.3 vs 67.1) with the lowest
  hallucination rate (3.8 %), but has no GGUF/MLX path, needs
  `trust_remote_code`, and ships a supplemental license agreement with an
  auto-termination AUP clause covering "unauthorized document scanning" —
  legally ambiguous for an OCR tool, so it stays out until that clears.

### Trigger and pipeline placement

- Hook: `parser.parse_file`. For a PDF, per-page character counts come from
  pdfminer (already installed). Pages below `RAG_OCR_MIN_CHARS_PER_PAGE`
  (default 25) are OCRed; pages with a real text layer keep their extracted
  text; the results concatenate in page order. Per-page, not
  whole-document: a text cover page over 50 scanned pages must still OCR
  (the AnythingLLM all-or-nothing detection is the known anti-pattern).
- With no OCR installed, the current `EmptyDocumentError` message gains a
  hint: "this looks like a scanned PDF; install minirag-mcp[ocr]".
- Image files: when an OCR tier is active, `.png .jpg .jpeg .tiff .bmp
  .webp` join `SUPPORTED_EXTENSIONS` (scanner and pipeline); title comes
  from the filename through the existing `_file_title` logic. Without OCR,
  behavior is unchanged — images stay invisible rather than erroring, since
  most image files in a docs tree are illustrations, not documents.
- OCR output joins the normal chunk → embed → store path. RapidOCR emits
  text lines (reading-order sort + paragraph assembly is ours); the VLM
  emits Markdown directly (strip a wrapping ```markdown fence if present).
- Chunk records for OCR-derived text carry a marker (e.g. `ocr:<engine>` in
  a metadata field) so `status`/listing can say how a document entered the
  index.

### Guards

- Blank-page guard: a page whose OCR yields nothing is recorded as such;
  a document where *every* page yields nothing is a loud failure, not an
  empty success.
- Hallucination note: on the measured УПД page the VLM filled an illegible
  handwritten stamp field with a plausible name taken from elsewhere in the
  document. OCR text is never authoritative over the source scan; the
  README documents this, and handwriting-heavy corpora should prefer the
  fast tier or human review.
- Timeouts per page (VLM pages can take minutes on weak hardware); sync
  already tolerates per-file failures without aborting the job.

## Image preparation before OCR

Researched (4-lens workflow + a 31-run degradation experiment on the real УПД
scan, 2026-08-09). The governing finding: both engines already do most
classical preprocessing internally, and DL-era engines are trained on
gray/RGB imagery — most Tesseract-era preparation lore does not transfer and
some of it actively harms.

### Adopted

- **Render policy is the preprocessing.** RapidOCR silently downscales any
  input beyond `Global.max_side_len=2000` px long side and cuts recognition
  crops from the *downscaled* image — a 300-DPI A4 render (2480×3508) drops
  to ~170 DPI effective. The [ocr] tier renders at 300 DPI and passes
  `{"Global.max_side_len": 4000, "Det.limit_type": "max",
  "Det.limit_side_len": 1600}` so detection runs on a cheap small copy while
  recognition crops stay sharp. Measured neutral on the УПД test page (its
  source is only 2227 px long side — an 11 % resample), same wall time; the
  11-page bench decides finally. Never render bigger without raising the cap
  — the engine provably discards those pixels.
- **VLM input cap.** Qwen3-VL's processor accepts up to 16.7 Mpx (~16 k
  visual tokens); its own OCR cookbooks cap at ~2.1 Mpx. The [ocr-max] tier
  caps renders at 2–4 Mpx (≈ A4 at 150–216 DPI) — feeding a raw 300-DPI
  render is pure prefill waste, and on ollama it re-arms the num_ctx trap.
- **Zero-recompression payload.** For a page that is one full-page
  DCT/JPX-encoded image (the standard scan), `PdfImage.extract()` returns
  the byte-identical embedded JPEG in ~0 ms — the ideal endpoint payload
  (no second JPEG generation). Guard: extraction skips the page `/Rotate`
  attribute (rendering applies it automatically — verified), so the caller
  must rotate per `page.get_rotation()`; fall back to rendering for
  multi-image pages, masks, or CCITT/JBIG2. No intermediate files anywhere:
  BGR ndarray straight into RapidOCR; PNG when encoding is forced.
- **Page orientation 0/90/180/270** — the one filter with measured
  modern-engine gains (up to 14 % closed / 4× open-weights,
  arXiv:2511.04161; the PP-OCR team ship a 99.06 % orientation classifier in
  their own default pipeline). RapidOCR's `cls` stage only fixes 180° flips
  per text line; sideways pages are otherwise lost, and standalone images
  add the EXIF problem (cv2.imread ignores EXIF orientation).
  Implementation: `rapid-orientation` (Apache-2.0, 6.5 MB ONNX, the same
  onnxruntime+opencv stack), confidence-gated so a misclassification cannot
  rotate a correct page. Its accuracy on Russian office scans is unverified
  — measured acceptance lives in the bench task.

### Candidate, decided by the bench

- **CLAHE** (contrast equalization, ~10 ms/page): on the degradation stand
  it gained +1 on the clean scan and +4 on the worst degraded variant, but
  lost −1 on the noise-only variant — so it ships only if the 11-page bench
  confirms it, and then gated on a low-contrast page metric, not
  unconditionally.

### Rejected, with reasons

- **Binarization** (Otsu/Sauvola/adaptive): DL recognizers are trained on
  gray/RGB; measured grayscale > binarized across architectures
  (arXiv:2008.02777); thresholding destroys печати/подписи — exactly this
  corpus's hard content. It exists in OCR lore to compensate for
  Tesseract's weak internal Otsu.
- **Denoising** (fastNlMeans): 0.6–0.9 s/page (~20 % of the RapidOCR
  budget), −1 on the clean scan in our experiment, engines train with noise
  augmentation. Median-3×3 stays a documented manual escape hatch at most.
- **Global deskew**: PP-OCR perspective-rectifies each detected quad, and
  the stand confirmed it — a 2.5° skew cost nothing (23/27 vs 21/27
  baseline noise), while the naive minAreaRect deskew *broke* the worst
  variant (11/27 vs 15/27 without it). Skip; revisit only if corpus
  inspection shows >3–5° skews are common.
- **unpaper/ScanTailor-class cleaning**: system binaries (breaks uvx), and
  OCRmyPDF's own docs warn cleaning may delete faint content — печати are
  the first casualty.
- **Super-resolution** (Real-ESRGAN/TSRN): measured gains exist only in the
  ~16 px camera-crop regime, cost dwarfs the budget on 300-DPI scans.
- **UVDoc dewarping**: targets photographed curved pages; flatbed/ADF
  corpus has none. Reconsider only if phone photos enter the corpus.

## Measured evidence (this session)

Test asset: an 11-page PDF of real scanned office documents (УПД, счета,
платёжки, акт, спецификация, приказ) assembled from a private slide deck.
It contains corporate data and lives strictly outside this repository; its
location is recorded in the local issue-tracker memory. Verified to yield
0 chars through the current parser. Head-to-head on the УПД page, entity
checklist = 27 verbatim-searchable facts (names, ИНН/БИН, amounts, dates,
people):

| Engine | Entities | Time/page | Hardware |
|---|---|---|---|
| RapidOCR eslav mobile (CPU) | 21/27 | ~4 s | M2 Pro, 16 GB |
| Qwen3-VL-8B-Instruct Q4 (ollama) | **27/27** | 161 s | M2 Pro, 16 GB |

RapidOCR's six misses are real recognition errors on blurry regions: р→о
and ц→и confusions inside low-contrast company names, Cyrillic «Б» read as
Latin «6»/«E» in codes, one dropped product name, one dropped total amount.
The VLM read all of these correctly and additionally produced real Markdown
tables. Expect the M5 to cut the VLM time several-fold (and MLX further);
the full 11-page benchmark on target hardware is an acceptance task, not a
prerequisite for implementation.

## Out of scope

- pdf-inspector (firecrawl): classification/routing only, no OCR; pre-1.0
  with CID-font bugs. Revisit if per-page routing of mixed PDFs or better
  text-PDF tables become priorities.
- VLM captioning of embedded images in pptx/docx (markitdown's `llm_client`
  already supports this shape) — separate future feature; those images are
  illustrations, not scans.
- Cloud OCR of any kind — excluded by constraint, not by omission.

## Rejected alternatives (license/hardware, all verified)

chandra (best measured Russian CER 0.51 % — OpenRAIL weights, revenue cap);
HunyuanOCR (license excludes EU/UK/KR); DeepSeek-OCR-2 (Apache, but collapses
on degraded Cyrillic, 26 % artifact rate on blank pages); Nanonets-OCR2 (no
weights license); marker/surya (OpenRAIL-M cap + llama-server binary);
MinerU (custom license, auto-termination); olmOCR (GPU-only, 12 GB VRAM min);
Nougat (CC-BY-NC, "Russian unsupported"); tesseract routes (system binary or
no Windows wheels); EasyOCR (PyTorch cost, stalled, no evidence over
baseline); docling as OCR (same eslav ceiling; its TableFormer structure
value may return later as an independent feature).
