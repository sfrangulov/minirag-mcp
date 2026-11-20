# minirag v2 — ingest (plan 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `sync` parses, chunks and embeds in worker processes and writes in batches; OCR runs through Tesseract in its own reading order; ~~struck-through `.docx` text stops being indexed~~ struck-through text stays indexed, marked `~~...~~` in every chunk that holds it (superseded by the owner, 2026-11-19; Task 1).

**Architecture:** `Pipeline` is split into a worker half (`prepare_file`: parse, chunk, page-map, embed; no store, no labels) and a parent half (`records`: labels and `ChunkRecord`s). `parallel.py` runs the worker half in a spawn-context process pool and isolates a worker that dies; `BatchWriter` writes the parent's records through a new `Store.replace_sources` in one commit per batch. `ocr.py` drops RapidOCR, cv2 and `assemble_lines` for one `tesserocr` handle per language set, fed by pinned `tessdata_fast` files. ~~The `.docx` fix lives in mammoth's style map, not in Markdown.~~ Struck-through text keeps markitdown's `~~...~~` (superseded by the owner, 2026-11-19): the parser rewrites double strikethrough as single, and the packer (`chunker/strike.py`, `chunker/pack.py`) closes and re-opens the markers wherever a unit boundary cuts a struck run.

**Tech Stack:** Python 3.11+, `concurrent.futures.ProcessPoolExecutor` (spawn), LanceDB, tesserocr 2.13 (libtesseract 5.5.4), Pillow, pypdfium2, markitdown/mammoth, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-19-minirag-v2-design.md` (section 5). Shared names: `docs/superpowers/plans/2026-11-19-minirag-v2-00-contract.md`. Plan 1 (core) has already run, and this plan builds on what it left (`docs/superpowers/plans/2026-11-19-minirag-v2-01-core.md`):
- `Embedder(model_name, cache_dir, *, endpoint=None, prefixes=None, threads=None, allow_fallback=True)`, picklable (configuration and expected probe kept; loaded model, token counter, HTTP session and fallback state dropped), with public `threads`, `backend` (`"endpoint"` | `"in-process"` | `"in-process (endpoint unreachable)"`) and `warning` (plan 1 Tasks 3, 10).
- `Pipeline(store, embedder, config, labels=None, *, before_store=None)`, `Pipeline.before_store` (a `handle._BeforeStore` the handle builds for each generation; `IndexHandle.ensure_probe()` calls the open generation's), `Pipeline._chunk` packing to `token_budget - prefix_tokens`, `Pipeline.reingest_stored(info, markdown)` (plan 1 Tasks 3, 10, 11).
- `Store._document_row(source, records) -> dict`, `Store._write_documents(rows)`, `Store._delete_documents(sources)`, `Store.backfill_documents(...)`, and `IndexBusyError` raised by `replace_source`/`move_source`/`delete_source`/`set_document_labels` while `identity.rebuild_running(db)` (plan 1 Tasks 7, 11).
- `sync._run_sync_unlocked(pipeline, store, roots, max_file_size, scope, on_event)` ending with `backfill_signatures`, `backfill_documents`, `update_indices` and the embedding-fallback error; `SyncManager(db_path, run)`; `rebuild.rebuild` re-ingests by calling `_run_sync_unlocked` on the staging store with a pipeline over the new embedder and no `before_store` (plan 1 Tasks 3, 7, 11).

## Global Constraints

- Python >= 3.11; everything through `uv`; CI matrix 3.11-3.14 on Linux and macOS.
- Gate after every task: `uv run pytest -q`, `uv run ruff check src tests`, `uv run ruff format --check src tests`.
- `uv run --extra ocr ...` leaves the extra installed; "green without the extra" holds only after `uv sync --exact`.
- The fast suite stays offline; anything that downloads (traineddata, models) is `@pytest.mark.slow`. Do not remove the `addopts` marker filter.
- Every fix's test is run against the unfixed code first and must go red (CLAUDE.md traps).
- English in code, docs, commits; comments state constraints only.
- MCP and CLI stay identical: behaviour lives in `sync.py`/`ocr.py`/`parser.py`, read from `Config`.
- No private corpus data: OCR numbers from the private spike appear only as aggregates (as in the spec).
- `[ocr]` extra: `tesserocr>=2.13` replaces rapidocr, rapid-orientation, opencv-python (+171.5 MB -> ~+9 MB).
- Traineddata `rus`, `eng`, `osd` from `tesseract-ocr/tessdata_fast` at commit `87416418657359cb625c412a48b6e1d6d41c29bd`, sha256-pinned, downloaded to `cache_dir/ocr` on first use.
- `PSM.AUTO_OSD`; PDF pages rendered at 300 dpi with `SetSourceResolution(300)`; `ocr_engine` = `"tesseract"`.
- Per-file failures never abort a sync; the sync lock is taken and released exactly as today.
- Commits: conventional style, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Track work in `bd`, not TodoWrite.

## Review Focus

1. A worker process killed mid-file (a C++ abort inside libtesseract or pdfium, or the OOM killer) -> that one file fails with a message saying the worker died; the sync finishes, releases its lock, and never hangs. Test: Task 7 `test_a_worker_that_dies_fails_only_its_own_file`.
2. The default worker count where each worker loads the in-process model (measured peak RSS 3.27 GB with e5-large) -> the count is capped by memory instead of driving a 16 GB laptop into swap. Test: Task 6 `test_default_workers` rows.
3. Worker processes of the stdio MCP server writing to stdout (a library's print, a progress bar) -> nothing reaches the protocol channel. Test: Task 7 `test_workers_never_write_to_stdout`.
4. An image with transparency or a palette (exported diagram, screenshot) -> read as text on white, not as a black page. Test: Task 3 `test_an_image_with_transparency_is_read_on_white`.
5. Two OCR calls at once in one process (an MCP `ingest_file` during a sync small enough to stay in-process) -> serialized on the one Tesseract handle, both correct. Test: Task 3 `test_two_recognitions_at_once_never_share_the_engine`.

## Decisions made here

- **`SCHEME_VERSION` is not bumped.** It marks chunker changes, and parser changes have never bumped it (`3716fb8`, data-URI stripping). Every index holding text from the old parser or RapidOCR is a 1.x index, which v2 never writes and re-indexes from scratch at `.minirag/v2/`; no v2 index exists before 2.0.0a1 (plan 4), so plan 1's identity would have nothing to detect. OCR'd rows are told apart anyway by their stored `ocr_engine`. If a 2.0.0aN ships before this plan lands, bump it then.
- **Default workers = physical cores - 1, capped by memory.** Half of physical RAM divided by one worker's footprint (1.5 GiB of interpreter and libraries, plus the model when it runs in-process). Measured 2026-11-19 on the owner's machine: one process embedding in-process peaks at 3.27 GB with e5-large (fastembed lists 2.24 GB) and 1.66 GB with MiniLM (0.22 GB). With an endpoint the cap is 21 on 64 GB, so the owner gets 17 workers. An explicit `RAG_SYNC_WORKERS` is taken as given. Flag to the owner: the cap goes beyond the spec's literal default.
- **A sync with fewer than 8 files to read stays in-process** (`FILES_PER_WORKER = 4`, workers <= files // 4): spawning workers that each load a model costs more than reading a handful of files.
- **Labels are computed in the parent**, inside the sync's one labelling job: the classify model is loaded once, never once per worker.
- **`RAG_OCR_LANG` takes Tesseract spelling** (`rus+eng` default; any `+`-join of `rus`, `eng`). The 1.x values `eslav` and `en` map to `rus+eng` and `eng`; everything else is refused, because only `rus`, `eng`, `osd` are pinned.
- **Struck-through text is kept and marked** (owner, 2026-11-19; supersedes the decision struck through below). markitdown's `~~...~~` stays: a deleted clause still records what was removed. `<w:dstrike/>` (double strikethrough) arrives as plain text because mammoth ignores it, so it is rewritten to `<w:strike` before conversion and marked like single strike. The packer pairs markers per paragraph, GFM-style (a lone `~~` and `~~` in code are literal), and where a unit boundary cuts a struck run it closes the run at the end of one unit and re-opens it at the start of the next, the markers counted in the budget; the re-opening marker is context before `own_start`, and a unit whose own text gained the closing one is not `verbatim`. Tracked deletions (`<w:del>`) are still dropped by mammoth. `SCHEME_VERSION` is not bumped.
- ~~**Struck-through text is dropped in mammoth, not in Markdown.** markitdown turns `<w:strike/>` into `<s>` and then `~~text~~`, but a literal `~~` typed in the document arrives identically, so a Markdown regex would delete real text. `<w:dstrike/>` (double strikethrough) arrives as plain text because mammoth ignores it; it is rewritten to `<w:strike` before conversion. Tracked deletions (`<w:del>`) are already dropped by mammoth.~~
- **Workers embed with the pipeline's own embedder, pickled** (plan 1 Task 3 gives `Embedder` its `threads` and pickling), never one built from the configuration: the pipeline's embedder is the one for the index's recorded scheme, so a scoped sync on a `rebuild pending` index and a rebuild's re-ingest embed exactly as the parent would, and each copy carries the parent's expected probe (plan 1 Task 10) and checks it before its first embedding.
- **The probe is armed before the workers start.** The parallel path writes through `Store.replace_sources`, which `before_store` does not wrap, so the parent calls `pipeline.before_store()` once (when set) before handing the embedder to the workers: on a fresh index that records the probe and arms `expect_probe`. A rebuild's pipeline has no hook; the rebuild arms its new embedder itself.
- **Workers never fall back in-process** (coordinator decision): the worker count is sized for the endpoint, and N workers each loading e5 (~3.3 GB) would exhaust memory. ~~In a worker, an unreachable embedding endpoint fails that file with the reason `"embedding endpoint unreachable: <url>"`; after 3 consecutive such failures the parent stops dispatching, marks the remaining files as not synced with the same reason, and the sync result carries one `"embedding: endpoint unreachable, N files not synced"` error.~~ **Amended 2026-11-20 (plan 2 final review; rulings C12, Task 7):** in a worker, an embedding endpoint that fails (unreachable, timed out, or a bad reply: any `EndpointError`) fails that file with the error's own message, which names the endpoint URL and the cause; after 3 consecutive such failures the parent stops dispatching, counts the files it never handed out in `failed` without an error per file, and the sync result carries one `"embedding: endpoint unreachable, N files not synced"` error. The next sync picks them up (hashes unchanged). Workers get the pickled embedder with fallback disabled (`Embedder.allow_fallback = False`, a plan-1 attribute this plan adds if plan 1 did not). The serial (single-process) path keeps plan 1's loud in-process fallback.
- **Contract changes** (consumers named): `parallel.prepare_files` + `parallel.BatchWriter` (sync.py only). New: `Config.sync_workers` (plan 3 maps a `sync_workers` per-index key), `Store.replace_sources`, `Pipeline.prepare/prepare_file/records/store_prepared` + `PreparedDocument`, `Config.ocr_lang` now in Tesseract spelling (plan 3's `ocr_lang` key). Plan 1's rebuild re-ingests through `_run_sync_unlocked`, so it gets parallel ingest for free.

---

### Task 1: Struck-through `.docx` text is dropped at parse time

> **Superseded (owner, 2026-11-19):** struck-through text is still text — keep it. Instead of
> dropping it: keep markitdown's `~~...~~`, make the markers survive chunk splits (a chunk
> boundary inside a struck run closes `~~` at the end of one chunk and reopens it at the start
> of the next), and add one line to the server instructions (`~~...~~` marks deleted text;
> never cite it as current). Tests: a struck run split across two chunks yields balanced
> markers in both; a struck run is still found by keyword search. See bd minirag-mcp-g7q.12.

**Files:**
- Modify: `src/minirag_mcp/ingest/parser.py` (`parse_file`, the markitdown branch)
- Test: `tests/test_parser.py`

**Interfaces:**
- Produces: `DOCX_STYLE_MAP = "strike => !"`; `_docx_stream(path: Path) -> io.BytesIO | None` (the file with `<w:dstrike` rewritten to `<w:strike` in every `word/*.xml` part, or None when no part holds one or the zip cannot be read).

- [ ] **Step 1: Write the failing tests** (a minimal docx built with `zipfile`, no new dependency)

```python
_CT = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
       '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
       '<Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" '
       'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
_RELS = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
         '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
         'Target="word/document.xml"/></Relationships>')

def _r(text, props=""): return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{text}</w:t></w:r>'
def _p(*runs): return "<w:p>" + "".join(runs) + "</w:p>"
def _docx(tmp_path, body, name="spec.docx"):  # writes [Content_Types].xml, _rels/.rels, word/document.xml
    ...

def test_docx_struck_through_runs_are_dropped(tmp_path):
    body = (_p(_r("Keep this. "), _r("OLDCLAUSE 4.2", "<w:strike/>"), _r(" and the tail."))
            + "<w:tbl><w:tr><w:tc>" + _p(_r("cell kept")) + "</w:tc><w:tc>"
            + _p(_r("CELLSTRUCK", "<w:strike/>")) + "</w:tc></w:tr></w:tbl>"
            + _p(_r("WHOLEPARA", "<w:b/><w:strike/>")))
    md = parse_file(_docx(tmp_path, body)).markdown
    assert "Keep this." in md and "the tail." in md and "cell kept" in md
    for gone in ("OLDCLAUSE", "CELLSTRUCK", "WHOLEPARA", "~~"):
        assert gone not in md

def test_docx_double_struck_runs_are_dropped(tmp_path):
    md = parse_file(_docx(tmp_path, _p(_r("kept "), _r("DOUBLESTRUCK", "<w:dstrike/>")))).markdown
    assert "kept" in md and "DOUBLESTRUCK" not in md

def test_docx_literal_tildes_and_switched_off_strikes_are_kept(tmp_path):
    body = _p(_r("Literal ~~tildes~~ stay")) + _p(_r("off", '<w:strike w:val="0"/>'), _r(" false", '<w:strike w:val="false"/>'))
    md = parse_file(_docx(tmp_path, body)).markdown
    assert "Literal ~~tildes~~ stay" in md and "off false" in md

def test_a_docx_that_is_not_a_zip_still_fails_to_convert(tmp_path):
    bad = tmp_path / "bad.docx"; bad.write_bytes(b"not a zip")
    with pytest.raises(ParserError, match=r"Failed to convert .*bad\.docx"):
        parse_file(bad)
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_parser.py -k docx -v`
Expected: the first two FAIL (`~~OLDCLAUSE 4.2~~` and `DOUBLESTRUCK` are in the Markdown); the last two PASS today and guard the fix (a Markdown regex would break the tildes test).

- [ ] **Step 3: Implement in `parse_file`**

For suffix `.docx` only: `stream = _docx_stream(path)`; with a stream, `_md().convert_stream(stream, stream_info=StreamInfo(extension=".docx", filename=path.name), style_map=DOCX_STYLE_MAP)`, otherwise `_md().convert(str(path), stream_info=..., style_map=DOCX_STYLE_MAP)`. Every other format is called exactly as today. `_docx_stream` scans `word/*.xml` parts for `b"<w:dstrike"` first and copies the zip (same `ZipInfo`s) only when one has it; `zipfile.BadZipFile`/`OSError` returns None so markitdown reports the failure as today. Constraint comment: mammoth merges a style map into its default one, and `!` drops the run; mammoth has no `dstrike` support.

- [ ] **Step 4: Run** `uv run pytest tests/test_parser.py -v` then the gate. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/ingest/parser.py tests/test_parser.py
git commit -m "fix(parser): drop struck-through .docx text at parse time"
```

Record on bd issue `minirag-mcp-g7q.12` that the struck-through item is done (the other items stay gated on the ruler).

---

### Task 2: Tesseract languages and pinned traineddata

**Files:**
- Modify: `src/minirag_mcp/ocr.py` (additions only; nothing calls them yet)
- Test: `tests/test_ocr.py`

**Interfaces:**
- Produces (values verified 2026-11-19 by downloading that commit; identical to the files the OCR spike measured):

```python
TESSDATA_COMMIT = "87416418657359cb625c412a48b6e1d6d41c29bd"   # tessdata_fast main, 2024-08-01
TESSDATA_SHA256 = {
    "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",  # 4,113,088 bytes
    "osd": "9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff",  # 10,562,727 bytes
    "rus": "e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225",  # 3,861,738 bytes
}
_TESSDATA_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{commit}/{name}.traineddata"
DEFAULT_LANGUAGES = "rus+eng"
LANGUAGE_ALIASES = {"eslav": "rus+eng", "en": "eng"}

def tesseract_languages(value: str) -> str        # normalized "rus+eng"; ValueError naming rus, eng, eslav, en
def tessdata_dir(cache_dir: Path) -> Path         # cache_dir / "ocr" / f"tessdata_fast-{TESSDATA_COMMIT[:12]}"
def ensure_tessdata(cache_dir: Path, languages: str) -> Path   # the dir; OcrError on any failure
```

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("value,expected", [("rus+eng", "rus+eng"), ("eng", "eng"),
    (" ENG + rus ", "eng+rus"), ("eslav", "rus+eng"), ("en", "eng")])
def test_tesseract_languages(value, expected):
    assert ocr.tesseract_languages(value) == expected

@pytest.mark.parametrize("value", ["", "ch", "deu", "rus+", "rus+rus", "osd"])
def test_tesseract_languages_refuses(value):
    with pytest.raises(ValueError, match=r"rus.*eng.*eslav"):
        ocr.tesseract_languages(value)
```

Fixture `tessdata_server(monkeypatch)`: a stdlib `ThreadingHTTPServer` on 127.0.0.1 serving `files: dict[path, bytes]` and recording `hits`; it sets `ocr._TESSDATA_URL` to `f"http://127.0.0.1:{port}/{{commit}}/{{name}}.traineddata"` and `ocr.TESSDATA_SHA256` to the digests of `f"{name} data".encode()`, served at `/{TESSDATA_COMMIT}/{name}.traineddata`.

```python
def test_tessdata_is_fetched_once_from_the_pinned_commit(tmp_path, tessdata_server):
    d = ocr.ensure_tessdata(tmp_path, "rus+eng")
    assert d == ocr.tessdata_dir(tmp_path) and d.parent == tmp_path / "ocr"
    assert sorted(p.name for p in d.iterdir()) == ["eng.traineddata", "osd.traineddata", "rus.traineddata"]
    assert len(tessdata_server.hits) == 3
    assert all(h.startswith(f"/{ocr.TESSDATA_COMMIT}/") for h in tessdata_server.hits)
    ocr.ensure_tessdata(tmp_path, "rus+eng")
    assert len(tessdata_server.hits) == 3

def test_only_the_languages_asked_for_and_osd_are_fetched(tmp_path, tessdata_server):
    ocr.ensure_tessdata(tmp_path, "eng")
    assert sorted(h.rsplit("/", 1)[1] for h in tessdata_server.hits) == ["eng.traineddata", "osd.traineddata"]

def test_a_download_with_the_wrong_digest_is_discarded(tmp_path, tessdata_server):
    tessdata_server.files[f"/{ocr.TESSDATA_COMMIT}/rus.traineddata"] = b"tampered"
    with pytest.raises(ocr.OcrError, match=r"rus\.traineddata.*sha256"):
        ocr.ensure_tessdata(tmp_path, "rus")
    assert [p.name for p in ocr.tessdata_dir(tmp_path).iterdir() if p.name.startswith(("rus", ".rus"))] == []

def test_a_cached_file_with_the_wrong_digest_is_refused(tmp_path, tessdata_server):
    d = ocr.tessdata_dir(tmp_path); d.mkdir(parents=True); (d / "osd.traineddata").write_bytes(b"x")
    with pytest.raises(ocr.OcrError, match="delete it to download it again"):
        ocr.ensure_tessdata(tmp_path, "eng")

def test_an_unreachable_host_is_a_loud_ocr_error(tmp_path, monkeypatch):
    # _TESSDATA_URL pointed at a port that was bound and then closed
    with pytest.raises(ocr.OcrError, match=r"could not download (eng|osd)\.traineddata .*127\.0\.0\.1"):
        ocr.ensure_tessdata(tmp_path, "eng")
```

- [ ] **Step 2: Run** `uv run pytest tests/test_ocr.py -k "tessdata or tesseract_languages" -v`. Expected: FAIL (`AttributeError: ... tesseract_languages`).

- [ ] **Step 3: Implement in `ocr.py`**

`tesseract_languages`: `strip().lower()`, then an alias maps directly; otherwise split on `+` with each part stripped; every part must be a key of `TESSDATA_SHA256` other than `osd`, no empty part, no repeat. Refusal text: `f"{value!r} is not a language set OCR can read: use rus, eng or both joined by '+' (rus+eng), or the 1.x names eslav, en"`. `ensure_tessdata`: names = the languages plus `osd`; an existing file is checked against its digest once per process (cache keyed by path, size, `st_mtime_ns`, as `labels/model._sha256` does; hash in 1 MiB blocks); a missing one is fetched with `requests.get(url, stream=True, timeout=60)` into a `NamedTemporaryFile(dir=<tessdata dir>, prefix=f".{name}.", delete=False)`, hashed while written, and `os.replace`d into place only when the digest matches. Constraint comment: several sync workers may fetch at once, and the replace is what keeps any of them from reading a partial file. Messages: `could not download {name}.traineddata from {url}: {e}; OCR needs it once, then works offline`, `{url} has sha256 {actual}, expected {expected}; the download was discarded`, `{path} has sha256 {actual}, expected {expected}; delete it to download it again`. Any exception removes the temp file.

- [ ] **Step 4: Run** the tests and the gate. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/ocr.py tests/test_ocr.py
git commit -m "feat(ocr): Tesseract language codes and sha256-pinned tessdata_fast download"
```

---

### Task 3: OCR through Tesseract, in its own reading order

**Files:**
- Modify: `src/minirag_mcp/ocr.py` (replace `available`, `_engine`, `_recognize`, `ocr_pdf`, `ocr_image`; delete `ENGINE_RAPIDOCR`, `assemble_lines`, `_orient`, `_orientation`, `_orientation_engine`, the numpy/cv2 code)
- Modify: `src/minirag_mcp/ingest/parser.py:451,534`, `src/minirag_mcp/results.py:243` (`ENGINE_TESSERACT`)
- Modify: `src/minirag_mcp/config.py` (`RAG_OCR_LANG`)
- Modify: `pyproject.toml` (`[ocr]`), `uv.lock`, `.github/workflows/ci.yml` (slow job), `README.md` (OCR section, `RAG_OCR_LANG` row)
- Test: `tests/test_ocr.py`, `tests/test_config.py:110-128`, `tests/test_parser.py:388,473,493,510`, `tests/test_cli.py:152,161,496`, `tests/test_server.py:210,376` (`"rapidocr"` -> `"tesseract"`; `tests/test_scanner.py` keeps `"rapidocr"` as stored data)

**Interfaces:**
- Consumes: Task 2 (`tesseract_languages`, `ensure_tessdata`, `DEFAULT_LANGUAGES`).
- Produces: `ENGINE_TESSERACT = "tesseract"`; `DPI = 300`; `_engine(config) -> tuple[PyTessBaseAPI, threading.Lock]` cached per `(config.ocr_lang, tessdata dir)`; `_recognize(image: PIL.Image.Image, config, path, *, dpi: int | None) -> str`; `ocr_pdf`/`ocr_image` keep their signatures; `Config.ocr_lang` holds Tesseract spelling (default `"rus+eng"`).

- [ ] **Step 1: Write the two-column test and watch it fail on today's engine**

The `.venv` currently has the RapidOCR extra (`uv run python -c "import rapidocr"` succeeds; otherwise `uv sync --extra ocr` before the code changes).

```python
@pytest.mark.slow
@pytest.mark.skipif(not ocr.available(), reason="needs the [ocr] extra")
def test_real_engine_keeps_the_columns_of_a_two_column_page_apart(tmp_path):
    # Image.new("L", (2480, 1400), 255); ImageFont.load_default(size=44); for i in range(10):
    #   left  "Left column line {100+i} reads here" at (150, 150 + 100*i)
    #   right "Right side entry {500+i} goes on"   at (1340, 150 + 100*i); saved with dpi=(300, 300)
    text = ocr.ocr_image(p, load_config({}, cwd=tmp_path))
    assert all(f"{100 + i}" in text and f"{500 + i}" in text for i in range(10))
    assert text.index("109") < text.index("500")
```

Run: `uv run pytest -m slow tests/test_ocr.py::test_real_engine_keeps_the_columns_of_a_two_column_page_apart -v`
Expected: FAIL — measured 2026-11-19, today's path returns `Left column line 100 reads here Right side entry 500 goes on` as one line (g7q.14).

- [ ] **Step 2: Replace the fast tests**

Delete the `assemble_lines`, cv2, rapidocr-import and orientation tests. Add a fake module: `_FakeTesserocr` with `PSM = SimpleNamespace(AUTO_OSD=1)` and `PyTessBaseAPI(**kw)` recording `kw` in `created`; its API records each call name in `calls`, `SetSourceResolution` values in `resolutions`, the images in `images`, and `GetUTF8Text` returns `reply(image)` (default: next of `texts`). Fixture `fake_tess(monkeypatch, tmp_path)` installs it in `sys.modules`, sets `ocr.available` -> True, `ocr.ensure_tessdata` -> `lambda cache_dir, langs: tmp_path / "td"`, and clears `ocr._engines`. Pillow is a core dependency, so image files are real.

```python
def test_the_engine_reads_with_auto_osd_and_the_configured_languages(fake_tess, tmp_path):
    ocr._engine(load_config({"RAG_OCR_LANG": "eslav"}, cwd=tmp_path))
    assert fake_tess.created == [{"path": str(tmp_path / "td"), "lang": "rus+eng", "psm": 1}]

def test_recognize_keeps_tesseracts_own_order(fake_tess, tmp_path):
    fake_tess.texts = ["Left 1\nLeft 2\n\nRight 1\nRight 2\n"]
    out = ocr._recognize(Image.new("L", (8, 8), 255), cfg, tmp_path / "p.png", dpi=None)
    assert out == "Left 1\nLeft 2\n\nRight 1\nRight 2"

def test_pdf_pages_render_at_300_dpi_and_say_so(fake_tess, monkeypatch, tmp_path):
    # _FakePdfium page.render(scale) records scale; bitmap.to_pil() -> Image.new("RGB", (8, 8), "white")
    ocr.ocr_pdf(p, cfg, [0])
    assert scales == [300 / 72] and api.resolutions == [300]
    assert api.calls == ["SetImage", "SetSourceResolution", "GetUTF8Text"]

def test_every_frame_of_a_multi_page_tiff_is_read(fake_tess, tmp_path):
    # three frames 100x10, 101x10, 102x10 saved with save_all=True; reply = lambda im: f"w{im.width}"
    assert ocr.ocr_image(tiff, cfg) == "w100\n\nw101\n\nw102"

def test_an_image_file_keeps_its_recorded_dpi(fake_tess, tmp_path):
    # PNG saved with dpi=(200, 200) -> resolutions == [200]; one saved without dpi -> []

def test_an_image_with_transparency_is_read_on_white(fake_tess, tmp_path):
    # RGBA 20x20 all (0, 0, 0, 0) except (5, 5) = (0, 0, 0, 255); also a "P" image with transparency
    ocr.ocr_image(png, cfg)
    seen = api.images[0]
    assert seen.mode == "RGB" and seen.getpixel((0, 0)) == (255, 255, 255) and seen.getpixel((5, 5)) == (0, 0, 0)

def test_two_recognitions_at_once_never_share_the_engine(fake_tess, tmp_path):
    # SetImage raises AssertionError if another SetImage..GetUTF8Text is in progress, then sleeps 0.05 s;
    # two threads call _recognize; assert both returned and no thread recorded an exception

def test_engine_failure_names_the_file(fake_tess, tmp_path):     # GetUTF8Text raises RuntimeError("tesseract aborted")
    with pytest.raises(ocr.OcrError, match=r"scan\.png.*tesseract aborted"): ...

def test_available_needs_tesserocr(monkeypatch):                  # find_spec -> None for "tesserocr"
    assert ocr.available() is False
```

Keep and adapt: `test_available_is_bool`, `test_pdf_page_texts_counts_text_and_blank_pages`, `test_ocr_image_without_extra_raises_loudly`, the three pdfium tests (`_FakeBitmap.to_pil()` instead of `to_numpy()`), `test_ocr_pdf_broken_pypdfium2_install_raises_ocr_error`; rename the rapidocr/cv2 broken-install tests to `tesserocr` and `PIL`; `test_ocr_image_undecodable_bytes_raise` matches `r"could not read or decode.*broken\.png"` (real Pillow, garbage bytes); `test_ocr_image_unreadable_file_says_reading_failed` unchanged; add `test_ocr_image_reads_a_non_ascii_path` (`Скан договора.png`).

Slow tests rewritten with Pillow and `skipif(not ocr.available())` instead of `importorskip("rapidocr")`: two-frame TIFF (`FIRST PAGE 111` / `SECOND PAGE 222`, 3 lines each), generated page `INVOICE 12345 TOTAL 999`, rendered PDF (`pdf_builder`, `"OCR PDF CHECK 555 " * 5`), and a rotated page: 12 lines `The quick brown fox number {700+i} jumps over the lazy dog again` on 2480x1400, rotated 90 and 180 with `expand=True`, each must contain `700` and `711` (measured: AUTO_OSD reads both; a 3-line page at 180 degrees does not, orientation needs a page's worth of text).

`tests/test_config.py`: default `ocr_lang == "rus+eng"`; `RAG_OCR_LANG=eslav` -> `"rus+eng"`; `eng` -> `"eng"`; `ch` and blank -> `ConfigError` matching `r"RAG_OCR_LANG.*rus.*eng"`.

- [ ] **Step 3: Run** `uv run pytest tests/test_ocr.py tests/test_config.py -q`. Expected: FAIL (no Tesseract path yet).

- [ ] **Step 4: Implement**

- `available()`: `find_spec` of `tesserocr` and `pypdfium2`.
- `_engine`: `td = ensure_tessdata(config.cache_dir, config.ocr_lang)`; `tesserocr.PyTessBaseAPI(path=str(td), lang=config.ocr_lang, psm=tesserocr.PSM.AUTO_OSD)`; import failure -> `_import_error_as_ocr_error`; init failure -> `OcrError("OCR engine failed to initialize: ...")`.
- `_recognize`: under the handle's lock, `SetImage`, then `SetSourceResolution(dpi)` when given, then `GetUTF8Text()`; any exception -> `OcrError(f"OCR engine failed on {path}: {e}")`; returns the text stripped, never reordered. Constraint comments: the handle is not thread-safe; `SetImage` resets the resolution; never call `DetectOrientationScript` on this handle — on one initialised without `osd` as its language it aborts the process (`std::out_of_range`, observed 2026-11-19).
- `ocr_pdf`: `page.render(scale=DPI / 72).to_pil()` -> `_recognize(..., dpi=DPI)`; the existing open/lookup/render guards stay.
- `ocr_image`: `PIL.Image.open(path)` and `ImageSequence.Iterator`, frame by frame; frames in mode `1`, `L`, `RGB` go as they are, any other mode is converted to RGBA and alpha-composited onto white, then RGB; a frame's `info["dpi"][0]` rounded is passed when >= 70, else None (Tesseract estimates). Open/decode failures -> `OcrError(f"could not read or decode image {path}: {e}")`; texts of non-empty frames joined with `"\n\n"`.
- `parser.py`/`results.py`: `ocr.ENGINE_TESSERACT`. `config.py`: `ocr_lang = tesseract_languages(env.get("RAG_OCR_LANG", DEFAULT_LANGUAGES))`, `ValueError` -> `ConfigError(f"RAG_OCR_LANG: {e}")`; field default `"rus+eng"`.
- `pyproject.toml`:

```toml
ocr = [
    # Wheels bundle libtesseract 5.5.4 for macOS 15+, manylinux, musllinux and Windows;
    # anywhere else pip builds it from source against a system Tesseract.
    "tesserocr>=2.13",
    # pdfplumber already pulls these in, but we import them directly: PDF pages are
    # rendered with pypdfium2 and image files decoded with Pillow. Both bounds are at or
    # below floors already imposed, so declaring them can never move the resolution.
    "pypdfium2>=5",
    "pillow>=10.3",
]
```

- `ci.yml`: the `slow` job runs `uv sync --extra ocr` (first real tesserocr run on Linux; the fast job stays without the extra).
- `README.md` OCR section: Tesseract (tesserocr wheels, libtesseract 5.5.4; macOS 15+, Linux, Windows; older macOS builds from source and needs a system Tesseract); ~9 MB of wheels instead of "roughly 160 MB"; `rus`/`eng`/`osd` traineddata (18.5 MB) fetched once into `CACHE_DIR/ocr`, pinned by commit and sha256, offline afterwards; text in Tesseract's own reading order, so two-column pages keep their columns; orientation by Tesseract's OSD; `ocrEngine`/`[ocr:tesseract]`/`status.ocr` read `"tesseract"`; the accuracy paragraph replaced by the spec's aggregate (83 hand-checked lines on real scans: Tesseract 83/83, CER 0.001; RapidOCR dropped 12, one a contract sum; weakest on a poor fax scan, word recall 0.81), keeping "OCR text is not authoritative over the source scan". `RAG_OCR_LANG` row: default `rus+eng`, any `+`-join of `rus`, `eng`; `eslav`/`en` accepted as their 1.x meanings; anything else is an error naming the accepted values.

- [ ] **Step 5: Lock, sync both ways, run everything**

```bash
uv lock
grep -c -E '^name = "(rapidocr|rapid-orientation|opencv-python)"' uv.lock   # 0
grep -A1 '^name = "tesserocr"' uv.lock                                       # version = "2.13.0"
uv sync --exact && uv run python -c "import tesserocr"                       # ModuleNotFoundError
uv run pytest -q                                                             # PASS without the extra
uv sync --extra ocr && uv run --extra ocr pytest -q                          # PASS with it
uv run --extra ocr pytest -m slow tests/test_ocr.py -v                       # PASS, incl. the two-column test
uv sync --exact && uv run pytest -q                                          # PASS, extra gone again
uv run ruff check src tests && uv run ruff format --check src tests
grep -rn -i -E "rapidocr|assemble_lines|cv2|rapid_orientation" src README.md pyproject.toml .github   # no hits
```

The slow run downloads 18.5 MB of traineddata into the user cache on first use.

- [ ] **Step 6: Commit**

```bash
git add src/minirag_mcp/ocr.py src/minirag_mcp/ingest/parser.py src/minirag_mcp/results.py \
  src/minirag_mcp/config.py pyproject.toml uv.lock .github/workflows/ci.yml README.md tests/
git commit -m "feat(ocr): Tesseract in its own reading order instead of RapidOCR"
bd close minirag-mcp-g7q.9
bd close minirag-mcp-g7q.14
```

---

### Task 4: Pipeline in two halves

**Files:**
- Modify: `src/minirag_mcp/ingest/pipeline.py` (`_chunk_and_store` is removed; `ingest_file`, `ingest_data`, `ingest_url` and plan 1's `reingest_stored` use the halves)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: plan 1's `Pipeline(..., before_store=)`, `_chunk` (prefix room), `reingest_stored`.
- Produces:

```python
@dataclass(frozen=True)
class PreparedDocument:
    source: str
    source_type: str
    title: str
    chunks: tuple[Chunk, ...]
    vectors: tuple[Sequence[float], ...]
    pages: tuple[str, ...]
    file_hash: str = ""
    mtime: float = 0.0
    ocr_engine: str = ""
    file_time: float | None = None
    embedding_warning: str | None = None     # the embedder's `warning` after embedding this document  [SUPERSEDED 2026-11-20, ruling C10: dropped, see below]

class Pipeline:
    def prepare(self, markdown: str, *, source: str, source_type: str, title: str, seed_title: bool,
                file_hash: str = "", mtime: float = 0.0, ocr_engine: str = "",
                file_time: float | None = None, page_starts: pages.PageStarts = ()) -> PreparedDocument
    def prepare_file(self, path: Path, *, rules_checked: bool = False) -> PreparedDocument
    def records(self, doc: PreparedDocument) -> tuple[list[ChunkRecord], str | None]
    def store_prepared(self, doc: PreparedDocument) -> IngestResult
```

`prepare`/`prepare_file` (ignore check, `_checked_stat`, parse, hash, chunk, page-map, embed) never touch `self.store` or `self.labels`; ~~`embedding_warning` = `getattr(self.embedder, "warning", None)` read after embedding~~ (superseded 2026-11-20, ruling C10: `PreparedDocument` has no `embedding_warning`; workers never fall back, so it would always be None). `records` labels inside the active job and stamps `ingested_at` now. `store_prepared` calls `self.before_store()` (when set) and then `store.replace_source`, as plan 1's write path does. `ingest_file` = `prepare_file`, then `with self.job(): return self.store_prepared(doc)`; `ingest_data`/`ingest_url`/`reingest_stored` = `with self.job(): return self.store_prepared(self.prepare(...))`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prepare_file_needs_neither_a_store_nor_labels(tmp_path, fake_embedder):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    f = tmp_path / "doc.md"; f.write_text("# T\n\n" + "Sentence about topic. " * 80)
    doc = Pipeline(None, fake_embedder, cfg).prepare_file(f)
    assert len(doc.chunks) == len(doc.vectors) == len(doc.pages) > 1
    assert (doc.source, doc.source_type, doc.file_hash, doc.mtime) == (str(f), "file", file_sha256(f), f.stat().st_mtime)

def test_a_prepared_document_survives_pickling(tmp_path, fake_embedder):
    assert pickle.loads(pickle.dumps(doc)) == doc

def test_ingest_file_stores_exactly_what_was_prepared(pipe):
    # [c.text for c in store.all_chunks(src)] == [c.text for c in doc.chunks]; stored vectors == doc.vectors

def test_a_worker_embedder_never_falls_back(tmp_path, dead_url):
    # Embedder(endpoint=EmbeddingEndpoint(Endpoint(dead_url, "m"))), allow_fallback=False
    # prepare_file raises EndpointError naming the URL; no fastembed model is loaded
```

- [ ] **Step 2: Run** `uv run pytest tests/test_pipeline.py -v`. Expected: FAIL (`AttributeError: prepare_file`).
- [ ] **Step 3: Implement** the split; no behaviour change.
- [ ] **Step 4: Run** the gate; `tests/test_pipeline*.py`, `tests/test_sync.py`, `tests/test_rebuild.py`, `tests/test_handle.py` and `tests/test_parity.py` must pass unchanged (among them plan 1's `test_the_store_hook_runs_before_every_write`, `test_chunks_leave_room_for_the_passage_prefix` and `test_a_reingest_keeps_what_the_disk_cannot_give_back`).
- [ ] **Step 5: Commit** `refactor(pipeline): separate preparing a document from storing it`.

---

### Task 5: `Store.replace_sources` — many sources, one commit

**Files:**
- Modify: `src/minirag_mcp/store.py` (`replace_source` delegates for non-empty records)
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: plan 1's `Store._document_row`, `Store._write_documents`, `IndexBusyError` and `identity.rebuild_running`.
- Produces: `Store.replace_sources(batch: Sequence[tuple[str, Sequence[ChunkRecord]]], *, indexed: Container[str]) -> None`.
  First, like every write plan 1 guards, it raises `IndexBusyError` while `rebuild_running(self.db_path)`. Sources in `indexed` go through one `merge_insert(["id", "ingested_at"])` with `when_not_matched_by_source_delete(_in_clause(those sources))`; the rest through one `add`. The caller vouches that those hold no rows; if a racing single-file ingest added some meanwhile, the source ends up doubled, which `list_sources` flags and the next sync repairs (the race `lock.py` documents). After the chunk write, one `self._write_documents([self._document_row(source, records) for each source])` writes every source's `documents` row (plan 1: one merge on `["source", "ingested_at"]` that drops the sources' other rows; warns, never raises). Signatures for all sources go through one `_put_signatures` call, best effort with the same warning as `_sign`. Additive missing columns are dropped as in `replace_source`. A batch entry with no records, or a record whose `source` differs from its key, is a `ValueError`. `replace_source(source, records)` with records = `replace_sources([(source, records)], indexed={source})`; its empty-records branch (chunks deleted, `_delete_documents([source])`) and its own busy check stay.

- [ ] **Step 1: Write the failing tests**

```python
def test_replace_sources_writes_new_sources_in_one_commit(store):
    v = store._table.version
    store.replace_sources([("/a.md", [rec("/a.md", 0, "a")]), ("/b.md", [rec("/b.md", 0, "b"), rec("/b.md", 1, "c")])], indexed=set())
    assert store._table.version == v + 1
    assert sorted(s.source for s in store.list_sources()) == ["/a.md", "/b.md"]

def test_replace_sources_drops_the_stale_chunks_of_an_indexed_source(store):
    store.replace_source("/a.md", [rec("/a.md", i, f"old {i}") for i in range(5)])
    store.replace_sources([("/a.md", [rec("/a.md", i, f"new {i}") for i in range(2)]),
                           ("/b.md", [rec("/b.md", 0, "b")])], indexed={"/a.md"})
    assert [c.text for c in store.all_chunks("/a.md")] == ["new 0", "new 1"]
    assert not store.get_source("/a.md").duplicated

def test_replace_sources_writes_what_replace_source_writes(tmp_path):
    # same records into two stores, per source vs one batch: chunk rows (all columns, sorted by id),
    # signature rows and plan 1's documents rows (every column, ingested_at included: same records) are equal

def test_replace_sources_writes_one_document_row_per_source_in_one_call(store, monkeypatch):
    # spy on store._write_documents: one call with the rows of /a.md and /b.md; documents table has one row each

def test_replace_sources_is_refused_while_a_rebuild_runs(store):
    # staging_dir(store.db_path).mkdir(); with sync_lock(store.db_path): replace_sources(...) raises IndexBusyError,
    # chunk_count unchanged (plan 1's test_writes_are_refused_while_a_rebuild_runs sets up the same condition)

def test_replace_sources_signs_every_source(store):
    assert store.unsigned_sources() == []

def test_replace_sources_writes_an_index_that_cannot_gain_the_page_column(tmp_path, monkeypatch):
    # set up as test_an_index_that_cannot_gain_the_page_column_still_searches; the batch is written
```

- [ ] **Step 2: Run** `uv run pytest tests/test_store.py -k replace_sources -v`. Expected: FAIL (`AttributeError`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the gate (every `tests/test_store*.py` passes unchanged).
- [ ] **Step 5: Commit** `feat(store): write many sources in one commit`.

---

### Task 6: How many workers — `RAG_SYNC_WORKERS`

**Files:**
- Create: `src/minirag_mcp/parallel.py`
- Modify: `src/minirag_mcp/config.py`
- Test: `tests/test_parallel.py`, `tests/test_config.py`

**Interfaces:**
- Produces:

```python
# parallel.py
FILES_PER_WORKER = 4
WORKER_BASE_BYTES = 3 * 2**29            # 1.5 GiB: interpreter and libraries, measured (see Decisions)
def physical_cores() -> int              # macOS: `sysctl -n hw.physicalcpu`; Linux: distinct (physical id, core id)
                                         # in /proc/cpuinfo; else os.cpu_count(); never above the CPUs this
                                         # process may use (os.sched_getaffinity where it exists); at least 1
def _linux_physical_cores(cpuinfo: str) -> int | None
def total_memory() -> int | None         # SC_PHYS_PAGES * SC_PAGE_SIZE; None when unavailable
def worker_footprint(embedder) -> int    # WORKER_BASE_BYTES, plus int(fastembed's size_in_GB * 10**9) for the model
                                         # unless getattr(embedder, "backend", "") == "endpoint" or it is unlisted
def default_workers(cores: int, memory: int | None, footprint: int) -> int
                                         # max(1, cores - 1), capped at max(1, memory // 2 // footprint)
def worker_count(configured: int | None, files: int, *, default: int) -> int
                                         # max(1, min(configured or default, files // FILES_PER_WORKER))
def worker_threads(workers: int) -> int  # max(1, physical_cores() // workers)

# config.py
Config.sync_workers: int | None = None   # env RAG_SYNC_WORKERS, integer in [1, 256]; None = default_workers(...)
```

> **Amended 2026-11-20 (plan 2 final review, ruling I2):** `total_memory()` reads physical
> memory on Windows with `GlobalMemoryStatusEx` (ctypes), where `os.sysconf` does not exist;
> when memory is still unknown and the footprint holds the model (it runs in-process in every
> worker), `default_workers` is at most `UNKNOWN_MEMORY_WORKERS = 2`. The test row
> `(8, None, int(3.6 * GiB), 7)` below is superseded: it is 2, and `cores - 1` stays only for
> an endpoint's footprint.

- [ ] **Step 1: Write the failing tests**

```python
GiB = 2**30

@pytest.mark.parametrize("cores,memory,footprint,expected", [
    (18, 64 * GiB, int(1.5 * GiB), 17),    # endpoint: the cores decide
    (18, 64 * GiB, int(3.6 * GiB), 8),     # in-process e5: memory decides
    (8, 16 * GiB, int(3.6 * GiB), 2),
    (8, None, int(3.6 * GiB), 7),          # memory unknown
    (1, 64 * GiB, GiB, 1),
    (4, GiB, 4 * GiB, 1),
])
def test_default_workers(cores, memory, footprint, expected):
    assert parallel.default_workers(cores, memory, footprint) == expected

@pytest.mark.parametrize("configured,files,default,expected",
    [(None, 3, 8, 1), (None, 100, 8, 8), (2, 12, 8, 2), (16, 20, 8, 5), (4, 7, 8, 1), (1, 1000, 8, 1)])
def test_worker_count(configured, files, default, expected):
    assert parallel.worker_count(configured, files, default=default) == expected

def test_worker_footprint_counts_the_model_only_when_it_runs_in_process(fake_embedder):
    e5 = SimpleNamespace(model_name="intfloat/multilingual-e5-large", backend="in-process")
    assert parallel.worker_footprint(e5) == parallel.WORKER_BASE_BYTES + 2_240_000_000
    e5.backend = "endpoint"
    assert parallel.worker_footprint(e5) == parallel.WORKER_BASE_BYTES
    assert parallel.worker_footprint(fake_embedder) == parallel.WORKER_BASE_BYTES

def test_linux_physical_cores_counts_distinct_cores():
    # 8 "processor" blocks: physical id 0/1 x core id 0/1 x two hyperthreads each
    assert parallel._linux_physical_cores(SAMPLE) == 4

def test_physical_cores_is_within_the_logical_count():
    assert 1 <= parallel.physical_cores() <= os.cpu_count()
```

`tests/test_config.py`: unset -> `None`; `"4"` -> `4`; `"0"`, `"257"`, `"x"` -> `ConfigError` naming `RAG_SYNC_WORKERS`.

- [ ] **Step 2: Run** `uv run pytest tests/test_parallel.py tests/test_config.py -v`. Expected: FAIL (`ModuleNotFoundError: minirag_mcp.parallel`).
- [ ] **Step 3: Implement.** `physical_cores` never raises: a failing `sysctl` or unreadable `/proc/cpuinfo` falls back to the logical count.
- [ ] **Step 4: Run** the gate.
- [ ] **Step 5: Commit** `feat(sync): RAG_SYNC_WORKERS, defaulting to physical cores minus one within memory`.

---

### Task 7: The worker pool

**Files:**
- Modify: `src/minirag_mcp/parallel.py`
- Create: `tests/parallel_helpers.py` (spawned workers import it: pytest puts `tests/` on `sys.path`, and spawn hands the parent's `sys.path` to its children)
- Test: `tests/test_parallel.py`

**Interfaces:**
- Consumes: Task 4 (`Pipeline.prepare_file`, `PreparedDocument`); plan 1 Tasks 3 (`Embedder.threads`, pickling) and 10 (the expected probe survives pickling).
- Produces:

```python
class Preparer(Protocol):
    def prepare_file(self, path: Path, *, rules_checked: bool) -> PreparedDocument: ...

class Prepared(NamedTuple):
    path: Path
    document: PreparedDocument | None
    error: str | None

WORKER_DIED = ("the worker process reading this file stopped abruptly (a crash in a native "
               "library, or killed for memory); it was not indexed")

def worker_pipeline(config: Config, embedder, threads: int | None) -> Pipeline
    # embedder is the parent pipeline's, unpickled in the worker (its own copy);
    # sets embedder.threads = threads; returns Pipeline(None, embedder, config) (no labels, no before_store)
def prepare_files(make_preparer: Callable[[], Preparer], paths: Sequence[Path], *, workers: int) -> Iterator[Prepared]
```

Behaviour of `prepare_files`:
- `ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker, initargs=(make_preparer,))`. Constraint comment: spawn, never fork — the parent may be the MCP server with LanceDB and sync threads running.
- `_init_worker`: `os.dup2(2, 1)` first (a stdio MCP server's stdout is the protocol channel and children inherit it); `os.environ.setdefault("OMP_THREAD_LIMIT", "1")`; builds the preparer. An exception while building it is kept and returned by every task as that file's error, never raised: an initializer that raises breaks the pool, which would read as every file crashing.
- `_prepare(path: str) -> tuple[document | None, str | None]` calls `prepare_file(Path(path), rules_checked=True)`; an `Exception` becomes `str(e)` (what the serial sync records).
- ~~Results come back in input order, with at most `2 * workers` tasks in flight.~~ **Superseded 2026-11-20 (plan 2 final review, ruling I1):** results come back as each file finishes (`concurrent.futures.wait(..., return_when=FIRST_COMPLETED)`), with at most `2 * workers` tasks in flight; in input order, one slow file held back every file after it.
- On `BrokenProcessPool`: the in-flight paths are re-run one at a time in a fresh single-worker pool; a path that breaks it yields `WORKER_DIED` and gets a new single-worker pool for the next one; then a fresh full pool takes the remaining paths.
- Every pool is shut down (`wait=True, cancel_futures=True`) on exit, on an exception and when the generator is closed early.

`tests/parallel_helpers.py`:

```python
class Preparer:
    def __init__(self, *, crash_on: str = "", fail_on: str = "", shout: bool = False): ...
    def prepare_file(self, path, *, rules_checked):
        # shout: print("LEAK-print"); os.write(1, b"LEAK-fd\n")
        # path.name == crash_on: os._exit(3);  == fail_on: raise ValueError(f"cannot read {path.name}")
        return path.read_text()          # stands in for a PreparedDocument

def broken_factory():
    raise RuntimeError("no model here")
```

- [ ] **Step 1: Write the failing tests** (each spawns real processes; 20 files `00.md`..`19.md` holding `text {i}`, `workers=2`)

```python
def test_every_path_comes_back_in_input_order(tmp_path):
    got = list(parallel.prepare_files(Preparer, paths, workers=2))
    assert [g.path for g in got] == paths
    assert [g.document for g in got] == [f"text {i}" for i in range(20)] and all(g.error is None for g in got)

def test_a_file_that_raises_fails_alone(tmp_path):
    got = list(parallel.prepare_files(partial(Preparer, fail_on="05.md"), paths, workers=2))
    assert (got[5].document, got[5].error) == (None, "cannot read 05.md")
    assert all(g.error is None for i, g in enumerate(got) if i != 5)

def test_a_worker_that_dies_fails_only_its_own_file(tmp_path):
    got = list(parallel.prepare_files(partial(Preparer, crash_on="07.md"), paths, workers=2))
    assert [g.path for g in got] == paths and got[7].error == parallel.WORKER_DIED
    assert [g.document for i, g in enumerate(got) if i != 7] == [f"text {i}" for i in range(20) if i != 7]

def test_a_preparer_that_cannot_be_built_fails_each_file_with_its_reason(tmp_path):
    got = list(parallel.prepare_files(broken_factory, paths[:3], workers=2))
    assert [g.error for g in got] == ["no model here"] * 3

def test_workers_never_write_to_stdout(tmp_path, capfd):
    list(parallel.prepare_files(partial(Preparer, shout=True), paths[:4], workers=2))
    out = capfd.readouterr()
    assert "LEAK" not in out.out and "LEAK-fd" in out.err

def test_worker_pipeline_prepares_like_the_parent(tmp_path, fake_embedder):
    pipe = parallel.worker_pipeline(cfg, fake_embedder, 3)
    assert pipe.store is None and pipe.labels is None and fake_embedder.threads == 3
    assert pipe.prepare_file(f, rules_checked=True) == Pipeline(None, fake_embedder, cfg).prepare_file(f)
```

- [ ] **Step 2: Run** `uv run pytest tests/test_parallel.py -v`. Expected: FAIL (`AttributeError: prepare_files`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests, then prove two of them can fail: with the `dup2` line removed `test_workers_never_write_to_stdout` must go red; with `BrokenProcessPool` re-raised instead of isolated `test_a_worker_that_dies_fails_only_its_own_file` must go red. Restore both, run the gate.
- [ ] **Step 5: Commit** `feat(sync): worker processes that prepare files, isolating one that dies`.

---

### Task 8: `BatchWriter` — the parent's batched writes

**Files:**
- Modify: `src/minirag_mcp/parallel.py`
- Test: `tests/test_parallel.py`

**Interfaces:**
- Consumes: Task 5 (`Store.replace_sources`).
- Produces:

```python
BATCH_ROWS = 2000     # ~65 MB of 1024-dim vectors held as Python floats; ~200 commits per 400k rows
Written = tuple[str, str | None]          # (source, error); error None when stored

class BatchWriter:
    def __init__(self, store: Store, *, indexed: Container[str], batch_rows: int | None = None)  # None -> BATCH_ROWS, read here
    def add(self, source: str, records: Sequence[ChunkRecord]) -> list[Written]  # writes once >= batch_rows rows are held
    def flush(self) -> list[Written]
```

When `replace_sources` raises, the batch is written source by source with `replace_source` (idempotent: the records keep their stamps), and each source's own exception becomes its error.

- [ ] **Step 1: Write the failing tests** (no processes; `rec` as in `tests/test_store.py`)

```python
def test_batch_writer_writes_when_the_batch_is_full_and_on_flush(store):
    w = parallel.BatchWriter(store, indexed=set(), batch_rows=5)
    assert w.add("/a.md", recs("/a.md", 3)) == [] and store.chunk_count() == 0
    assert w.add("/b.md", recs("/b.md", 3)) == [("/a.md", None), ("/b.md", None)] and store.chunk_count() == 6
    assert w.add("/c.md", recs("/c.md", 1)) == []
    assert w.flush() == [("/c.md", None)] and store.chunk_count() == 7

def test_a_failed_batch_is_retried_source_by_source(store, monkeypatch):
    # store.replace_sources raises RuntimeError -> flush() returns no errors and all sources are stored

def test_a_source_that_cannot_be_written_fails_alone(store, monkeypatch):
    # replace_sources raises; replace_source raises RuntimeError("disk full") for "/b.md" only
    # -> flush() == [("/a.md", None), ("/b.md", "disk full")] and "/a.md" is stored
```

- [ ] **Step 2: Run** `uv run pytest tests/test_parallel.py -k batch -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the gate.
- [ ] **Step 5: Commit** `feat(sync): batched writes that fall back to one source at a time`.

---

### Task 9: Parallel `sync`

**Files:**
- Modify: `src/minirag_mcp/sync.py` (`_run_sync_unlocked`)
- Modify: `tests/conftest.py` (fixture `inline_workers`), `tests/test_parity.py`, `README.md` (Configuration table row)
- Create: `tests/test_sync_parallel.py`

**Interfaces:**
- Consumes: Tasks 2, 4, 5, 6, 7, 8; plan 1's `Pipeline.before_store`, the `_run_sync_unlocked` tail (Tasks 3, 7) and `rebuild.rebuild` (Task 11), which calls `_run_sync_unlocked` on the staging store.
- Produces: no new public names. `run_sync`, `_run_sync_unlocked` and `SyncManager` keep the signatures plan 1 left; the worker count comes from `pipeline.config.sync_workers`, so CLI `sync` (plan 1's `IndexHandle.run_sync`) and MCP `sync_start` read the same setting, and a rebuild's re-ingest runs in workers too.

Behaviour of `_run_sync_unlocked`:
- `listing = store.list_sources()` read once, passed to `compute_diff`; `indexed = {i.source for i in listing}`.
- Moves first, as today (including their serial read-instead fallback). Then `workers = parallel.worker_count(cfg.sync_workers, len(diff.to_ingest), default=parallel.default_workers(parallel.physical_cores(), parallel.total_memory(), parallel.worker_footprint(pipeline.embedder)))`.
- `workers == 1`: today's loop over `ingest(path)`.
- Otherwise, still inside the run's `pipeline.job()` (so `records` labels with the run's one model): `pipeline.before_store()` once when it is set (plan 1: on a fresh index this records the probe and arms `expect_probe`, which the workers' pickled copies then carry); when `ocr.available()`, `ocr.ensure_tessdata(cfg.cache_dir, cfg.ocr_lang)` once in the parent, `OcrError` ignored (each file that needs OCR then fails with its own message); `make = functools.partial(parallel.worker_pipeline, cfg, pipeline.embedder, parallel.worker_threads(workers))`; a `BatchWriter(store, indexed=indexed)`; for each `Prepared`: an error is a failed file (`errors.append({"source": str(path), "error": err})`, `emit(f"failed {path}: {err}")`); otherwise ~~its `embedding_warning`, when set, is remembered, then~~ `pipeline.records(doc)` (an exception there fails that file alone) and `writer.add`; finally `writer.flush()`. Each `Written` counts `ingested` with `emit(f"ingested {source}")`, or `failed` with its error. An `IndexBusyError` from a write fails that file like any other write error.
- Then deletes, `relabel_stale`, `backfill_signatures`, plan 1's `backfill_documents`, `update_indices`, label problems and plan 1's embedding-fallback error — unchanged, inside the same `pipeline.job()`. ~~The remembered worker warnings join that last step: one `{"source": str(store.db_path), "error": f"embedding: {warning}"}` per distinct warning, the parent's own included once (not failures).~~ (Superseded 2026-11-20, ruling C10: workers never fall back, so they have no warning to remember; the parent's own is the one reported.)
- Call `parallel.prepare_files` and `ocr.ensure_tessdata` through their modules so tests can stand them in.

`inline_workers` fixture (conftest): monkeypatches `parallel.prepare_files` with an in-process stand-in that builds `make_preparer()` once, yields a `Prepared` per path (an `Exception` -> `str(e)`), and appends each call's `workers` to the list it returns.

- [ ] **Step 1: Write the failing tests**

`test_parallel_and_serial_sync_write_identical_indexes` — real worker processes:
- `monkeypatch.setattr(parallel, "BATCH_ROWS", 7)`.
- Root with `.minirag/classify.toml` (a path facet `initiative`, `alpha` path rule, default `general`; a model facet `type` with `minutes`/`report`), `model_factory.default = {"report": 1.0}`; `alpha/shrink.md` of 6 parts, `keep.md`.
- Two pipelines over the same root, each with a `FakeEmbedder` and `LabelerSource(cfg.roots, tmp_path / "cache", model_factory)` set up as `tests/test_pipeline_labels.py: make_corpus` does: A with `DB_PATH=tmp_path/"a"`, `RAG_SYNC_WORKERS=1`; B with `DB_PATH=tmp_path/"b"`, `RAG_SYNC_WORKERS=2`.
- Round 1: `run_sync` A, then B (2 files: both in-process).
- Round 2: `shrink.md` rewritten with 2 parts; 10 new multi-part files (half under `alpha/`); `broken.docx` holding `b"not a zip"`. `run_sync` A, then B with a spy wrapping the real `parallel.prepare_files`.

```python
assert spy_workers == [2]
assert counts_a == counts_b and counts_b["failed"] == 1 and counts_b["signed"] == 0
assert sorted(errors_a, key=str) == sorted(errors_b, key=str)
rows_a, rows_b = all_rows(store_a), all_rows(store_b)            # to_arrow().to_pylist()
assert Counter(r["id"] for r in rows_b) == Counter(r["id"] for r in rows_a)   # nothing dropped or doubled
by_id = {r["id"]: r for r in rows_a}
for r in rows_b:
    assert {k: v for k, v in r.items() if k != "ingested_at"} == {k: v for k, v in by_id[r["id"]].items() if k != "ingested_at"}
assert not any(r["id"].startswith(str(root / "alpha" / "shrink.md") + "#") and r["chunk_index"] >= 2 for r in rows_b)
assert any(r["labels_json"] for r in rows_b)                    # labels really ran, so the compare means something
written = sum(1 for r in rows_b if r["source"] in round2_sources)
assert written > 14 and written % 7                              # two full batches and a partial last one
assert signatures(store_a) == signatures(store_b) and store_b.unsigned_sources() == []   # (source, signature, shingles)
assert documents(store_a) == documents(store_b)                  # plan 1's table, every column but ingested_at
for q in ("alpha body", "paragraph 3", "unrelated words"):
    assert run_query(store_a, emb, cfg_a, q, top_k=20)["sources"] == run_query(store_b, emb, cfg_b, q, top_k=20)["sources"]
again, _ = run_sync(pipeline_b, store_b, cfg_b.roots, cfg_b.max_file_size)
assert (again["ingested"], again["deleted"], again["moved"]) == (0, 0, 0)
```

(`run_query(store, embedder, config, query, *, top_k=..., reranker=None)` as plan 1 left it; no reranker here. Pick part counts so `written % 7` holds; the assertion keeps it honest.)

```python
def test_a_small_sync_stays_in_this_process(tmp_path, fake_embedder, monkeypatch):
    # RAG_SYNC_WORKERS=4, 3 files; parallel.prepare_files replaced by a function raising AssertionError
    assert counts["ingested"] == 3

def test_failures_in_a_parallel_sync_are_counted_like_serial_ones(inline_workers, ...):
    # 9 files incl. broken.docx; store.replace_sources raises; replace_source raises for c.md only
    # -> counts ingested 7, failed 2; errors name broken.docx and c.md;
    #    on_event saw f"failed {c}: ..." and f"ingested {a}"

def test_a_parallel_sync_holds_the_sync_lock_while_it_works(inline_workers, ...):
    # the stand-in's preparer tries SyncLock(store.db_path).acquire() on its first file and records SyncLockBusy;
    # after run_sync returns, `with sync_lock(store.db_path): pass` succeeds

def test_tessdata_is_fetched_once_before_the_workers_start(inline_workers, monkeypatch, ...):
    # ocr.available -> True; ocr.ensure_tessdata recorded; 8 .md files, RAG_SYNC_WORKERS=2
    # -> one call (cfg.cache_dir, cfg.ocr_lang), before the first prepare_file;
    # with ensure_tessdata raising OcrError the sync still ingests all 8

def test_the_store_hook_runs_once_before_the_workers_start(inline_workers, ...):
    # Pipeline(..., before_store=lambda: seen.append(list(inline_workers))); 8 .md files, RAG_SYNC_WORKERS=2
    assert seen == [[]] and inline_workers == [2]           # once, and before prepare_files was called

def test_an_unreachable_endpoint_stops_dispatch_and_says_so(inline_workers, monkeypatch, ...):
    # worker preparers raise EndpointError for every file; 8 .md files, RAG_SYNC_WORKERS=2
    # at most 3 files attempted; counts["failed"] == 8; exactly one error mentions "endpoint unreachable, 5 files not synced"
    # a second sync with a working endpoint ingests all 8
```

`tests/test_parity.py`:

```python
async def test_a_parallel_sync_counts_the_same_through_both_shells(tmp_path, capsys, monkeypatch, fake_embedder, inline_workers):
    # 8 files; RAG_SYNC_WORKERS=2 via monkeypatch.setenv (CLI) and in the MCP env dict; separate DB_PATHs;
    # cli._make_embedder -> fake_embedder, as the other parity tests do;
    # CLI `sync --json` vs MCP sync_start + polling sync_status (as tests/test_server.py does)
    assert cli_counts == mcp_counts and inline_workers == [2, 2]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_sync_parallel.py tests/test_parity.py -v`. Expected: the parallel tests FAIL (`spy_workers == []`, `inline_workers == []`: today's sync never calls `prepare_files`). Plan 1's `tests/test_rebuild.py` and `tests/test_handle.py` pass before and after.
- [ ] **Step 3: Implement** in `sync.py`.
- [ ] **Step 4: Run, then prove the equivalence test can fail.** It passes; with the final `writer.flush()` removed it must go red (rows missing), and with `BatchWriter.add` queueing a source twice it must go red (Counter mismatch). Restore; run the gate, including `uv sync --exact && uv run pytest -q`.
- [ ] **Step 5: Document.** README Configuration table, after `RAG_MAX_FILES`:

  `| RAG_SYNC_WORKERS | physical cores - 1, within memory | Worker processes a sync parses, chunks and embeds in; the sync process writes their rows in batches. Unset, the count is also held to half of the machine's memory, counting about 1.5 GB per worker plus the embedding model when it runs in-process (each worker loads its own copy: about 3.3 GB with multilingual-e5-large). A sync with fewer than 8 files to read stays in one process. |`

- [ ] **Step 6: Commit and close**

```bash
git add src/minirag_mcp/sync.py tests/conftest.py tests/test_sync_parallel.py tests/test_parity.py README.md
git commit -m "feat(sync): parse, chunk and embed in worker processes, write in batches"
bd close minirag-mcp-g7q.6
```

Then follow CLAUDE.md's session-completion protocol (quality gates, `git pull --rebase`, `git push`, `git status` up to date).


### Task 10: Measure the parser-noise items on the ruler and fix the ones that move retrieval

> Added 2026-11-19 (spec section 5: "fixed only if the ruler shows they move retrieval" had no
> task). Items (bd minirag-mcp-g7q.12): boilerplate and table-header titles; transcript block
> headers repeated per chunk; PDF table-of-contents leader dots spliced into words; spreadsheet
> `Unnamed: N` / `NaN` noise and the sheet name inside cells; mixed Latin/Cyrillic look-alikes
> after OCR (largely fixed by Tesseract, Task 3). For each: build the variant on a copy of the
> owner's index (private ruler, bd memory `solidcore-ruler`; scripts measure.py/score.py there),
> measure hit@10/MRR on both fields with the sign test, and implement only the items that gain
> significantly (TDD, one commit each). Record every number, kept or dropped, in the
> g7q.12 notes; nothing private goes into the repo.

### Task 11: Measure Apple Vision as a macOS second reader for low-confidence Tesseract pages

> Added 2026-11-19 (owner asked why it was deferred). The spike's evidence rests on one poor
> scan. Collect every OCR'd page of the owner's corpus with Tesseract mean word confidence
> below 0.85 (plus the fax-quality specification folder named in the private OCR report),
> transcribe 10-15 regions by hand, and compare Tesseract alone vs Tesseract + Vision (2
> overlapping ROI bands, accurate, ru-RU+en-US, language correction on) on CER, word recall
> and the private ruler's questions that hit those documents. Pick the confidence threshold
> from the data. Implement the second reader (macOS-only, `sys_platform == 'darwin'` marker,
> pyobjc in the `[ocr]` extra) only if it gains; it re-OCRs only the low-confidence pages.
> Numbers go to bd minirag-mcp-g7q.9; private material stays outside the repo.
