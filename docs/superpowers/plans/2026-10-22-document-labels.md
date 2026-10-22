# Document Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-document labels (any facets a corpus defines, plus a document date), computed at ingest by path rules, defaults and an in-process Plumb-4B, stored as additive index columns and usable as filters in `query_documents` / `list_files` and the CLI.

**Architecture:** A new `labels/` package holds everything label-specific: the taxonomy file, date extraction, facet resolution, the llama.cpp readout, the `Labeler` that owns the model per job, and filter construction. `Store` gains five string columns and an extra SQL filter argument; `Pipeline` writes labels on every chunk it stores and re-labels stale sources from stored text; `results.py` turns filters and labels into the wire shape both shells share.

**Tech Stack:** Python ≥3.11 (`tomllib`), LanceDB 0.36 (`contains()` and string comparison in `where`, prefiltered on both search sides), `llama-cpp-python` 0.3.36 + `numpy` + `huggingface-hub` in a new `[classify]` extra, pytest with a fake choice model.

**Spec:** `docs/superpowers/specs/2026-10-22-document-labels-design.md` — read it before Task 1; every task argues from it.

## Global Constraints

- `requires-python = ">=3.11"`; the taxonomy file is read with `tomllib`.
- The base install gains no dependency and no compile step. `[classify]` = `llama-cpp-python>=0.3.36`, `numpy>=2.1`, `huggingface-hub>=0.20`.
- The feature is on only when `<roots[0]>/.minirag/classify.toml` exists. Without it: no new key in any tool, CLI or status output, no model factory ever called, behaviour identical. (The five index columns are still added on open, like `ocr_engine`: additive, `''` for every row.)
- The model is never loaded or downloaded at server start, nor by `status`, `list_files` or a query. It is loaded by the first document in a job that needs it and released when the outermost job ends.
- A labelling failure never aborts a sync or an ingest and never counts as `failed`; it leaves the facet unset and adds a reason to the job's report.
- Re-labelling never re-embeds: it reads stored chunk text and updates label columns only.
- MCP and CLI stay thin: label logic lives in `labels/` and `results.py`; both shells call the same functions.
- Public repository: tests use synthetic English documents and paths only. Nothing from the owner's corpus enters the repo; the owner-corpus check (Task 11) runs from `~/projects/test/minirag-labels-eval`.
- Fixed values: `N_CTX = 8192`, `MAX_EVIDENCE_CHARS = 6000`, `HEADER_CHARS = 1500`, `uncertain_below` default `0.7`, at most 16 values in a facet the model answers, default question `"What kind of document is this?"`, default file-name label `"File name"`.
- Nothing is printed to stdout outside the CLI's own output: stdout is the MCP stdio channel. llama.cpp logs go to stderr only (`verbose=False`).
- Gates for every task: `uv run pytest -q`, `uv run ruff check src tests`, `uv run ruff format --check src tests`.
- Comments state constraints the code cannot show; English only (CLAUDE.md).

## Review Focus

1. **`classify.toml` edited while the MCP server runs.** A long-lived server must pick up the edit (filters accept the new values, sync uses the new hash) without a restart. Test: `test_source_reloads_an_edited_file` (Task 5).
2. **Decomposed Cyrillic file names (NFD, as macOS stores them) against rules typed in an editor (NFC).** A path rule must match. Test: `test_nfd_path_matches_an_nfc_rule` (Task 3).
3. **An older index opened read-only, so the label columns could not be added.** A label or date filter must be refused with a message naming write access, not crash in SQL. Test: `test_a_filter_on_an_index_without_label_columns_is_refused` (Task 7).
4. **An `ingest_file` arriving while a sync is labelling.** Both threads share one model; llama.cpp contexts are not thread-safe. Test: `test_concurrent_jobs_never_enter_the_model_together` (Task 5).
5. **A filter on a facet whose `within` excludes most of the scope.** Documents outside `within` are excluded legitimately and must not inflate `filterCounts.unlabelled`. Test: `test_documents_outside_within_are_not_counted_as_unlabelled` (Task 7).

---

## File Structure

Create:
- `src/minirag_mcp/labels/__init__.py` — re-exports `Labeler`, `LabelerSource`, `JobReport`, `taxonomy_path`.
- `src/minirag_mcp/labels/taxonomy.py` — `classify.toml` → validated, hashable `Taxonomy`.
- `src/minirag_mcp/labels/dates.py` — document date from header, name, file time.
- `src/minirag_mcp/labels/resolve.py` — root-relative paths, per-facet resolution, `DocLabels` encoding and wire fields.
- `src/minirag_mcp/labels/model.py` — the only module importing `llama_cpp`, `numpy`, `huggingface_hub` (all lazily): prompt, calibration, model file resolution, `LlamaChoiceModel`.
- `src/minirag_mcp/labels/labeler.py` — `Labeler` (model per job, lock, report) and `LabelerSource` (reload on file change).
- `src/minirag_mcp/labels/filters.py` — filter parsing, SQL, Python predicate, loud counts.
- Tests: `tests/test_labels_taxonomy.py`, `tests/test_labels_dates.py`, `tests/test_labels_resolve.py`, `tests/test_labels_model.py`, `tests/test_labeler.py`, `tests/test_labels_filters.py`, `tests/test_store_labels.py`, `tests/test_pipeline_labels.py`, `tests/integration/test_classify_model.py`.

Modify: `src/minirag_mcp/store.py`, `src/minirag_mcp/results.py`, `src/minirag_mcp/ingest/pipeline.py`, `src/minirag_mcp/sync.py`, `src/minirag_mcp/server.py`, `src/minirag_mcp/cli/__init__.py`, `src/minirag_mcp/instructions.py`, `tests/conftest.py`, `tests/test_store.py` (`_row`), `tests/test_parity.py`, `tests/test_sync.py`, `pyproject.toml`, `README.md`, `CLAUDE.md`.

---

### Task 1: The taxonomy file

**Files:**
- Create: `src/minirag_mcp/labels/__init__.py`, `src/minirag_mcp/labels/taxonomy.py`
- Test: `tests/test_labels_taxonomy.py`

**Interfaces:**
- Produces:
  - `TAXONOMY_FILE = Path(".minirag") / "classify.toml"`; `taxonomy_path(roots: Sequence[Path]) -> Path` (= `roots[0] / TAXONOMY_FILE`)
  - `class TaxonomyError(Exception)`
  - `@dataclass(frozen=True) class Value: id: str; description: str; paths: tuple[str, ...] = ()`
  - `@dataclass(frozen=True) class Facet: name: str; values: tuple[Value, ...]; question: str = DEFAULT_QUESTION; within: tuple[str, ...] = (); default: str | None = None; override_at: float | None = None` with property `needs_model -> bool` (`default is None or override_at is not None`) and `value_ids -> tuple[str, ...]`
  - `@dataclass(frozen=True) class ModelSpec: temperature: float; path: Path | None = None; repo: str | None = None; file: str | None = None; revision: str | None = None; sha256: str | None = None; uncertain_below: float = 0.7; file_name_label: str = "File name"`
  - `@dataclass(frozen=True) class Taxonomy: facets: tuple[Facet, ...]; model: ModelSpec | None; source: Path` with property `hash -> str` and method `facet(name: str) -> Facet` (KeyError when unknown)
  - `load_taxonomy(path: Path) -> Taxonomy` (raises `TaxonomyError`)

The file format is the spec's "The configuration file" section. Validation rules, each raising `TaxonomyError` whose message names the TOML key:
- ids (facet names, value ids) match `^[a-z0-9][a-z0-9_-]*$` — so `|`, `=`, `~` and `'` can never appear in a stored key;
- unknown keys anywhere are errors (a typo like `overide_at` must not be ignored);
- each facet has ≥1 value; a facet with `needs_model` has ≤16 values; descriptions are non-empty strings;
- `paths` and `within` entries are non-empty, relative (no leading `/`), stored NFC-normalised with trailing `/` stripped;
- `default` names one of the facet's values; `override_at` in `(0, 1]`;
- `[model]` is required iff some facet `needs_model`; it has exactly one of `path` or all four of `repo`, `file`, `revision`, `sha256`; `temperature > 0`; `uncertain_below` in `(0, 1)`; `path` is `expanduser()`ed.

`hash` is `sha256` over `json.dumps(canonical, sort_keys=True, ensure_ascii=False)`, where `canonical` holds every facet field and every model field (path as a string). It must not depend on TOML formatting or key order.

- [ ] **Step 1: Write the failing tests**

```python
MINIMAL = '''
[model]
path = "/models/plumb.gguf"
temperature = 2.07

[facets.type]
values.minutes = { description = "Minutes. Date, participants, decisions." }
values.mail = { description = "Mail. Project correspondence." }
'''

def test_minimal_file_loads(tmp_path):
    t = load_taxonomy(write(tmp_path, MINIMAL))
    assert [f.name for f in t.facets] == ["type"]
    assert t.facet("type").value_ids == ("minutes", "mail")
    assert t.facet("type").question == "What kind of document is this?"
    assert t.model.temperature == 2.07 and t.model.uncertain_below == 0.7

def test_an_unknown_key_is_named(tmp_path):        # "overide_at = 0.9" under [facets.type]
    with pytest.raises(TaxonomyError, match=r"facets\.type\.overide_at"): ...

def test_default_must_be_one_of_the_values(tmp_path):   # match r"facets\.project\.default"
def test_model_section_is_required_only_when_a_facet_needs_the_model(tmp_path):
    # path+default facets alone load with no [model]; a facet with no default and no [model] -> match r"\[model\]"
def test_model_is_a_path_or_a_fully_pinned_repo(tmp_path):
    # both path and repo -> error; repo without sha256 -> match "sha256"
def test_a_model_facet_takes_at_most_16_values(tmp_path):   # 17 values -> match "16"
def test_ids_are_lowercase_slugs(tmp_path):        # "Mail" -> error; "client-data" loads
def test_paths_are_relative_and_normalised(tmp_path):
    # "/abs" -> error; "a/b/" -> "a/b"; an NFD "й" in a path is stored NFC
def test_hash_ignores_formatting_but_not_content(tmp_path):
    # MINIMAL with keys reordered and extra blank lines -> same hash;
    # one description changed -> different; temperature 2.0 -> different
def test_taxonomy_path_is_under_the_first_root(tmp_path):
    assert taxonomy_path((tmp_path / "a", tmp_path / "b")) == tmp_path / "a" / ".minirag" / "classify.toml"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_labels_taxonomy.py -q` — expect FAIL (module missing).
- [ ] **Step 3: Implement `taxonomy.py`** to the interfaces above; `__init__.py` re-exports `taxonomy_path` (more exports join in Task 5).
- [ ] **Step 4: Run** the tests — expect PASS; run the gates.
- [ ] **Step 5: Commit** `feat(labels): parse and validate the per-corpus taxonomy file`

---

### Task 2: Document dates

**Files:**
- Create: `src/minirag_mcp/labels/dates.py`
- Test: `tests/test_labels_dates.py`

**Interfaces:**
- Produces:
  - `HEADER_CHARS = 1500`
  - `@dataclass(frozen=True) class DocDate: value: str; source: str` — `value` is `YYYY-MM-DD` or `""`; `source` is `"document" | "name" | "file" | "none"`
  - `date_from_header(text: str) -> str | None`
  - `date_from_name(stem: str) -> str | None`
  - `document_date(stem: str | None, text: str, file_time: float | None) -> DocDate` — header, then name, then `file_time` (local date of the timestamp), else `DocDate("", "none")`

Date forms a rule may parse: `DD.MM.YYYY`, `DD.MM.YY`, `DD/MM/YYYY`, `YYYY-MM-DD`, `D <Russian month> YYYY` (stems `январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр`), `<English Month> D, YYYY`, `D <English Month> YYYY`. Two-digit years are 20YY; four-digit years must be 2000–2099; an impossible calendar date makes that rule miss and the next one run.

Header rules, in order, over `text[:HEADER_CHARS]`; the first that yields a valid date wins:
1. a labelled field: `(?i)(?:\*\*)?\b(?:date|sent|дата|отправлено)\b(?:\*\*)?\s*:\s*(?:\*\*)?` followed by a date form within the next 60 characters (each occurrence is tried; the printed-email case has several empty `Date:` labels);
2. a Teams recording header: English month, day, year, then a time: `(?i)\b(January|…|December) (\d{1,2}), (\d{4}),? \d{1,2}:\d{2}`;
3. a weekday-prefixed date (English or Russian weekday, optional comma) followed by `D Month YYYY` or `Month D, YYYY`;
4. a Russian outgoing-number line: `(?i)исх\.?\s*№?[^\n]{0,40}?\bот\s+(\d{1,2}[./]\d{1,2}[./]\d{2,4})`;
5. any date form on the first non-empty line only.

Never any other date from the body: that is the cover-letter trap (the first date in a cover letter belongs to the contract it cites).

Name rules, in order, over the file name without extension:
1. `YYYY-MM-DD`; 2. `DD-MM-YYYY`; 3. `DD.MM.YY(YY)`; 4. eight digits: `YYYYMMDD` if it starts with `20` and is a valid date, else `DDMMYYYY`; 5. `YY-MM-DD`; 6. a leading `DDMMYY` (with or without a following space). Every numeric group is bounded by `(?<!\d)…(?!\d)`.

- [ ] **Step 1: Write the failing tests** — two parametrised tables plus the precedence tests:

```python
@pytest.mark.parametrize("stem, expected", [
    ("030325 Weekly sync-ru-RU", "2025-03-03"),
    ("180225Requirements Sales", "2025-02-18"),
    ("Protocol_2026-09-30_client", "2026-09-30"),
    ("20260814 status W1", "2026-08-14"),
    ("19022025 Sales requirements", "2025-02-19"),
    ("27-03-2025 Rates", "2025-03-27"),
    ("25-02-12 Purchasing part 2", "2025-02-12"),
    ("(4821) X_60 Summary of 28.08.25", "2025-08-28"),
    ("QA X_517 review_26.08_2", None),
    ("31022026 impossible", None),
])
def test_date_from_name(stem, expected): ...

@pytest.mark.parametrize("text, expected", [
    ("General information\n**Date:** 26 августа 2026\nParticipants", "2026-08-26"),
    ("Дата: 3 сентября 2026\n", "2026-09-03"),
    ("**Workshop-Meeting Recording** August 5, 2026, 10:32AM 50m 59s", "2026-08-05"),
    ("Subject: Date: Date: From: To: RE: Workshop Friday, 21 August 2026 at 6:24", "2026-08-21"),
    ("Sent: Friday, August 21, 2026 at 6:57 PM", "2026-08-21"),
    ("# Call: demo. Transcript of the call 14.10.2026\n[00:00–02:00] hello", "2026-10-14"),
    ("Исх. № 12/3 от 12.08.2026\nTo the bank", "2026-08-12"),
    # the trap: a cover letter whose only dates belong to the contract it cites
    ("To the currency control department\nPlease amend contract No 1 dated 23.07.2025, ref. 4/118 of 28/01/2025.", None),
])
def test_date_from_header(text, expected): ...

def test_header_beats_name_and_name_beats_file_time(): ...
def test_file_time_is_the_last_resort_and_says_so():
    d = document_date("no date here", "nothing either", datetime(2026, 7, 23, 12).timestamp())
    assert d == DocDate("2026-07-23", "file")
def test_no_date_anywhere_is_none(): assert document_date(None, "text", None) == DocDate("", "none")
```

- [ ] **Step 2: Run** `uv run pytest tests/test_labels_dates.py -q` — expect FAIL.
- [ ] **Step 3: Implement `dates.py`** to the rules above.
- [ ] **Step 4: Run** — expect PASS; gates.
- [ ] **Step 5: Commit** `feat(labels): document date from header, file name or file time`

---

### Task 3: Facet resolution and label encoding

**Files:**
- Create: `src/minirag_mcp/labels/resolve.py`
- Test: `tests/test_labels_resolve.py`

**Interfaces:**
- Consumes: `Facet`, `Taxonomy` (Task 1); `DocDate` (Task 2); `minirag_mcp.scope.is_under`.
- Produces:
  - `@dataclass(frozen=True) class Label: value: str; confidence: float; source: str; second: str | None = None; uncertain: bool = False` — `source` is `"path" | "default" | "model"`
  - `@dataclass(frozen=True) class DocLabels: labels: Mapping[str, Label]; date: DocDate; taxonomy_hash: str` with methods `labels_json() -> str` and `label_keys() -> str`
  - `root_relative(source: str, roots: Sequence[Path]) -> str | None` — NFC-normalised path relative to the longest root holding `source`; `None` when no root holds it
  - `path_value(facet: Facet, rel: str) -> str | None` — the value whose `paths` entry is the longest `is_under` prefix of `rel` (both NFC)
  - `resolve_facet(facet: Facet, rel: str | None, ask: Callable[[], Mapping[str, float]] | None, uncertain_below: float) -> Label | None`
  - `decode_labels(labels_json: str) -> dict[str, Label]`
  - `document_fields(labels_json: str, doc_date: str, doc_date_source: str) -> dict` — `{"labels": {facet: {"value", "confidence", "source", "uncertain"}}, "date": {"value", "source"}}`, each key only when non-empty; `{}` when both are empty

`resolve_facet`, in order:
1. `facet.within` set and (`rel is None` or `rel` under none of them) → `None`, `ask` not called.
2. `rel` not None and `path_value` matches → `Label(v, 1.0, "path")`; if `facet.override_at` and `ask`: take `ask()`; when its top value differs from `v` and has probability ≥ `override_at`, return that model label instead.
3. `facet.default` set → `Label(default, 1.0, "default")` when `rel` is not None, else `None` (a text or URL document gets no default).
4. `ask` available → `Label(top, p_top, "model", second=second_best, uncertain=p_top < uncertain_below)`; else `None`.
Exceptions from `ask` propagate (the labeler handles them).

`labels_json()`: `json.dumps({facet: {"value", "confidence": round(p, 4), "source", "second", "uncertain"}}, sort_keys=True, ensure_ascii=False)`; `""` when no labels. `label_keys()`: `"|" + "|".join(sorted(entries)) + "|"`, entries `f"{facet}={value}"` for every label plus `f"{facet}~{second}"` for each uncertain label with a second; `""` when no labels.

- [ ] **Step 1: Write the failing tests**

```python
def test_root_relative_picks_the_longest_root(): ...   # roots /d and /d/inner; "/d/inner/x.pdf" -> "x.pdf"
def test_root_relative_outside_every_root_is_none(): ...
def test_longest_path_rule_wins_at_segment_boundaries():
    # values initiatives(paths ["ai"]) and fx(paths ["ai/fx"]): "ai/fx/a.pdf" -> fx; "ai-old/a.pdf" -> None
def test_nfd_path_matches_an_nfc_rule():
    source = unicodedata.normalize("NFD", "/d/ai/Новый проект/a.pdf")  # "й" decomposed, as APFS stores it
    # rule paths = ["ai/Новый проект"] typed NFC; root_relative(source, [Path("/d")]) + path_value must match it
def test_within_excludes_without_asking(): ...          # ask is a Mock; assert not called
def test_a_path_rule_is_not_overridden_below_override_at():
    # override_at 0.9; ask -> {"spec": 0.85, "instruction": 0.15}, path says instruction -> path label
def test_a_confident_model_overrides_a_path_rule():     # ask -> {"spec": 0.95, ...} -> Label("spec", 0.95, "model")
def test_default_applies_to_files_but_not_to_text_documents():
    # rel "x.pdf" -> default; rel None -> None, ask not called
def test_uncertain_answer_keeps_its_second_choice():
    lab = resolve_facet(f, "x", lambda: {"a": 0.5, "b": 0.3, "c": 0.2}, 0.7)
    assert lab == Label("a", 0.5, "model", second="b", uncertain=True)
    assert DocLabels({"t": lab}, DocDate("", "none"), "h").label_keys() == "|t=a|t~b|"
def test_certain_answer_has_no_soft_key(): ...          # {"a": 0.9, "b": 0.1} -> "|t=a|"
def test_labels_json_round_trips(): ...
def test_document_fields_omit_what_is_empty():
    assert document_fields("", "", "") == {}
```

- [ ] **Step 2: Run** — expect FAIL. **Step 3: Implement `resolve.py`.** **Step 4: Run** — PASS; gates.
- [ ] **Step 5: Commit** `feat(labels): resolve facets by path, default or model, and encode them`

---

### Task 4: The choice model

**Files:**
- Create: `src/minirag_mcp/labels/model.py`
- Test: `tests/test_labels_model.py`, `tests/integration/test_classify_model.py`
- Modify: `pyproject.toml` (extra and marker, so the integration test can be collected)

**Interfaces:**
- Consumes: `ModelSpec` (Task 1).
- Produces:
  - `INSTALL_HINT = "install the classify extra: uv tool install 'minirag-mcp[classify]' (or pip install 'minirag-mcp[classify]')"`
  - `LETTERS = "ABCDEFGHIJKLMNOP"`, `N_CTX = 8192`, `MAX_EVIDENCE_CHARS = 6000`
  - `class ModelError(Exception)`
  - `class ChoiceModel(Protocol): def probabilities(self, evidence: str, question: str, options: Sequence[tuple[str, str]]) -> dict[str, float]; def close(self) -> None` — `options` are `(value_id, description)`
  - `available() -> bool` — `importlib.util.find_spec` finds `llama_cpp` and `numpy`
  - `build_prompt(evidence: str, question: str, options: Sequence[tuple[str, str]]) -> str`
  - `calibrate(letter_logprobs: Sequence[float], temperature: float) -> list[float]` — `exp((z - max) / T)`, normalised
  - `resolve_model_file(spec: ModelSpec, cache_dir: Path) -> Path` — `spec.path` if it exists (else `ModelError` naming it); otherwise `huggingface_hub.hf_hub_download(spec.repo, spec.file, revision=spec.revision, cache_dir=cache_dir / "classify")`, then its sha256 must equal `spec.sha256` (else `ModelError` with expected and actual)
  - `class LlamaChoiceModel(ChoiceModel)`: `__init__(self, path: Path, temperature: float)`
  - `default_model_factory(spec: ModelSpec, cache_dir: Path) -> ChoiceModel` — `ModelError(INSTALL_HINT)` when not `available()`, else `LlamaChoiceModel(resolve_model_file(spec, cache_dir), spec.temperature)`

The prompt is the model card's reference format (the `jevk5` client; Qwen3.5 template with thinking off). Pin these exactly:

```python
SYSTEM_PROMPT = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
# user = json.dumps({"evidence": evidence, "criterion": question,
#                    "options": [{"letter": LETTERS[i], "description": f"{vid}: {desc}"}, ...]},
#                   ensure_ascii=False)
```

`LlamaChoiceModel`: `Llama(model_path=str(path), n_ctx=N_CTX, n_gpu_layers=-1, n_batch=2048, verbose=False)`; each letter must tokenise (`add_bos=False, special=False`) to exactly one token, else `ModelError`. Per call: tokenise the prompt with `add_bos=False, special=True`; ≥ `N_CTX` tokens → `ModelError` naming the count; `reset()`, `eval(tokens)`; logits from `llama_cpp.llama_get_logits_ith(self._llm.ctx, -1)` via `numpy.ctypeslib.as_array(..., shape=(n_vocab,))` as float64 (`Llama.scores` stays empty without `logits_all`); log-softmax; the option letters' log-probabilities through `calibrate`. `close()` calls `self._llm.close()`.

- [ ] **Step 1: Write the failing unit tests** (no extra needed):

```python
def test_prompt_is_the_reference_format():
    p = build_prompt("File name: a.docx\nhello", "What kind?", [("mail", "Mail."), ("minutes", "Minutes.")])
    assert p.startswith("<|im_start|>system\nApply the supplied criterion")
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    user = p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    assert json.loads(user) == {"evidence": "File name: a.docx\nhello", "criterion": "What kind?",
        "options": [{"letter": "A", "description": "mail: Mail."}, {"letter": "B", "description": "minutes: Minutes."}]}
def test_calibration_matches_the_reference_readout():
    z, t = [-0.1, -2.3, -5.0], 2.07
    w = [math.exp((v + 0.1) / t) for v in z]
    assert calibrate(z, t) == pytest.approx([x / sum(w) for x in w])
def test_available_is_false_without_the_extra(monkeypatch): ...   # find_spec -> None
def test_a_missing_local_model_file_is_named(tmp_path): ...
def test_a_downloaded_file_with_the_wrong_sha256_is_refused(tmp_path, monkeypatch):
    # monkeypatch huggingface_hub.hf_hub_download to write a file and return its path
def test_a_pinned_download_lands_in_the_cache_dir(tmp_path, monkeypatch): ...   # asserts cache_dir/"classify" and revision passed
def test_the_default_factory_without_the_extra_says_what_to_install(monkeypatch): ...  # ModelError match "minirag-mcp\[classify\]"
```

`huggingface_hub` is importable in the base install (fastembed brings it), so these tests run without the extra.

- [ ] **Step 2: Run** `uv run pytest tests/test_labels_model.py -q` — expect FAIL.
- [ ] **Step 3: Implement `model.py`**; all three heavy imports inside functions.
- [ ] **Step 4: Add the extra and the marker to `pyproject.toml`:**

```toml
classify = [
    "llama-cpp-python>=0.3.36",
    # llama-cpp-python already pulls this in, but we import it directly: the readout
    # views llama.cpp's logits buffer as an array.
    "numpy>=2.1",
    # fastembed already pulls this in, but we import it directly: a pinned model file
    # is fetched with hf_hub_download(repo, file, revision=, cache_dir=).
    "huggingface-hub>=0.20",
]
```

and `addopts = "-m 'not slow and not classify'"`, `markers` += `"classify: needs the [classify] extra and a local GGUF named by MINIRAG_TEST_GGUF"`.

- [ ] **Step 5: Write `tests/integration/test_classify_model.py`** (`pytestmark = pytest.mark.classify`; skip unless `available()` and `MINIRAG_TEST_GGUF` names a file): probabilities over three options sum to 1; a synthetic English transcript (`"[00:00–02:00] Anna: good morning\n[00:02] Ben: let us start"`) puts `transcript` first among transcript / minutes / mail; `close()` is idempotent.
- [ ] **Step 6: Run** the unit tests — PASS. Run the integration test once by hand: `uv run --extra classify env MINIRAG_TEST_GGUF=<path> pytest -m classify -q` — PASS; then `uv sync --exact` and the gates.
- [ ] **Step 7: Commit** `feat(labels): in-process choice model over a GGUF, behind the [classify] extra`

---

### Task 5: The labeler

**Files:**
- Create: `src/minirag_mcp/labels/labeler.py`; extend `src/minirag_mcp/labels/__init__.py`
- Modify: `tests/conftest.py` (fake model)
- Test: `tests/test_labeler.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `@dataclass class JobReport: problems: list[str]` with `add(problem: str) -> None` (keeps first occurrence only)
  - `class Labeler`:
    - `__init__(self, taxonomy: Taxonomy, roots: Sequence[Path], cache_dir: Path, model_factory: Callable[[ModelSpec, Path], ChoiceModel] = default_model_factory)`
    - `taxonomy: Taxonomy` (attribute)
    - `job(self) -> ContextManager[JobReport]` — re-entrant and shared across threads: the first entry creates the report, nested and concurrent entries get the same one, the last exit closes the model
    - `label(self, *, source: str, source_type: str, title: str, chunk_texts: Sequence[str], file_time: float | None, trace: dict[str, dict[str, float]] | None = None) -> DocLabels` — runs inside its own `job()` when called outside one; fills `trace[facet]` with the full distribution whenever the model was asked
    - `model_state(self) -> str` — `"loaded"`, `"unavailable"` (some facet needs the model and `model.available()` is false), or `"on demand"`
  - `class LabelerSource`: `__init__(self, roots: Sequence[Path], cache_dir: Path, model_factory=default_model_factory)`; `path: Path`; `get(self) -> tuple[Labeler | None, str | None]` — `(None, None)` when the file is absent, `(None, message)` when invalid, otherwise a `Labeler` that is reused while `(st_mtime_ns, st_size)` is unchanged and rebuilt when it changes

Behaviour to pin:
- Evidence: `f"{spec.file_name_label}: {name}\n" + " ".join(" ".join(chunk_texts).split())[:MAX_EVIDENCE_CHARS]`; `name` is `Path(source).name` for `source_type == "file"`, otherwise `title`. Built once per document, only if some facet asks.
- Date input: `document_date(Path(source).stem if file else None, "\n\n".join(chunk_texts), file_time)` — not whitespace-collapsed (line structure matters to the header rules).
- `rel = root_relative(source, roots)` for files, `None` otherwise.
- The model is created by the first `ask` in a job. One `threading.RLock` guards creating, calling and closing it. If creating fails, the reason goes to the report and nothing retries creating it in that job. If a single call raises `ModelError`, the report gets `f"{source}: {reason}"` and only that document's model facets stay unset.

Fake for tests, in `tests/conftest.py`:

```python
class FakeChoiceModel:
    """Answers by substring of the evidence. Uncertain and wrong answers are configured on
    purpose: a fake that is always confident hides the soft-filter and count paths."""
    def __init__(self, answers: Mapping[str, Mapping[str, float]], default: Mapping[str, float] | None = None, fail_on: str | None = None): ...
    calls: list[tuple[str, str]]   # (question, evidence)
    closed: bool
    def probabilities(self, evidence, question, options) -> dict[str, float]: ...  # restricted to the option ids, renormalised
    def close(self) -> None: ...

@pytest.fixture
def model_factory():  # returns a factory with .created (list of FakeChoiceModel) and .answers to configure
```

- [ ] **Step 1: Write the failing tests**

```python
def test_the_model_loads_once_per_job_and_closes_after(...): # 3 docs in one job -> created == 1, closed
def test_label_outside_a_job_is_its_own_job(...):            # 2 calls -> created == 2, both closed
def test_path_and_default_facets_never_load_the_model(...):  # created == []
def test_a_model_that_cannot_load_is_reported_once_and_not_retried(...):
    # factory raises ModelError("needs the extra") -> 3 docs: path facets set, model facets absent,
    # report.problems == ["needs the extra"], factory called once
def test_one_failing_document_does_not_poison_the_job(...):  # fail_on="HUGE": that doc unset, the next labelled
def test_evidence_is_the_file_name_label_and_6000_collapsed_chars(...):
    # file_name_label "Имя файла": evidence == "Имя файла: a.pdf\n" + collapsed[:6000]
def test_text_documents_use_their_title_and_no_path_rules(...):
def test_dates_do_not_need_the_model(...):                   # model unavailable, date from the stem
def test_trace_records_every_distribution_asked(...):
def test_concurrent_jobs_never_enter_the_model_together(...):
    # FakeChoiceModel subclass that fails if probabilities() is re-entered; 2 threads x 20 docs
def test_source_reloads_an_edited_file(tmp_path, ...):
    # get() twice -> same object; rewrite with one more value, os.utime bump -> new object, new hash;
    # invalid edit -> (None, message naming the key); file removed -> (None, None)
```

- [ ] **Step 2: Run** — expect FAIL. **Step 3: Implement `labeler.py`**, export `Labeler`, `LabelerSource`, `JobReport` from `labels/__init__.py`. **Step 4: Run** — PASS; gates.
- [ ] **Step 5: Commit** `feat(labels): labeler that owns the model per job and reloads an edited taxonomy`

---

### Task 6: Store columns and filtered reads

**Files:**
- Modify: `src/minirag_mcp/store.py`, `tests/test_store.py` (`_row` writes the new columns)
- Test: `tests/test_store_labels.py`

**Interfaces:**
- Produces:
  - `LABEL_COLUMNS = ("labels_json", "label_keys", "label_hash", "doc_date", "doc_date_source")`, all `pa.string()`, added to the create-table schema and to `_ADDITIVE_COLUMNS` with `"''"`
  - `ChunkRecord`, `SourceInfo`, `SearchResult` gain `str` fields defaulting to `""`: ChunkRecord all five; SourceInfo `labels_json`, `label_hash`, `doc_date`, `doc_date_source`; SearchResult `labels_json`, `doc_date`, `doc_date_source`
  - `_META_COLS` gains `labels_json`, `label_hash`, `doc_date`, `doc_date_source`
  - `search(..., filter_sql: str | None = None)`, `_iter_meta(scopes=(), filter_sql=None)`, `list_sources(scopes=(), filter_sql=None)` — the scope clause and `filter_sql` are ANDed (each parenthesised); the vector side keeps `prefilter=True`
  - `set_document_labels(self, source: str, *, labels_json: str, label_keys: str, label_hash: str, doc_date: str, doc_date_source: str) -> int` — `self._table.update(where=<source clause>, values={...})`, returns rows updated
  - `move_source` writes `label_hash = ""` on the moved rows: path-derived labels may differ at the new path, and an empty hash is what the re-label pass (Task 8) picks up

Rows from an index whose label columns could not be added are read with `row.get(col, "")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_new_index_has_the_label_columns(tmp_path): ...
def test_an_older_index_gains_the_label_columns_on_open(tmp_path):
    # tests/test_store.py's _older_table(db_path, set(LABEL_COLUMNS), [_row(...)]) -> Store opens,
    # missing_columns == (), list_sources()[0].labels_json == ""
def test_set_document_labels_touches_every_chunk_of_one_source_only(tmp_path): ...
def test_a_label_filter_reaches_the_keyword_side_before_its_window(tmp_path):
    # 120 one-chunk sources containing "alpha"; the 3 labelled ones repeat it least, so BM25 ranks
    # them outside the 50-row FTS window; search("alpha", far_vector, hybrid_weight=1.0, top_k=3,
    # filter_sql="contains(label_keys, '|type=minutes|')") returns exactly those 3
def test_scope_and_filter_are_both_applied(tmp_path): ...       # search and list_sources
def test_results_and_sources_carry_labels_and_date(tmp_path): ...
def test_a_moved_source_keeps_its_vectors_and_loses_its_label_hash(tmp_path): ...
```

- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** the store changes; update `_row` in `tests/test_store.py` to write the five new keys (its own assertion demands exactly the created columns).
- [ ] **Step 4: Run** — PASS. Then prove the window test can fail: temporarily drop `filter_sql` from the FTS query's `where` and filter `fts_rows` in Python after `.to_list()` instead; the test must FAIL; revert. Gates.
- [ ] **Step 5: Commit** `feat(store): additive label and date columns, filtered search and listing`

---

### Task 7: Filters and the wire shape

**Files:**
- Create: `src/minirag_mcp/labels/filters.py`
- Modify: `src/minirag_mcp/results.py`
- Test: `tests/test_labels_filters.py`, `tests/test_results.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 5, 6.
- Produces (`filters.py`):
  - `class LabelFilterError(ValueError)`
  - `@dataclass(frozen=True) class LabelFilter: labels: tuple[tuple[str, tuple[str, ...]], ...]; date_from: str | None = None; date_to: str | None = None`
  - `parse_filter(labels: Mapping[str, str | Sequence[str]] | None, date_from: str | None, date_to: str | None, *, labeler: Labeler | None, error: str | None, taxonomy_file: Path | None) -> LabelFilter | None` — `None` when nothing is asked
  - `to_sql(f: LabelFilter) -> str`
  - `source_matches(f: LabelFilter, info: SourceInfo) -> bool` — the same predicate in Python
  - `filter_counts(f: LabelFilter, infos: Iterable[SourceInfo], taxonomy: Taxonomy, roots: Sequence[Path]) -> dict[str, int]` — `{"unlabelled": n, "undated": m}`
  - `filter_notice(f: LabelFilter, labeler: Labeler) -> str | None` — when a filtered facet `needs_model` and `labeler.model_state() == "unavailable"`: `f"labels for {facet} come from the model, which this installation cannot run; {INSTALL_HINT}"`
- Produces (`results.py`):
  - `run_query(..., labels=None, date_from=None, date_to=None, label_source: LabelerSource | None = None)` — gets `(labeler, error)` from `label_source`; adds `filterCounts` when a filter is set and `filterNotice` when `filter_notice` returns one
  - `list_payload(..., labels=None, date_from=None, date_to=None, sort: str | None = None, label_source: LabelerSource | None = None)` — same two keys under the same conditions
  - `labels_status(source: LabelerSource | None, store: Store | None) -> dict` — `{}` when the file is absent; `{"labels": {"file": str, "error": str}}` when invalid; otherwise `{"labels": {"file", "model", ("modelHint" when unavailable), "facets": {name: {"question", "values": {id: {"description", "count"}}, "unlabelled", "uncertain"}}, "undated"}}`

SQL, per facet (values ORed, facets ANDed):
`(contains(label_keys, '|type=minutes|') OR contains(label_keys, '|type~minutes|') OR …)`; dates: `doc_date != '' AND doc_date >= '2026-10-12' AND doc_date <= '2026-10-18'`.

`parse_filter` errors (raise `LabelFilterError`):
- any filter with the file absent → `f"document labels are off: there is no {taxonomy_file}"`;
- an invalid file → `f"{taxonomy_file} is invalid: {error}"`;
- an unknown facet or value → the known facets or values listed;
- a date not `YYYY-MM-DD`, or `date_from > date_to`.

`filter_counts` counts each in-scope document once:
- **unlabelled:** a filtered facet applies to it (no `within`, or `rel` inside `within`) and it has no label for that facet;
- **undated:** a date filter is set and its `doc_date` is `""`.

A document outside a facet's `within` counts in neither. `list_payload` also counts in-scope `not_ingested` files as unlabelled when a label filter is set.

Wire rules:
- `run_query` converts `LabelFilterError` into `QueryError`. It refuses any filter, with `QueryError` saying the index needs reopening with write access, when `store.missing_columns` contains a label column.
- `aggregate_sources` rows and `list_payload` file rows merge in `document_fields(...)` (the keys appear only when non-empty).
- `list_payload` with a filter keeps only rows whose source matches. `sort="date"` orders by `doc_date` descending, undated last, ties in the existing order. Any other `sort` raises `QueryError`.
- With no filter and the file absent, payloads have exactly today's keys.

- [ ] **Step 1: Write the failing tests**

```python
def test_sql_and_python_predicates_agree(): ...            # same 6 SourceInfos through both
def test_an_uncertain_document_matches_its_second_choice(): ...
def test_filters_without_a_taxonomy_file_are_refused(): ... # match "document labels are off"
def test_unknown_facet_and_value_list_the_known_ones(): ...
def test_bad_and_reversed_dates_are_refused(): ...
def test_documents_outside_within_are_not_counted_as_unlabelled(): ...
def test_undated_documents_are_counted_when_dates_are_filtered(): ...
# tests/test_results.py
def test_a_filtered_query_returns_matching_sources_with_labels_and_counts(tmp_path, fake_embedder): ...
def test_an_unfiltered_query_has_exactly_the_old_keys(tmp_path, fake_embedder):
    assert set(payload) == {"results", "sources", "parents"} and "labels" not in payload["sources"][0]
def test_list_sorted_by_date_puts_undated_last(tmp_path, fake_embedder): ...
def test_a_filter_on_an_index_without_label_columns_is_refused(tmp_path, monkeypatch):
    # _older_table + add_columns raising (as test_store.py's _raise_commit_conflict) -> QueryError match "write access"
def test_labels_status_is_empty_without_the_file_and_names_an_invalid_one(tmp_path): ...
def test_labels_status_counts_values_unlabelled_and_uncertain(tmp_path): ...
def test_a_filter_on_a_model_facet_without_the_extra_says_what_to_install(tmp_path, fake_embedder, monkeypatch):
    # model.available -> False; filter type=minutes -> payload["filterNotice"] matches r"minirag-mcp\[classify\]"
```

- [ ] **Step 2: Run** — expect FAIL. **Step 3: Implement** `filters.py` and the `results.py` changes. **Step 4: Run** — PASS; gates.
- [ ] **Step 5: Commit** `feat(labels): label and date filters with loud counts, labels in results`

---

### Task 8: Pipeline and sync

**Files:**
- Modify: `src/minirag_mcp/ingest/pipeline.py`, `src/minirag_mcp/sync.py`
- Test: `tests/test_pipeline_labels.py`, `tests/test_sync.py`

**Interfaces:**
- Consumes: Tasks 5–7.
- Produces:
  - `Pipeline.__init__(self, store, embedder, config, labels: LabelerSource | None = None)`
  - `Pipeline.job(self) -> ContextManager[JobReport]`: gets `(labeler, error)` from `labels`. With a labeler, it enters `labeler.job()` and sets a `contextvars.ContextVar` to that labeler for the duration. Without one, it yields a fresh `JobReport`, holding `f"{labels.path} is invalid: {error}"` when there was an error.
  - `ingest_file`, `ingest_data`, `ingest_url` each run inside `with self.job():`. `_chunk_and_store` gains `file_time: float | None = None`; when a labeler is active it calls `labeler.label(...)` with the chunk texts it just cut and writes the five fields on every `ChunkRecord`. `ingest_file` passes `getattr(info, "st_birthtime", info.st_mtime)`.
  - `Pipeline.relabel_stale(self, scopes: tuple[str, ...] = ()) -> tuple[int, list[dict]]`: inside the active job, for every `SourceInfo` in scope whose `label_hash` differs from the active taxonomy's hash, it labels from `store.all_chunks(source)` texts and calls `store.set_document_labels`. It keeps the stored date unless `doc_date_source == ""`. A file's `file_time` comes from its stat when it still exists. Per-source exceptions become `{"source", "error"}` entries. Without a labeler it returns `(0, [])`.
  - `Pipeline.preview_labels(self, path: Path) -> tuple[DocLabels, dict[str, dict[str, float]]]`: parses and chunks exactly as `ingest_file` does, then labels with a trace. No embedding, no store write. Without a labeler it raises `LabelFilterError("document labels are off: …")`.
  - Sync: `SyncJob.counts` and the `_run_sync_unlocked` dict both gain `"relabelled": 0`. `_run_sync_unlocked` runs its ingest, move, delete and re-label work inside `with pipeline.job() as report:`. After the deletes it calls `relabel_stale((str(scope),) if scope else ())`, adds the count, and extends `errors`. Each `report.problems` entry becomes `{"source": str(<taxonomy path>), "error": f"labels: {problem}"}`. None of this touches `failed`.

- [ ] **Step 1: Write the failing tests**

```python
def test_ingest_writes_labels_and_date_on_every_chunk(...): ...
def test_without_classify_toml_nothing_is_labelled_and_no_model_is_made(...): ...  # factory never called, fields ""
def test_a_sync_loads_the_model_once_and_releases_it(...): ...                   # 5 files -> created == 1, closed
def test_editing_the_taxonomy_relabels_without_re_embedding(...):
    # sync; count embed_documents calls and read vectors; edit a description; sync again ->
    # counts["relabelled"] == 5, no new embed_documents call, vectors identical, labels updated
def test_a_file_moved_into_another_folder_gets_its_new_path_label(...):
    # initiative facet by path; move a file between two rule folders -> moved 1, relabelled 1,
    # new initiative value, no embed_documents call
def test_a_missing_model_is_one_error_and_not_a_failure(...):  # failed == 0, one "labels: ..." error, path labels present
def test_an_invalid_taxonomy_is_one_error_and_not_a_failure(...):
def test_sync_counts_include_relabelled_from_the_first_poll(): assert "relabelled" in SyncJob(job_id="x").counts
def test_preview_labels_writes_nothing(...): ...
```

- [ ] **Step 2: Run** — expect FAIL. **Step 3: Implement.** **Step 4: Run** — PASS, including the whole existing `tests/test_sync.py` and `tests/test_pipeline.py`; gates.
- [ ] **Step 5: Commit** `feat(labels): label at ingest, re-label stale sources during sync`

---

### Task 9: MCP and CLI surfaces

**Files:**
- Modify: `src/minirag_mcp/server.py`, `src/minirag_mcp/cli/__init__.py`, `src/minirag_mcp/instructions.py`
- Test: `tests/test_parity.py`, `tests/test_server.py`, `tests/test_cli.py`, `tests/test_instructions.py`

**Interfaces:**
- Consumes: Tasks 5, 7, 8.
- Produces:
  - `create_app(config, *, config_error=None, embedder=None, env=None, model_factory=None)`; `_Ctx` gains `labels: LabelerSource`, built as `LabelerSource(config.roots, config.cache_dir, model_factory or default_model_factory)` and handed to `Pipeline`
  - `query_documents(query: str, topK: int = 8, scope=None, labels: dict[str, str | list[str]] | None = None, dateFrom: str | None = None, dateTo: str | None = None)` → `run_query(..., label_source=c.labels)`
  - `list_files(scope=None, labels=None, dateFrom=None, dateTo=None, sort: str | None = None)` → `list_payload(..., label_source=c.labels)`
  - `status()` merges `labels_status(c.labels, c.store)`
  - CLI: module-level `_make_model_factory() -> Callable | None` (returns `None`, patched in tests like `_make_embedder`); `_load` builds the same `LabelerSource`; `query` and `list` gain `--label` (repeatable `facet=value`; repeating a facet ORs its values), `--date-from`, `--date-to`; `list` gains `--sort`; `status` merges `labels_status`; new sub-app `classify` with command `try` (`minirag-mcp classify try <path> [--json]`) printing each facet's label and the full distribution from `Pipeline.preview_labels`. A malformed `--label` fails with `--label takes facet=value, got 'x'`; a `QueryError` from `list` fails like `query` does.

Pinned text:
- `instructions.py`, `<thin_results>`: replace ``a narrower `scope` —`` with ``a narrower `scope` or `labels` —``. The built-in text stays ≤ 1700 characters; it is 1599 today.
- `query_documents` description, appended paragraph: "`labels` ({facet: value or [values]}) and `dateFrom`/`dateTo` (YYYY-MM-DD, inclusive) narrow the search to labelled documents when the corpus has a classify.toml; `status` lists its facets and values under `labels` — map an approximate name to a value from there. `filterCounts` then says how many documents in scope were left out only for having no label or no date; each `sources` row carries its `labels` and `date`."
- `list_files` description, appended: "`labels`, `dateFrom` and `dateTo` filter as in query_documents; sort=\"date\" lists newest first, documents without a date last. Rows carry `labels` and `date` when present."
- `status` description, appended: "labels, present only when the corpus has a classify.toml: its facets, value descriptions and document counts, unlabelled and uncertain counts, and the model state — or the file's error."

- [ ] **Step 1: Write the failing tests** — parity first, since that is the contract:

```python
async def test_a_label_filter_returns_the_same_sources_and_counts_from_both(...): ...
async def test_a_filter_without_classify_toml_is_refused_the_same_way_by_both(...): ...
async def test_status_reports_the_same_labels_block_in_both(...): ...
async def test_status_has_no_labels_key_without_the_file(...): ...
async def test_status_list_and_query_never_make_the_model(...):
    # classify.toml present, model_factory records creations; status, list_files and a filtered
    # query_documents through the server -> no model created

def test_classify_try_prints_the_distribution_and_writes_nothing(...): ...
def test_a_malformed_label_flag_names_the_expected_form(...): ...
def test_list_sorted_by_date_from_the_cli(...): ...
def test_instructions_mention_labels_within_budget():
    assert "`labels`" in SERVER_INSTRUCTIONS and len(SERVER_INSTRUCTIONS) <= MAX_BUILTIN_INSTRUCTIONS_CHARS
```

- [ ] **Step 2: Run** — expect FAIL. **Step 3: Implement.** **Step 4: Run** the full suite — PASS; gates.
- [ ] **Step 5: Commit** `feat(labels): label and date filters through MCP and CLI, classify try`

---

### Task 10: Documentation and the clean-install gate

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `src/minirag_mcp/server.py` (module docstring only if the tool count changed — it does not: still 11)

- [ ] **Step 1: README** — a "Document labels (optional)" section covering:
  - enabling: the file `<root>/.minirag/classify.toml` and the `[classify]` extra with its compile step (Xcode command-line tools on macOS);
  - the file format: a synthetic example with `project` by path and default, `type` by model, `uncertain_below`, `override_at`;
  - the pinned model: `crh225/plumb-4b-GGUF`, revision `71c1c57e523219124f7be9c486d475d19891185f`, `plumb-4b-v5-Q8_0.gguf`, sha256 `a0d2974e04541b79e167d8acabfb7a7e4b4f93f3b47ac4a006d196d240f68177`, `temperature = 2.07` from the model card, Apache-2.0;
  - "judge by form, not topic" as advice for writing descriptions;
  - what `uncertain` and `filterCounts` mean;
  - the date sources;
  - `minirag-mcp classify try`.
- [ ] **Step 2: CLAUDE.md** — the architecture block gains a `labels/` line ("taxonomy, dates, facet resolution, in-process choice model, filters"); "9 subcommands" becomes 10; the Build & Test block gains `uv run --extra classify env MINIRAG_TEST_GGUF=<gguf> pytest -m classify -q`.
- [ ] **Step 3: Clean-install gate** — `uv sync --exact`, then `uv run pytest -q`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv build`. All green without the extra.
- [ ] **Step 4: Commit** `docs: document labels, the classify extra and its model`

---

### Task 11: Owner-corpus check (outside the repository)

Nothing from this task is committed. Work in `~/projects/test/minirag-labels-eval` (README there); the owner's live corpus and index are not touched.

- [ ] **Step 1:** Write `~/projects/test/minirag-labels-eval/classify.toml` for the owner's corpus (its location and folder layout are in that folder's README):
  - `project`: a path rule for the initiatives folder, `default` for the rollout project;
  - `initiative`: `within` the initiatives folder, one value per initiative folder by path, the "several" value for the cross-initiative folder;
  - `type`: the ten values, descriptions and question from `spike/gold3.py`, with `file_name_label` in the corpus language;
  - `[model]`: `path` pointing at the local Plumb Q8_0 GGUF the spike used.
- [ ] **Step 2:** `ditto` the owner's corpus root (index included) to `<scratch>/corpus`, copy the file into `<scratch>/corpus/.minirag/`, then from the branch: `uv run --extra classify minirag-mcp sync --base-dir <scratch>/corpus --json`. Expect `relabelled` 746 (every source has an empty hash), `failed` 0, no `labels:` error.
- [ ] **Step 3:** A script in the eval folder compares `list --json` labels against `spike/gold3.py` by root-relative path. Expect:
  - type agreement within five documents of 727/746;
  - project and initiative fully path-determined (746/746);
  - every transcript, minutes and mail document dated, with its source.

  Record the numbers in `minirag-mcp-3hm` notes.
- [ ] **Step 4:** Try the owner's three questions through the CLI against the copy and check that each answer is right:
  - `list --label type=transcript --label project=initiatives --date-from <a recent week> --sort date`;
  - `query "что обсуждали" --label type=minutes --label type=transcript --scope <initiative folder>`;
  - `query "<topic>" --label type=spec`.
- [ ] **Step 5:** `uv sync --exact` in the repo to remove the extra; delete the scratch copy.
