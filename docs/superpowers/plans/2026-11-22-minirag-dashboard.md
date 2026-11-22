# Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only overview of every index — what is where and how much, and what did not get in — served by `minirag-mcp serve` at `/dashboard`, backed by a last-sync report every sync writes.

**Architecture:** The core gains `sync_report.py` (the report next to each index, written by `IndexHandle` after every sync) and exposes it through `status` (`lastSync`) and `list` (`lastSyncError`) in both shells. A `dashboard/` package computes per-index statistics from the store (no disk walk), renders self-contained HTML, and serves it behind the existing `LoopbackGuard`, which learns a session cookie for `/dashboard` only. `minirag-mcp dashboard` signs the browser in with a one-time code.

**Tech Stack:** Python ≥3.11, Starlette (via fastmcp), LanceDB store, cyclopts CLI, pytest; stdlib only for HTML, HTTP client (`urllib.request`) and browser (`webbrowser`).

**Spec:** `docs/superpowers/specs/2026-11-22-minirag-dashboard-design.md` (approved by the owner 2026-11-22). Read it first; every task's requirements include it.

## Global Constraints

- No new runtime dependencies. HTML via stdlib and `html.escape`; HTTP client in the CLI via `urllib.request`.
- English for every user-visible string (pages, CLI, errors). Code, comments and commits in English (CLAUDE.md).
- Behaviour lives in the core; MCP and CLI show the same data identically (CLAUDE.md, «Architecture»).
- The page never references anything off the machine: no `http://` or `https://` URL other than the server's own, no web fonts, no CDN.
- The token never appears in a URL, a `Location` header, a page, a log line or an exception message.
- `/mcp/...` keeps its exact current rule: Host, Origin, bearer header. A dashboard cookie never authorizes it.
- Cookie: name `minirag_session`, `HttpOnly`, `SameSite=Strict`, `Path=/dashboard`, `Max-Age=43200`; the server also refuses a session older than 12 h. Login code: single use, valid 60 s.
- Report file: `last-sync.json` in the index folder (`Config.db_path`), written atomically (temp file + `os.replace`), at most 1000 entries in `errors` with `errorsTotal`.
- Gates for every task: `uv run pytest -q`, `uv run ruff check src tests`, `uv run ruff format --check src tests`. Never `uv sync --extra`. TDD: each new test is red on the head before the change (CLAUDE.md, «A test that cannot fail is not coverage»).
- One conventional commit per task ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; do not push.

## Review Focus

- An index that is `unavailable`, `rebuild pending` or `syncing` while a page loads: the page shows its state and reason and every other index normally — never a 500. (Task 3 test `test_an_unavailable_index_gets_its_reason_and_no_statistics`; Task 5 test `test_the_overview_serves_with_one_index_unavailable`.)
- A `last-sync.json` that is truncated, not JSON, or of an unknown format: `status`, `list` and the page say the report is unreadable and carry on. (Task 1 `test_an_unreadable_report_is_reported_not_raised`; Task 2 `test_status_reports_an_unreadable_report`.)
- Folder names on cloud roots arrive NFD while others are NFC: one folder must not appear twice. (Task 3 `test_nfd_and_nfc_spellings_of_a_folder_are_one_folder`.)
- An index of ~250 000 chunks: the statistics are computed once per index version off the event loop; a second page load with nothing changed reads the cache. (Task 3 `test_statistics_are_cached_until_the_index_changes`; Task 5 computes via `anyio.to_thread.run_sync`.)
- The server restarted or the session expired: the dashboard answers 401 with the one line telling the user to run `minirag-mcp dashboard`, never a stack trace. (Task 5 `test_an_expired_session_is_told_how_to_sign_in`.)

---

### Task 1: The last-sync report, written after every sync

**Files:**
- Create: `src/minirag_mcp/sync_report.py`
- Modify: `src/minirag_mcp/sync.py` (the large-table error message at ~line 260 becomes a module constant), `src/minirag_mcp/handle.py` (`run_admitted`, `_run_held`, `_sync_held` callers ~lines 545-610)
- Test: `tests/test_sync_report.py` (new), `tests/test_handle.py`

**Interfaces:**
- Produces, in `sync_report.py`:
  - `REPORT_FILE = "last-sync.json"`, `ERRORS_CAP = 1000`, `FORMAT = 1`
  - `@dataclass(frozen=True) class SyncRecord: started_at: str; finished_at: str; ran_by: str; scope: str | None; outcome: str; error: str | None; counts: dict[str, int]; errors: list[dict]; rebuild: dict | None` — `ran_by` is `"cli"` or `"server"`, `outcome` `"succeeded"` or `"failed"`; `errors` as the sync returns them (`{"source", "error"}`), large tables included.
  - `record_sync(db_path: Path, record: SyncRecord) -> str | None` — merges the record into the file and writes it; returns a one-line warning when it cannot (never raises `OSError` or `ValueError`).
  - `read_report(db_path: Path) -> dict | None` — `None` when the file does not exist; `{"unreadable": "<why>"}` when it exists but is not a report of `FORMAT`; else `{"format": 1, "latest": {...}, "coverage": {...} | None}`.
  - `split_large_tables(errors: list[dict]) -> tuple[list[dict], list[str]]` — failures and the sources of large tables, told apart by `sync.LARGE_TABLE_ERROR_PREFIX`.
- Produces, in `sync.py`: `LARGE_TABLE_ERROR_PREFIX = "skipped: more than "`, used where the large-table error is built.
- File layout: `latest = {startedAt, finishedAt, ranBy, scope, version, outcome, error, counts, rebuild}`; `coverage = {startedAt, finishedAt, ranBy, version, counts, errors, errorsTotal, largeTables}` with `errors` = failures only (≤ 1000), `largeTables` = sources. `coverage` is replaced only when `scope is None and outcome == "succeeded"`; every record replaces `latest`.

- [ ] **Step 1: Write the failing unit tests** in `tests/test_sync_report.py`:
  `test_a_first_full_sync_writes_latest_and_coverage` (both present; `coverage["errors"]` excludes the large-table entry, `coverage["largeTables"] == [that source]`);
  `test_a_scoped_sync_replaces_latest_and_keeps_coverage` (`latest["scope"] == str(scope)`, `coverage` equal to the earlier one);
  `test_a_failed_sync_replaces_latest_and_keeps_coverage` (`latest["outcome"] == "failed"`, `latest["error"]` set);
  `test_errors_are_capped_with_their_total` (1500 failures → `len(errors) == 1000`, `errorsTotal == 1500`);
  `test_the_report_is_written_atomically` (no `*.tmp` left; a crash between write and replace — monkeypatch `os.replace` to raise — leaves the previous file intact and returns a warning);
  `test_an_unreadable_report_is_reported_not_raised` (garbage bytes, `{"format": 99}`, a JSON list → `read_report` gives `{"unreadable": ...}`; `record_sync` over garbage writes a fresh report with `coverage` from this record only if it qualifies, else `None`);
  `test_a_missing_report_reads_as_none`.
- [ ] **Step 2: Run** `uv run pytest tests/test_sync_report.py -q` — expect ImportError/FAIL.
- [ ] **Step 3: Implement** `sync_report.py` as specified; `sync.py` uses `LARGE_TABLE_ERROR_PREFIX` for the message it already builds (text unchanged).
- [ ] **Step 4: Write the failing integration tests** in `tests/test_handle.py` (real sync over a temp root with the fake embedder, as the existing sync tests there do):
  `test_a_cli_sync_writes_the_report` (`handle.run_admitted(...)` → `read_report(db)["latest"]["ranBy"] == "cli"`, counts equal to the returned counts);
  `test_a_background_sync_writes_the_report` (`handle.start_sync()` then wait → `ranBy == "server"`);
  `test_a_sync_that_raises_writes_a_failed_latest` (monkeypatch the scan to raise → exception propagates unchanged, `latest["outcome"] == "failed"`);
  `test_a_rebuild_report_lands_in_the_live_folder` (a `rebuild pending` index synced → `read_report(config.db_path)` has it, and no report in `<db>.next` remains);
  `test_a_report_that_cannot_be_written_leaves_the_sync_result_unchanged` (monkeypatch `record_sync` target dir read-only or `os.replace` raising → same counts/errors returned, the warning reaches `on_event`).
- [ ] **Step 5: Run** them — expect FAIL.
- [ ] **Step 6: Implement** in `handle.py` a private `_sync_reported(scope, on_event, on_rebuild, ran_by)` that stamps `started_at`, calls `_sync_held`, and in all outcomes calls `record_sync` while the sync lock is still held, passing a warning to `on_event` when one comes back; `run_admitted(..., ran_by: str = "cli")` and `_run_held` (`ran_by="server"`) call it. Timestamps in the format `sync._now()` already uses.
- [ ] **Step 7: Run** the gates. **Step 8: Commit** `feat(sync): every sync records a last-sync report next to the index`.

### Task 2: `status` shows the last sync; `list` shows each file's last error

**Files:**
- Modify: `src/minirag_mcp/results.py` (`status_payload` ~line 362, `list_payload` ~line 951), `src/minirag_mcp/server.py:604` and `src/minirag_mcp/cli/__init__.py:746` (callers of `list_payload`), the CLI human output of `status` and `list`
- Test: `tests/test_results.py`, `tests/test_parity.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `read_report`, `split_large_tables` (Task 1).
- Produces: `status_payload(...)["lastSync"] = {"finishedAt", "ranBy", "scope", "outcome", "failed", "largeTables"}` from `latest` (`failed` and `largeTables` from `latest["counts"]`), absent when `read_report` is `None`, `{"unreadable": why}` when unreadable. `list_payload(..., db_path: Path | None = None)`: when given, every row whose `source` is in `coverage["errors"]` gains `"lastSyncError": "<error>"`.
- CLI human `status` prints one line: `last sync: <finishedAt> (<ranBy>[, scope <scope>]) <outcome>, <failed> failed, <largeTables> large tables` or `last sync: no report yet` / `last sync: report unreadable: <why>`. CLI human `list` appends `  ! <error>` under a row that has `lastSyncError`.

- [ ] **Step 1: Failing tests:** `test_status_has_last_sync_after_a_sync`, `test_status_has_no_last_sync_before_any_report`, `test_status_reports_an_unreadable_report` (results-level); `test_list_marks_the_files_the_last_full_sync_failed` (two files, one failing conversion → only it has `lastSyncError`); parity: `test_last_sync_is_identical_in_mcp_and_cli` and `test_last_sync_error_is_identical_in_mcp_and_cli` following the existing pattern in `tests/test_parity.py`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** Both callers pass `db_path=config.db_path`.
- [ ] **Step 4: Run** the gates. **Step 5: Commit** `feat(status,list): last sync in status, each file's last-sync error in list`.

### Task 3: Dashboard data — per-index statistics from the store, cached

**Files:**
- Create: `src/minirag_mcp/dashboard/__init__.py`, `src/minirag_mcp/dashboard/data.py`
- Modify: `src/minirag_mcp/store.py` (add `table_version() -> int` returning `self._table.version`)
- Test: `tests/test_dashboard_data.py`

**Interfaces:**
- Consumes: `IndexRegistry.entries`, `IndexEntry.state()/runtime()/config/description`, `Store.list_sources()`, `status_payload` labels part (`labels_status`), `read_report` (Task 1).
- Produces (all in `dashboard/data.py`, frozen dataclasses):
  - `FolderStats(name: str, path: str, documents: int, chunks: int, children: tuple[FolderStats, ...])` — two levels under a root, largest `chunks` first; `name` NFC-normalised; files directly in a root are the folder `"(files at the root)"`, directly in a first-level folder `"(files in this folder)"`.
  - `RootStats(path: str, documents: int, chunks: int, folders: tuple[FolderStats, ...])`
  - `IndexView(name: str, description: str, state: str, why: str, documents: int, chunks: int, roots: tuple[RootStats, ...], facets: dict, latest: dict | None, coverage: dict | None, report_problem: str | None, exclude: tuple[str, ...], max_table_rows: int | None, model: str, fts_language: str | None, reranker: str | None)` — for an `unavailable` index `documents == chunks == 0`, `roots == ()`, `why` set.
  - `FileRow(source: str, title: str, chunks: int, labels: dict, ingested: str | None)`
  - `class DashboardData: def __init__(self, registry: IndexRegistry)`; `def overview(self) -> list[IndexView]` (in `indexes.toml` order); `def index(self, name: str) -> IndexView | None`; `def folder(self, name: str, path: str) -> list[FileRow] | None` — `None` unless `path` is a root of that index or a folder that holds at least one indexed source of it (string comparison over NFC-normalised source paths; never touches the disk).
  - Cache: per index, keyed on `(store.table_version(), mtime_ns of last-sync.json or None, config identity)`; the views are recomputed only when the key changes.
- Source → root: the longest root that is a path prefix (`scope.is_under`); a source under no root (data/url sources) is counted in the totals but in no root.

- [ ] **Step 1: Failing tests:** `test_documents_and_chunks_add_up_per_root_and_folder` (two roots, nested files → exact numbers); `test_folders_are_ranked_by_chunks`; `test_nfd_and_nfc_spellings_of_a_folder_are_one_folder`; `test_files_at_a_root_get_their_own_row`; `test_an_unavailable_index_gets_its_reason_and_no_statistics` (use the `index_states` fixture); `test_folder_outside_every_root_or_with_dotdot_gives_none` (`"../x"`, `"/etc"`, a root's parent → `None`; a monkeypatched `Path.stat`/`os.scandir` that raises proves no disk access); `test_statistics_are_cached_until_the_index_changes` (spy on `list_sources`: two `overview()` calls → one call; after an ingest → two); `test_the_report_is_attached_to_its_index` (a written report shows as `latest`/`coverage`; an unreadable one as `report_problem`).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Gates.** **Step 5: Commit** `feat(dashboard): per-index statistics from the store, cached by version`.

### Task 4: Dashboard pages — self-contained HTML in the agreed look

**Files:**
- Create: `src/minirag_mcp/dashboard/render.py`, `src/minirag_mcp/dashboard/style.css` (verbatim from Appendix A)
- Test: `tests/test_dashboard_render.py`

**Interfaces:**
- Consumes: `IndexView`, `FileRow`, `FolderStats` (Task 3).
- Produces: `render_overview(views: list[IndexView], *, version: str, generated: str) -> str`; `render_index(view: IndexView, *, version: str, generated: str) -> str`; `render_folder(view: IndexView, path: str, files: list[FileRow], *, version: str, generated: str) -> str`; `render_message(title: str, text: str) -> str` (401/404 pages). CSS read once with `importlib.resources.files("minirag_mcp.dashboard").joinpath("style.css")` and inlined in a `<style>` element.
- Page content (spec «Goals» 1-3 and «Visual direction»; the layout of Appendix B): header band `minirag` + version (not uppercased) + index count + generated time; overview = summary table (Index, State, Documents, Chunks, Last sync, Failed, Large tables, Exclusions; total row) then per index: description, «Where it comes from» roots table with share bars, «Kinds of documents» composition bar from the `type` facet (top 5 + «other kinds», ramp `#002B49 #5E8AB4 #0085CA #8CC8E7 #B2DAEF #D9D9D9`) with an uncertain-count `Read:` line, figure tiles (documents, chunks; failed and large tables only when a coverage exists), «Last sync» line; index page = folders in `<details>` rows (first level summary, second level rows, at most 12 then «… N more folders», each folder name links to its folder page), tiles, «Last sync» definition list from `latest` plus «what did not get in» from `coverage` with its own date, an «Index» panel (state, model, search, large-table limit), «Failed at the last full sync» table, «Skipped as large tables» table, «Kept out by exclusion patterns (N)» list in `<code>`; folder page = files table (Title, Chunks, Kind, Ingested). Links: `/dashboard`, `/dashboard/index/<name>`, `/dashboard/index/<name>/folder?path=<urlencoded>`. States `ready` navy, `rebuild pending`/`syncing` amber, `unavailable` red with the reason. No report: «no report yet» in grey.
- [ ] **Step 1: Failing tests:** `test_a_file_name_with_markup_renders_as_text` (`<script>alert(1)</script>"'&` in a source, title and folder → only escaped forms in the output); `test_the_page_requests_nothing_off_the_machine` (no `http://`/`https://`/`//` in `src=`/`href=`/`url(` and no `@import`); `test_the_version_keeps_its_case` (`2.0.0a2` appears as is); `test_overview_lists_every_index_with_its_totals`; `test_an_index_without_a_report_says_so`; `test_a_failed_latest_is_shown_beside_the_coverage`; `test_the_css_is_inlined`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Gates.** **Step 5: Commit** `feat(dashboard): render the overview, index and folder pages`.

### Task 5: Access and routes — sessions, the guard, `/dashboard`

**Files:**
- Create: `src/minirag_mcp/dashboard/sessions.py`, `src/minirag_mcp/dashboard/app.py`
- Modify: `src/minirag_mcp/http_server.py` (`LoopbackGuard`, `build_http_app`, `serve`)
- Test: `tests/test_dashboard_http.py`; keep `tests/test_http_server.py` green unchanged

**Interfaces:**
- Consumes: `DashboardData` (Task 3), render functions (Task 4), `__version__`.
- Produces:
  - `sessions.py`: `COOKIE = "minirag_session"`, `COOKIE_PATH = "/dashboard"`, `CODE_TTL = 60.0`, `SESSION_TTL = 43200.0`; `class Sessions(clock: Callable[[], float] = time.monotonic)` with `new_code() -> str`, `redeem(code: str) -> str | None` (single use, `None` when unknown/expired), `valid(session_id: str | None) -> bool`. Codes and ids from `secrets.token_urlsafe(32)`, compared with `hmac.compare_digest`.
  - `app.py`: `dashboard_app(registry: IndexRegistry, sessions: Sessions) -> Starlette` with `GET /dashboard`, `GET /dashboard/index/{name}`, `GET /dashboard/index/{name}/folder?path=`, `GET /dashboard/login?code=` (valid → 303 `Location: /dashboard` + `Set-Cookie` per Global Constraints; invalid → 401 page), `POST /dashboard/code` (→ `{"code": "..."}` JSON). Statistics and rendering run in `anyio.to_thread.run_sync`. Unknown index or folder → 404 page.
  - `http_server.py`: `build_http_app(mcp, *, token, allowed_hosts, registry: IndexRegistry | None = None) -> Starlette`. With `registry`, one `Sessions` is created and the guard routes by path: `/` → 303 to `/dashboard` (no auth needed, no data); `/dashboard/login` → passes to the app without auth (the code is the credential); `/dashboard/code` → bearer required; any other `/dashboard...` → valid cookie required, else 401 page «Not signed in. Run `minirag-mcp dashboard` to open this page signed in.»; everything else → today's rule, cookie ignored. Host and Origin rules first, for every path. `serve` passes its registry.
- [ ] **Step 1: Failing tests** (served over real HTTP with `serving()` like `tests/test_http_server.py`): `test_mcp_with_a_dashboard_cookie_and_no_bearer_is_refused`; `test_a_dashboard_page_without_a_cookie_is_refused_with_how_to_sign_in`; `test_a_code_signs_in_once` (second redeem 401); `test_a_code_expires_after_a_minute` (fake clock); `test_an_expired_session_is_told_how_to_sign_in` (fake clock past 12 h); `test_the_cookie_is_scoped_to_dashboard` (`Set-Cookie` has `Path=/dashboard; HttpOnly; SameSite=Strict; Max-Age=43200`); `test_the_token_appears_nowhere` (code request, login redirect, every page, captured logs: no `TOKEN` substring); `test_a_code_needs_the_bearer` (`POST /dashboard/code` without it → 401); `test_a_foreign_origin_is_refused_on_dashboard_paths`; `test_the_root_redirects_to_the_dashboard`; `test_the_overview_serves_with_one_index_unavailable` (via `index_states`); `test_a_folder_outside_the_index_is_not_found`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Gates**, and `tests/test_http_server.py` unchanged and green. **Step 5: Commit** `feat(serve): /dashboard behind a session cookie, signed in by a one-time code`.

### Task 6: `minirag-mcp dashboard`, docs, packaging

**Files:**
- Modify: `src/minirag_mcp/cli/__init__.py` (new command after `serve`), `README.md` (new section «Dashboard» after the `serve` section, and the CLI command table), `CLAUDE.md` (architecture block: `cli/ 14 subcommands`, lines for `sync_report.py` and `dashboard/`), `docs/superpowers/specs/2026-11-22-minirag-dashboard-design.md` only if a decision changed
- Test: `tests/test_cli_index.py` (or a new `tests/test_cli_dashboard.py`)

**Interfaces:**
- Consumes: the server settings of `indexes.toml` (`registry.server.host/port/token_file` as `serve` reads them), `read_token`, `POST /dashboard/code` (Task 5).
- Produces: `minirag-mcp dashboard [--no-open] [--config PATH]`. It asks the running server for a code with `urllib.request` (bearer header, 5 s timeout) and opens `http://<host>:<port>/dashboard/login?code=<code>` with `webbrowser.open`, printing `Opened the minirag dashboard in your browser.`; with `--no-open` it prints the URL alone on stdout. A server that does not answer → exit 1 with `No minirag server answers at http://<host>:<port>. Start it with: minirag-mcp serve` (the LaunchAgent starts it at login). The token is never printed.
- [ ] **Step 1: Failing tests:** `test_dashboard_prints_a_signed_in_url_without_the_token` (serve the app on a free port, write that port into the test `indexes.toml`, run with `--no-open`, then GET the printed URL with redirects → 200 overview); `test_dashboard_without_a_server_fails_in_one_line`; `test_dashboard_opens_the_browser` (monkeypatch `webbrowser.open`, assert it got the URL).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Docs** (README: what the page shows, `minirag-mcp dashboard`, the cookie and its limits including the host-not-port residual risk, the report file). **Step 5: Gates**, and `uv build` then `unzip -l dist/*.whl | grep dashboard/style.css` shows the file. **Step 6: Commit** `feat(cli): minirag-mcp dashboard opens the page signed in`.

### Task 7 (controller): review, merge, release 2.0.0a2, switch the owner's server

Not a subagent task. After the whole-branch review is clean: PR into `main`, merge; `uv run bump-my-version bump pre_n` (2.0.0a1 → 2.0.0a2) on a clean `main`; push commit and tag over SSH 443; `gh release create v2.0.0a2 --verify-tag --prerelease --notes-file <notes>`; wait for `release.yml` and PyPI. Then the owner's machine, each file backed up first: the LaunchAgent pin, `~/vaults/solidcore/scripts/minirag.sh` and the `mr` helper to `==2.0.0a2`, the constraints header; restart the LaunchAgent; `minirag-mcp dashboard --no-open` and a real page load; rollback steps into the private handover.

---

## Appendix A: `src/minirag_mcp/dashboard/style.css` (verbatim)

```css
:root{--navy:#002B49;--bright:#0085CA;--steel:#5E8AB4;--b60:#8CC8E7;--b30:#B2DAEF;--wash:#E8F2F9;--panel:#F2F6F9;
--hair:#E5E9EE;--body:#3B3939;--source:#646464;--caption:#767171;--rule:#969696;--grid:#D9D9D9;--band:#F2F2F2;
--good:#29702A;--bad:#9E2A2B;--amber:#CF7F00;color-scheme:light}
*{box-sizing:border-box}
html{font-family:Arial,Helvetica,sans-serif;color:var(--body);background:#fff;font-size:15px;line-height:1.45;
-webkit-text-size-adjust:100%}
body{margin:0}
::selection{background:var(--b30);color:var(--navy)}
a{color:var(--navy);text-decoration:none}
a:hover{text-decoration:underline;text-decoration-color:var(--bright);text-underline-offset:4px;text-decoration-thickness:2px}
td a{white-space:nowrap}
a:focus-visible,summary:focus-visible{outline:2px solid var(--bright);outline-offset:2px}
.wrap{max-width:1240px;margin:0 auto;padding:0 32px}
header.top{background:var(--navy);color:#fff}
header.top .wrap{display:flex;align-items:baseline;gap:24px;padding-top:18px;padding-bottom:18px;flex-wrap:wrap}
.brand{font-weight:700;font-size:20px;letter-spacing:.04em;text-transform:uppercase;color:#fff}
.meta{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#B2DAEF}
.meta b{color:#fff;font-weight:700}
.ver{text-transform:none;letter-spacing:.04em}
.crumbs{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#B2DAEF}
.crumbs a{color:#fff}
h1{font-size:26px;line-height:1.15;margin:40px 0 6px;color:var(--navy);text-transform:uppercase;letter-spacing:.03em}
h2{font-size:20px;line-height:1.2;margin:0;color:var(--navy);text-transform:uppercase;letter-spacing:.03em}
h3{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--navy);margin:28px 0 10px;display:flex;align-items:center;gap:8px}
h3::before{content:'';width:8px;height:8px;background:var(--bright);flex:none}
p.lead{color:var(--navy);font-size:15px;margin:0 0 28px;max-width:75ch}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;color:var(--navy)}
thead th{background:var(--navy);color:#fff;font-weight:700;font-size:11px;letter-spacing:.1em;text-transform:uppercase;
text-align:left;padding:10px 10px;white-space:nowrap}
thead th.num{text-align:right}
tbody td{padding:9px 10px;border-bottom:.5px solid var(--rule);vertical-align:middle}
tbody tr:nth-child(even) td{background:var(--band)}
tfoot td{padding:9px 10px;font-weight:700;background:var(--band);border-bottom:.5px solid var(--rule)}
.state{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}
.state.ok{color:var(--navy)} .state.warn{color:var(--amber)} .state.bad{color:var(--bad)}
.bad{color:var(--bad);font-weight:700} .good{color:var(--good);font-weight:700} .dim{color:var(--caption)}
.idx{border-top:1px solid var(--navy);padding:26px 0 34px}
.idx-head{display:flex;justify-content:space-between;gap:24px;align-items:baseline;flex-wrap:wrap}
.idx-desc{max-width:75ch;margin:8px 0 0;color:var(--body);font-size:14px}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:32px;margin-top:20px}
.tiles{display:grid;grid-template-columns:1fr 1fr;border-top:.5px solid var(--rule)}
.tile{padding:12px 0 12px;border-bottom:.5px solid var(--rule)}
.tile .fig{font-size:26px;font-weight:700;color:var(--navy);font-variant-numeric:tabular-nums;line-height:1.1}
.tile .cap{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--caption);margin-top:4px;font-weight:700}
.bar{display:flex;height:10px;width:100%;background:var(--hair)}
.bar span{display:block;height:100%}
.share{display:flex;align-items:center;gap:10px}
.share .bar{flex:1;min-width:60px;height:8px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:10px;font-size:12px;color:var(--source)}
.legend i{display:inline-block;width:9px;height:9px;margin-right:6px;vertical-align:-1px}
.panelbar{background:var(--navy);color:#fff;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
padding:8px 12px}
.panel{background:var(--panel);padding:12px 12px 14px;margin-bottom:20px;font-size:13px;color:var(--navy)}
dl{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;margin:0}
dt{font-weight:700} dd{margin:0}
.read{font-size:13px;margin:10px 0 0;color:var(--caption)} .read b{color:var(--navy)}
details{border-bottom:.5px solid var(--rule)}
details>summary{list-style:none;cursor:pointer;display:grid;grid-template-columns:18px minmax(0,1fr) 90px 110px 180px;gap:10px;
align-items:center;padding:9px 10px;color:var(--navy);font-size:14px}
details>summary::-webkit-details-marker{display:none}
details>summary .tw{width:0;height:0;border-left:5px solid var(--bright);border-top:4px solid transparent;border-bottom:4px solid transparent;
transition:transform .15s ease-out}
details[open]>summary .tw{transform:rotate(90deg)}
details>summary:hover{background:var(--wash)}
.sub{display:grid;grid-template-columns:18px minmax(0,1fr) 90px 110px 180px;gap:10px;padding:6px 10px;font-size:13px;color:var(--body);
border-top:.5px solid var(--hair)}
.sub:nth-child(even){background:#FAFBFC}
.thead{display:grid;grid-template-columns:18px minmax(0,1fr) 90px 110px 180px;gap:10px;background:var(--navy);color:#fff;
font-weight:700;font-size:11px;letter-spacing:.1em;text-transform:uppercase;padding:10px 10px}
.root{margin:22px 0 6px;font-size:13px;color:var(--navy);font-weight:700}
.root .dim{font-weight:400}
code{font-family:Menlo,Consolas,monospace;font-size:12.5px;color:var(--navy);background:var(--panel);padding:1px 5px}
ul.pat{list-style:none;padding:0;margin:0}
ul.pat li{padding:6px 0;border-bottom:.5px solid var(--hair);font-size:13px}
footer{margin:48px 0 40px;font-size:12px;color:var(--source);border-top:.5px solid var(--rule);padding-top:12px}
.note{border:1.5px solid var(--bright);background:var(--wash);padding:10px 14px;font-size:13px;color:var(--navy);margin:20px 0 0}
@media (max-width:900px){.grid{grid-template-columns:minmax(0,1fr)}header.top .wrap{gap:6px 16px}.wrap{padding:0 16px}
details>summary,.sub,.thead{grid-template-columns:14px minmax(0,1fr) 64px 84px}
details>summary>:nth-child(5),.sub>:nth-child(5),.thead>:nth-child(5){display:none}}
```

## Appendix B: layout reference

```
minirag   5 indexes · 127.0.0.1:8765                       2.0.0a2 · generated 22.11.2026 14:20
WHAT THE INDEXES HOLD
5 indexes, 2 123 documents in 487 478 chunks. Counts come from the indexes themselves; …
| INDEX | STATE | DOCUMENTS | CHUNKS | LAST SYNC | FAILED | LARGE TABLES | EXCLUSIONS |
| alpha | READY |       212 | 85 318 | 22.11 14:12 · cli | 0 | 3 | 28 |
…
ALPHA                                                                       READY
<description>
| WHERE IT COMES FROM | DOCUMENTS | CHUNKS | SHARE OF CHUNKS |      212        85 318
| ~/Documents/…       |        49 | 24 560 | ███░░░░ 29%     |      DOCUMENTS  CHUNKS
■ KINDS OF DOCUMENTS  ██████████▒▒▒▒▒▓▓▓░░  report 102 · data 38 · …   Last sync: …
```
