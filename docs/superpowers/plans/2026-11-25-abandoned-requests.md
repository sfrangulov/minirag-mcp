# Abandoned Requests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a client gives up on a call, the handler returns at once with no second
response, and a read stops its work at the next checkpoint.

**Architecture:** `guard` in `server.py` becomes async. It runs the synchronous tool body
through a new `abandon.run_abandonable`, which starts a worker thread with
`abandon_on_cancel=True`. A `ContextVar` holding a `threading.Event` travels into that
thread; on cancellation the event is set and the cancellation re-raised. The read paths in
`results.py` call `abandon.checkpoint()`, which raises `Abandoned` once the event is set
and does nothing outside a server call.

**Tech Stack:** Python 3.11+, FastMCP 3.4, mcp 1.29, anyio 4, pytest (asyncio auto mode).

**Spec:** `docs/superpowers/specs/2026-11-25-abandoned-requests-design.md`

## Global Constraints

- Writes get no checkpoints. An abandoned ingest, delete, sync or label write runs to the
  end in its thread.
- `Abandoned` derives from `BaseException`, so neither `guard`'s `except Exception` nor
  the core's fallback handlers (rerank, embedding) can swallow it or turn it into a
  degraded result.
- `checkpoint()` outside a server call is a no-op; the CLI must behave exactly as before.
- Every fix test must go red on the code before the fix (CLAUDE.md, "A test that cannot
  fail is not coverage"). Run each one against the unfixed code and record that it failed.
- Gates: `uv run pytest -q`, `uv run ruff check src tests`,
  `uv run ruff format --check src tests`.

## Review Focus

1. **A client that abandons by closing its HTTP session without sending
   `notifications/cancelled`.** This is exactly what Skach does. The handler is cancelled
   by the session's task group, not by a responder. Covered by the HTTP tests in Task 2,
   which abandon that way.
2. **Two concurrent calls on one index, one abandoned.** The other must finish normally.
   Each call has its own event, created per call, never module-global. Pinned by
   `test_a_concurrent_call_is_not_stopped_by_another_abandoned_one` in Task 1.
3. **Rerank or embedding fallback during an abandoned query.** `Abandoned` raised inside
   the `try` of `_Recorder.__call__` must not become a rerank warning. Because it is a
   `BaseException` it cannot; the checkpoint sits before the `try` regardless.
4. **The stdio server (`minirag-mcp --index NAME`).** It goes through the same `guard`;
   no separate path is needed. The existing stdio and entry-point tests must stay green.
5. **Tool schemas.** FastMCP builds each tool's input schema from the signature `guard`
   exposes. `test_all_thirteen_tools_listed` and the existing schema tests must stay
   green, and the schema of `query_documents` must keep its parameters.

---

### Task 1: `abandon` module and an async `guard`

**Files:**
- Create: `src/minirag_mcp/abandon.py`
- Modify: `src/minirag_mcp/server.py` (`guard`, inside `create_app`)
- Modify: `CLAUDE.md` (architecture list: one line for `abandon.py`)
- Test: `tests/test_abandon.py`

**Interfaces:**
- Produces:
  - `class Abandoned(BaseException)`
  - `def checkpoint() -> None`: raises `Abandoned` when the current call's stop event is
    set; returns otherwise, including outside any call.
  - `async def run_abandonable(fn: Callable[[], T]) -> T`: runs `fn` in a worker thread
    via `anyio.to_thread.run_sync(fn, abandon_on_cancel=True)` with a fresh
    `threading.Event` set in the module's `ContextVar` for that call. On the cancellation
    exception (`anyio.get_cancelled_exc_class()`) it sets the event and re-raises.

- [ ] **Step 1: Write the failing tests** in `tests/test_abandon.py`:
  - `test_checkpoint_outside_a_call_does_nothing`: `checkpoint()` returns `None`.
  - `test_an_abandoned_call_stops_at_its_next_checkpoint`:
    - `fn` loops for up to 5 s, calling `checkpoint()` and `time.sleep(0.01)`. It sets
      `ended` in a `finally` and records whether it left by `Abandoned`.
    - Run `run_abandonable(fn)` under `anyio.move_on_after(0.1)`.
    - The `await` returns within 0.3 s of starting.
    - `ended.wait(1.0)` is `True` and the exit was by `Abandoned`.
  - `test_a_finished_call_returns_its_value`:
    `await run_abandonable(lambda: 42) == 42`.
  - `test_a_concurrent_call_is_not_stopped_by_another_abandoned_one`:
    - Two calls run in one task group. Each `fn` loops for 0.5 s, calling
      `checkpoint()`.
    - Only the first runs under `move_on_after(0.1)`.
    - The second returns its value; the first ended by `Abandoned`.

- [ ] **Step 2: Run and see them fail**

  Run: `uv run pytest tests/test_abandon.py -q`
  Expected: collection error, `ModuleNotFoundError: minirag_mcp.abandon`.

- [ ] **Step 3: Implement `src/minirag_mcp/abandon.py`** to the interfaces above. The
  module docstring states the constraint the code cannot show: an `Exception` subclass
  would be swallowed by the core's fallback handlers and turned into a degraded answer
  that keeps computing.

- [ ] **Step 4: Run the tests and see them pass**

  Run: `uv run pytest tests/test_abandon.py -q`
  Expected: 4 passed.

- [ ] **Step 5: Make `guard` async.**
  - `inner` keeps its body (the `_ANSWERING` set and reset, and the `ToolError` mapping)
    as a nested synchronous function.
  - `guard` returns `async def outer(*args, **kwargs)`, decorated with `@wraps(fn)`,
    which does
    `return await run_abandonable(functools.partial(inner, *args, **kwargs))`.
  - `@wraps` keeps the signature and annotations that FastMCP reads.
  - `_ANSWERING` and the HTTP request context reach the thread in the copied context,
    as they do today through FastMCP's own `run_sync`.

- [ ] **Step 6: Run the whole fast suite**

  Run: `uv run pytest -q`
  Expected: all passed. Any failure here is a change in behavior and must be understood,
  not papered over.

- [ ] **Step 7: Add the line for `abandon.py` to the CLAUDE.md architecture list:**
  `abandon.py  a call its client gave up on stops at the next checkpoint`.

- [ ] **Step 8: Lint, format and commit**

  ```bash
  uv run ruff check src tests && uv run ruff format --check src tests
  git add src/minirag_mcp/abandon.py src/minirag_mcp/server.py tests/test_abandon.py CLAUDE.md
  git commit -m "feat(server): a call its client abandons ends at once instead of answering twice"
  ```

### Task 2: Checkpoints in the read paths, proven over HTTP

**Files:**
- Modify: `src/minirag_mcp/results.py`:
  - the scan loop in `run_find`;
  - `run_query`, after the embedding and after `store.search`;
  - `_Recorder.__call__`, before the `try`.
- Modify: `tests/http_helpers.py` (a helper that abandons a call the way Skach does)
- Test: `tests/test_abandoned_calls.py`

**Interfaces:**
- Consumes: `abandon.checkpoint()` from Task 1.
- Produces, in `tests/http_helpers.py`:
  `async def abandon_call(base: str, index: str, tool: str, args: dict, after_s: float) -> None`.
  - It opens `mcp.ClientSession` over `streamable_http_client(f"{base}/mcp/{index}", ...)`
    with an `httpx.AsyncClient(headers=AUTH)`, then initializes and calls `tool`.
  - It leaves the session when `asyncio.timeout(after_s)` fires and swallows that
    `TimeoutError`.
  - It never sends `notifications/cancelled`.

- [ ] **Step 1: Write the failing tests** in `tests/test_abandoned_calls.py`, over the
  `site` fixture. Each test first ingests its corpus through a normal call to the
  `alpha` index.
  - `test_an_abandoned_find_mentions_stops_scanning`:
    - The corpus is a Markdown file in `indexes_file.roots["alpha"]` that yields at least
      20 chunks, each mentioning `apricot`.
    - Monkeypatch `minirag_mcp.literal.matches` with a wrapper that sleeps 0.05 s per
      call, appends `time.monotonic()` to a list, and delegates.
    - Call `abandon_call(..., "find_mentions", {"terms": ["apricot"]}, after_s=0.3)`, then
      `await asyncio.sleep(1.5)`.
    - Assert that the last recorded call came less than 0.5 s after the abandonment.
    - Assert that fewer calls were made than a full, unabandoned scan of the same corpus
      makes.
    - Also assert, with `caplog`, that no record contains "Request already responded
      to". Keep this assertion only if it goes red on the unfixed code; otherwise delete
      it and say so in the commit message.
  - `test_an_abandoned_query_does_not_search_after_embedding`:
    - Monkeypatch the `fake_embedder` fixture's `embed_query`, the same instance the
      `site` registry uses, so that it sleeps 1.0 s first.
    - Monkeypatch `Store.search` to record that it was called.
    - Call `abandon_call(..., "query_documents", {"query": "apricot"}, after_s=0.3)`,
      then `await asyncio.sleep(1.5)`.
    - Assert `Store.search` was never called.
  - `test_an_abandoned_ingest_still_completes`:
    - This is a guard for the anti-goal. It passes on the old code too; that is stated in
      its docstring.
    - Monkeypatch the parser so that `ingest_file` takes 1.0 s.
    - Abandon an `ingest_file` after 0.3 s, then wait 2 s.
    - A normal `list_files` call shows the file.

- [ ] **Step 2: Run them against the code without checkpoints and see the first two fail**

  Run: `uv run pytest tests/test_abandoned_calls.py -q`
  Expected:
  - `test_an_abandoned_find_mentions_stops_scanning` FAILS: the scan runs to the end.
  - `test_an_abandoned_query_does_not_search_after_embedding` FAILS: `Store.search` is
    called.
  - The ingest guard passes.

- [ ] **Step 3: Add the checkpoints.** Each is one `checkpoint()` call:
  - in `run_find`, at the top of the `for chunk in store.literal_candidates(...)` body;
  - in `run_query`, right after the embedding and right after `store.search(...)`
    returns;
  - in `_Recorder.__call__`, before its `try`.

  No checkpoint goes into `store.py`, `pipeline.py`, `sync.py` or any write path.

- [ ] **Step 4: Run them and see them pass, then run the whole fast suite**

  Run: `uv run pytest tests/test_abandoned_calls.py -q && uv run pytest -q`
  Expected: all passed.

- [ ] **Step 5: Lint, format and commit**

  ```bash
  uv run ruff check src tests && uv run ruff format --check src tests
  git add src/minirag_mcp/results.py tests/http_helpers.py tests/test_abandoned_calls.py
  git commit -m "feat(search): an abandoned find_mentions or query stops at its next checkpoint"
  ```

### Task 3: Ship 2.0.0a3 and put it under the owner's server

The owner asked for this release on 2026-11-25.

- [ ] **Step 1: Push the branch and open a PR.** The body states the measurement from the
  spec, ends with the attribution line, and links the spec. Wait for CI to go green.
- [ ] **Step 2: Merge the PR into `main`, then update local `main`:**
  `git switch main && git pull --rebase`.
- [ ] **Step 3: Bump and tag 2.0.0a3.**
  - First run `uv run bump-my-version bump pre_n --dry-run --verbose`. Expected:
    2.0.0a2 → 2.0.0a3.
  - Then run it without `--dry-run`. It commits `chore: release v2.0.0a3` and tags
    `v2.0.0a3`.
- [ ] **Step 4: Push and publish.**
  - `git push && git push origin v2.0.0a3`.
  - `gh release create v2.0.0a3 --verify-tag --prerelease --notes-file <notes>`. The notes
    are a few lines in the style of `gh release view v2.0.0a2`.
  - Watch `release.yml` until it publishes.
- [ ] **Step 5: Upgrade the owner's install and restart the server.**
  - `uv tool upgrade minirag-mcp`. If the installed range excludes pre-releases, use
    `uv tool install --prerelease allow 'minirag-mcp>=2.0.0a3,<3'`.
  - Check that no Skach session is running: `~/.cache/skach/session.json` is absent, or
    its pid is dead.
  - `launchctl kickstart -k gui/$(id -u)/local.minirag-mcp`.
  - Confirm that `minirag-mcp --version` prints 2.0.0a3 and that `/mcp/aida` answers.
- [ ] **Step 6: Measure on the live server.**
  - Replay the AIDA record through Skach's retriever, using the scratch `replay.py` from
    the investigation.
  - Confirm that the server's CPU drops between calls once each one is abandoned, and
    that the log gains no "Request already responded to".
- [ ] **Step 7: File a bd issue for anything left over, and close the session per
  CLAUDE.md.**
