# minirag v2, plan 3: the index server. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One long-running `minirag-mcp serve` hosts every index in `indexes.toml` over
token-guarded loopback streamable HTTP. Every tool takes an optional `index`, a new
`list_indexes` tool lists them, `minirag-mcp connect` bridges Claude Desktop's stdio to
the server, and the CLI gains `--index`. The env-var single-index mode runs through the
same registry.

**Architecture:** `indexes.py` turns `indexes.toml` into one plan-1 `Config` per index. It
does this by handing `load_config` a synthesized environment, so every value is checked
by the same code that checks it today. `registry.py` holds one plan-1 `IndexHandle` per
index (the handle opens the index lazily, follows its recorded scheme and picks up a
swapped generation; the registry builds nothing of its own beyond a pool that lets indexes
of one model share an embedder), reloads the list when the file changes or on SIGHUP, and
owns every refusal. The env-var mode is a one-entry registry whose index is named `default`. `server.py` reads the
connection (its default index and allow-list) from the HTTP request's ASGI scope. That
scope is set by middleware in `http_server.py`, which also enforces the token, `Host` and
`Origin` before FastMCP sees the request. `bridge.py` is a FastMCP proxy served over stdio.

**Tech Stack:** Python >= 3.11, FastMCP >= 3.4 (`FastMCP.http_app`,
`fastmcp.server.dependencies.get_http_request`, `fastmcp.server.create_proxy`), uvicorn,
Starlette middleware, `tomllib`, cyclopts meta app, httpx (tests).

**Spec:** `docs/superpowers/specs/2026-11-19-minirag-v2-design.md`. This plan implements
section 1, plus the "which backend is live" part of section 2 for `list_indexes`. It uses
the names in `docs/superpowers/plans/2026-11-19-minirag-v2-00-contract.md`. Plans 1 and 2
have already landed.

## Decisions taken from the installed packages (FastMCP 3.4.6, cross-checked on 3.4.0)

- **One app, one route.** `mcp.http_app(path="/mcp", middleware=[...])` serves a single
  route. `IndexPathMiddleware` rewrites `/mcp/<index>` to `/mcp` and puts the parsed
  `Connection` into the ASGI scope under `CONNECTION_SCOPE_KEY`. A tool reads it with
  `get_http_request().scope`. This was verified on FastMCP 3.4.0 (Python 3.11) and 3.4.6:
  the scope key and the query string reach the tool. With no HTTP request (stdio, or the
  in-memory test client), `get_http_request()` raises `RuntimeError`. There is no mount per
  index, so a reload needs no route changes.
- **The Host/Origin/token guard is our own ASGI middleware.**
  `http_app(host_origin_protection=...)` only exists from FastMCP 3.4.3. Versions
  3.4.0–3.4.2 lack it, and `fastmcp>=3.4` admits them. It is also off by default, and the
  `FASTMCP_HTTP_ALLOWED_HOSTS` environment variable can widen it. Do not pass it.
- **The bridge** is `create_proxy(Client(StreamableHttpTransport(url, headers=...)),
  name=..., instructions=...)`. Verified: the header and the query string reach the
  server, a `ToolError`'s text passes through, and `instructions` can be set. When the
  server is down the proxy fails with "Client failed to connect: All connection attempts
  failed", which does not name the URL. So `connect` checks the port first.
- **SIGHUP.** uvicorn 0.52 handles SIGINT and SIGTERM only, so SIGHUP is free for
  `loop.add_signal_handler`.
- **Config directory (coordinator's decision, now in spec section 1).** platformdirs 4.11
  returns `~/Library/Application Support/minirag` on macOS, so it is used on Windows only.
  `default_config_path()` resolves in this order:
  1. `$MINIRAG_CONFIG` when it is set.
  2. On macOS, Linux and any other non-Windows platform,
     `Path($XDG_CONFIG_HOME or ~/.config) / "minirag" / "indexes.toml"`.
  3. On Windows (`sys.platform == "win32"`),
     `platformdirs.user_config_path("minirag") / "indexes.toml"`.

  The token file defaults to the same directory as the `indexes.toml` in use.

## What plans 1 and 2 left, and what this plan adds

Consumed (`docs/superpowers/plans/2026-11-19-minirag-v2-01-core.md`, `-02-ingest.md`):
- `handle.IndexHandle(config, *, embedder_factory=None, model_factory=None)` with `get() ->
  OpenIndex` (opens once under a lock, retries a failed open, reopens on a new generation),
  `state() -> tuple[IndexState, str]` (never raises; an open failure is `(UNAVAILABLE,
  str(error))`; a running job is `SYNCING`; else `index_state`, which plan 3's tests patch
  as `minirag_mcp.handle.index_state`), `.sync -> SyncManager`, `run_sync(scope, on_event)`,
  `notes()`. `OpenIndex(config, store, embedder, pipeline, labels, identity, reranker)`.
  `EmbedderFactory = Callable[[EmbeddingScheme], object]`; `configured_scheme(config)`;
  `make_embedder(config, scheme, *, threads=None)`; `IndexUnavailableError`.
- `results.status_payload(config, handle)` (the whole per-index status, with `indexState`,
  `indexGeneration`, `embedding`, `embeddingBackend`, `reranker`, `chunkBudget`,
  `rebuildReasons`, `indexError`), `results.rerank_status(reranker)`,
  `LabelerSource.backend`; per-result `embeddingBackend`, `reranked`, `warnings`.
- `Config.embed_endpoints: Mapping[str, Endpoint]` keyed by fastembed model id,
  `rerank_endpoint`, `classify_endpoint`, `index_name`, `document_weight`, `rerank_top`
  (plan 1); `sync_workers`, `ocr_lang` in Tesseract spelling (plan 2).
- `SyncManager.running` (a job pending or running); `FakeEmbedder.backend`, `warning`,
  `prefixes`, `max_tokens` (plan 1's conftest).

Added here (contract file lists them):
- `indexes.py`: `ServerConfig`, `load_server_config`, `default_config_path(env=None)`
  (`None` means `os.environ`; platformdirs on Windows only).
- `registry.py`: `Connection`, `Use`, `IndexRegistry`, `IndexEntry` (wraps one
  `IndexHandle`), `EmbedderPool` (over `make_embedder`), `IndexRefused`, `IndexUnavailable`.
- `results.with_index` (per-result `index`, `indexState`, `indexWarning`: plan 3's),
  `index_status`, `registry_status`, `list_indexes_payload`.

## Global Constraints

- Python >= 3.11; everything runs through `uv`; CI runs 3.11–3.14 on Linux and macOS.
  Use no FastMCP API absent from 3.4.0.
- Config `~/.config/minirag/indexes.toml` on macOS and Linux (`$XDG_CONFIG_HOME` is
  honoured), and platformdirs on Windows. `MINIRAG_CONFIG` overrides it; the CLI flag
  `--config` overrides both.
- Token file `~/.config/minirag/token`, i.e. the same directory as the `indexes.toml` in
  use; `[server] token_file` overrides it. Mode `0600`, generated on the first `serve`. A
  bearer token is required even on loopback.
- `[server]` defaults: `host = "127.0.0.1"`, `port = 8765`. A non-loopback `host` is
  refused unless `allow_remote = true`, which is not documented.
- Index names are lowercase letters, digits and `-`. The default `db` is
  `<first root>/.minirag/v2`. `classify.toml` stays at `<first root>/.minirag/classify.toml`.
- URLs: `/mcp/<index>` makes `<index>` the default. `?allow=a,b` restricts the connection
  to its default plus `a` and `b`. `/mcp` has no default.
- The 12 tools keep their names and contracts, and each gains `index: str | None = None`.
  One call touches exactly one index. Every result carries `"index"`. Writes are allowed
  into any reachable named index.
- Refusals are loud. They name the index and the reachable ones, and never return an
  empty result.
- `Host` must be a loopback name. A non-loopback `Origin` is refused.
- Claude Code truncates instructions at 2048 characters. The built-in text stays
  `<= MAX_BUILTIN_INSTRUCTIONS_CHARS` (1700).
- MCP and CLI stay identical; behaviour lives in the core. Write English everywhere.
  Comments state constraints only.
- Every new test must fail on unfixed code. Each security test is run once against the
  app with its guard removed, and must go red.
- After every task: `uv run pytest -q`, `uv run ruff check src tests` and
  `uv run ruff format --check src tests` are green. Then commit, ending the message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A half-saved or broken `indexes.toml` while `serve` runs** (a syntax error, or a
   deleted file). The server keeps serving the previous list, and `list_indexes` and
   `status` carry `configError` naming the line. Nothing disappears silently. Tests:
   Task 3, `test_a_broken_edit_keeps_the_previous_indexes_and_says_why`; Task 4,
   `test_a_broken_edit_is_reported_and_the_old_indexes_keep_answering`.
2. **A typo in a key** (`tokn_budget`, `fts_langauge`) is a `ConfigError` naming the key
   and the table, not a setting silently left at its default. Test: Task 1,
   `test_unknown_keys_are_refused`.
3. **Two indexes resolving to one index directory** (same first root, default `db`) is a
   `ConfigError` naming both indexes. Two indexes must never write one LanceDB directory.
   Test: Task 1, `test_two_indexes_sharing_an_index_directory_are_refused`.
4. **A relative `roots` or `db`.** launchd starts the server with cwd `/`, so such a path
   is refused as "must be absolute or start with ~". Test: Task 1,
   `test_a_relative_path_is_refused`.
5. **`connect` while `serve` is not running yet** (Claude Desktop starting at login before
   the LaunchAgent) waits up to `CONNECT_WAIT_SECONDS`, then exits 1 naming the URL. Test:
   Task 8, `test_connect_fails_loudly_when_the_server_is_not_running`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/minirag_mcp/indexes.py` (new) | `indexes.toml` -> `ServerConfig`; TOML keys -> `load_config` env names; default path |
| `src/minirag_mcp/registry.py` (new) | `Connection`, `Use`, refusals, `IndexEntry` over plan 1's `IndexHandle`, `EmbedderPool`, `IndexRegistry` (single and file mode, reload) |
| `src/minirag_mcp/http_server.py` (new) | token file, URL -> `Connection`, ASGI guard and path middleware, `build_http_app`, `serve` |
| `src/minirag_mcp/bridge.py` (new) | `bridge_url`, `build_bridge`, `wait_for_server`, `run_bridge` |
| `src/minirag_mcp/security.py` | `LOOPBACK_NAMES`, `host_name`, `is_loopback_name` |
| `src/minirag_mcp/results.py` | `with_index`, `index_status`, `registry_status`, `list_indexes_payload` |
| `src/minirag_mcp/server.py` | registry-backed `create_app`, `index` on every tool, `list_indexes` |
| `src/minirag_mcp/instructions.py` | `<indexes>` block |
| `src/minirag_mcp/cli/__init__.py`, `__main__.py` | meta app (`--index`, `--config`), registry-backed `_open`, `indexes`, `serve`, `connect` |
| `docs/launchd/local.minirag-mcp.plist` (new) | LaunchAgent template (plan 4's README and migration use it) |
| `mcpb/manifest.json`, `pyproject.toml`, `uv.lock`, `CLAUDE.md` | tool list, direct deps, architecture block |
| `tests/index_helpers.py`, `tests/http_helpers.py` (new) | `IndexesFile`, `serving(app)` |

---

### Task 1: `indexes.toml` loader

**Files:**
- Create: `src/minirag_mcp/indexes.py`, `tests/index_helpers.py`, `tests/test_indexes.py`
- Modify: `src/minirag_mcp/security.py` (loopback names), `tests/conftest.py` (fixture `indexes_file`)

**Interfaces:**
- Consumes: `config.load_config`, `ConfigError`, `Config` (with plan 1's `index_name`,
  `embed_endpoints`, `rerank_endpoint`, `classify_endpoint`, `document_weight`,
  `rerank_top`, default `db_path`; plan 2's `sync_workers`, `ocr_lang`);
  `endpoints.Endpoint`, `endpoints.read_api_key`, `EndpointError`.
- Produces:
  ```python
  # security.py
  LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")
  def host_name(value: str) -> str           # "[::1]:8765" -> "::1", "LOCALHOST:1" -> "localhost"
  def is_loopback_name(value: str) -> bool   # host_name(value) in LOOPBACK_NAMES, no ipaddress
  # indexes.py
  CONFIG_ENV = "MINIRAG_CONFIG"; DEFAULT_HOST = "127.0.0.1"; DEFAULT_PORT = 8765
  INDEX_KEYS: dict[str, str]                 # [index.NAME] key -> env name load_config reads
  SERVER_ENV_KEYS: frozenset[str]            # env names covered by [server] and the endpoint tables:
                                             # CACHE_DIR, ENDPOINT_API_KEY_FILE, RERANK_URL, RERANK_MODEL,
                                             # CLASSIFY_URL, CLASSIFY_MODEL, EMBED_URL, EMBED_ENDPOINT_MODEL
  @dataclass(frozen=True)
  class ServerConfig:                        # exactly the contract's fields
      host: str; port: int; token_file: Path
      indexes: dict[str, Config]; descriptions: dict[str, str]
  def default_config_path(env: Mapping[str, str] | None = None) -> Path
      # MINIRAG_CONFIG (~ expanded) if set and non-blank; else sys.platform == "win32":
      # platformdirs.user_config_path("minirag") / "indexes.toml"; else
      # Path(XDG_CONFIG_HOME or Path.home() / ".config") / "minirag" / "indexes.toml"
  def load_server_config(path: Path, env: Mapping[str, str]) -> ServerConfig
      # token_file defaults to path.parent / "token"
  ```
  `tests/index_helpers.py`: `IndexesFile(path, env, roots: dict[str, Path], text)` with
  `.write(text)`, and `make_indexes_file(tmp_path) -> IndexesFile`. The conftest fixture
  `indexes_file` returns it. It writes `<tmp>/config/indexes.toml` with no `[server]`
  table, and `env = {"HOME": str(tmp_path)}`. It defines three indexes, in this order:
  - `alpha`: description "Alpha documents", root `<tmp>/corpora/alpha` holding
    `alpha.md` = "# Alpha\n\nAlpha body about apricots and audits."
  - `beta`: description "Beta documents", `fts_language = "Russian"`, root `<tmp>/corpora/beta`
    holding `beta.md` about bananas.
  - `gamma`: description "Gamma documents",
    `model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`, root
    `<tmp>/corpora/gamma` holding `gamma.md` about grapes.

**The schema this task fixes.** Each kind of table, its keys and its errors:

- **Top level.** Only `server`, `embedding`, `rerank`, `classify` and `index` are allowed.
  Any other key is a `ConfigError`.
- **`[server]`.** Keys:
  - `host` (str)
  - `port` (int, 1..65535)
  - `token_file`, `api_key_file`, `cache_dir` (path strings)
  - `allow_remote` (bool)

  It is not required.
- **`[embedding."<fastembed id>"]`, `[rerank]`, `[classify]`.** Each takes exactly `url`
  and `name`, both required strings.
- **`[index.NAME]`.** `NAME` must `fullmatch` `[a-z0-9][a-z0-9-]*`. Keys:
  - `roots`: a non-empty list of non-empty strings
  - `description`: str, default `""`
  - plus `INDEX_KEYS`

  At least one index is required.

**Paths.** `~` and `~/...` expand against `env["HOME"]`, falling back to `Path.home()`.
After expansion a path must be absolute.

**Building each `Config`.**
1. Synthesize `index_env`:
   - every `INDEX_KEYS` value present, converted to str (a bool becomes "true"/"false";
     an int or float becomes `str(v)`; any other type is a `ConfigError` naming the key)
   - `CACHE_DIR` from `[server] cache_dir`
   - `ENDPOINT_API_KEY_FILE` from `[server] api_key_file` (which must exist, else a
     `ConfigError` naming the path)
   - `RERANK_URL`/`RERANK_MODEL` and `CLASSIFY_URL`/`CLASSIFY_MODEL` from their tables
   - never `EMBED_URL`/`EMBED_ENDPOINT_MODEL`: those name one endpoint for the configured
     model only, and plan 1's `embed_endpoints` is a mapping (step 3)
2. Call `load_config(index_env, base_dir_flags=expanded_roots, cwd=path.parent)`.
3. Apply `dataclasses.replace(cfg, index_name=NAME, embed_endpoints=EMBEDDINGS)`, where
   `EMBEDDINGS` is built once per file from every `[embedding."<id>"]` table:
   `{id: Endpoint(url, name, read_api_key(api_key_file))}`. Every index gets the whole
   mapping, so an index whose recorded model differs from its configured one still embeds
   through the router when that model has a table (plan 1's `make_embedder` picks the entry
   by model). A `url` not starting `http://` or `https://` is a `ConfigError` naming
   `[embedding."<id>"] url`, the rule `load_config` applies to `EMBED_URL`; an unreadable
   key file is a `ConfigError` naming its path.

The process's own environment never reaches an index from the file.

**Error messages.** A `ConfigError` from step 2 is re-raised as
`f"{path}: [index.{NAME}] {message}"`, with each whole-word env name in the message
replaced by its TOML spelling. For example `RAG_HYBRID_WEIGHT` becomes `hybrid_weight`,
`RERANK_URL` becomes `[rerank] url`, and `RERANK_MODEL` becomes `[rerank] name`.
Syntax errors become `f"{path}: {tomllib error}"`, which carries "(at line N, column M)".

`INDEX_KEYS` = `model`→`MODEL_NAME`, `db`→`DB_PATH`, `fts_language`→`RAG_FTS_LANGUAGE`,
`token_budget`→`CHUNK_TOKEN_BUDGET`, `hybrid_weight`→`RAG_HYBRID_WEIGHT`,
`document_weight`→`RAG_DOCUMENT_WEIGHT`, `rerank_top`→`RAG_RERANK_TOP`,
`grouping`→`RAG_GROUPING`, `max_distance`→`RAG_MAX_DISTANCE`, `max_files`→`RAG_MAX_FILES`,
`max_file_size`→`MAX_FILE_SIZE`, `allow_private_urls`→`ALLOW_PRIVATE_URLS`,
`ocr_lang`→`RAG_OCR_LANG`, `ocr_min_chars_per_page`→`RAG_OCR_MIN_CHARS_PER_PAGE`,
`sync_workers`→`RAG_SYNC_WORKERS`. With `SERVER_ENV_KEYS` this covers every name
`load_config` reads after plans 1 and 2; the coverage test below catches any later one,
which goes in under its env name, lowercased, with `RAG_` dropped.

- [ ] **Step 1: Write the failing tests** (`tests/test_indexes.py`)

```python
def test_each_index_table_becomes_a_config(indexes_file):
    sc = load_server_config(indexes_file.path, indexes_file.env)
    assert list(sc.indexes) == ["alpha", "beta", "gamma"]
    alpha = sc.indexes["alpha"]
    assert alpha.roots == (indexes_file.roots["alpha"].resolve(),)
    assert alpha.db_path == indexes_file.roots["alpha"].resolve() / ".minirag" / "v2"
    assert alpha.index_name == "alpha" and sc.descriptions["alpha"] == "Alpha documents"
    assert sc.indexes["beta"].fts_language == "Russian"
    assert (sc.host, sc.port) == ("127.0.0.1", 8765)
    assert sc.token_file == indexes_file.path.parent / "token"

def test_every_index_setting_maps_to_what_load_config_reads(tmp_path):
    # A dict subclass recording .get/__getitem__/__contains__ keys; load_config is run with
    # {} and with EMBED_URL/RERANK_URL/CLASSIFY_URL set (ConfigError swallowed) so
    # conditional reads are seen too.
    assert recorded - {"BASE_DIR", "BASE_DIRS"} == set(INDEX_KEYS.values()) | SERVER_ENV_KEYS

def test_index_settings_reach_the_config(...):  # hybrid_weight = 0.75, max_files = 3,
    ...                                          # allow_private_urls = true, grouping = "related"
def test_the_process_environment_does_not_reach_a_file_index(...):  # env has RAG_HYBRID_WEIGHT=0.9
    assert cfg.hybrid_weight == 0.6

def test_a_bad_value_names_the_index_and_its_key(...):   # [index.alpha] hybrid_weight = 1.5
    msg = str(excinfo.value)
    assert str(path) in msg and "[index.alpha]" in msg and "hybrid_weight" in msg
    assert "RAG_HYBRID_WEIGHT" not in msg

def test_a_syntax_error_names_the_line(...):              # "[server]\nport 8765\n"
    assert "line 2" in str(excinfo.value)

@pytest.mark.parametrize("text, key", [
    ('[index.alpha]\nroots = ["~/a"]\ntokn_budget = 200\n', "tokn_budget"),
    ('[server]\nprot = 1\n[index.a]\nroots = ["~/a"]\n', "prot"),
    ('[indexes.a]\nroots = ["~/a"]\n', "indexes"),
    ('[rerank]\nurl = "http://127.0.0.1:1/v1/rerank"\nname = "r"\nmodle = "x"\n'
     '[index.a]\nroots = ["~/a"]\n', "modle"),
])
def test_unknown_keys_are_refused(tmp_path, text, key): ...  # key in message

@pytest.mark.parametrize("name", ["Client", "a_b", "-a", "a b"])
def test_index_names_are_lowercase_letters_digits_and_dashes(tmp_path, name): ...

def test_two_indexes_sharing_an_index_directory_are_refused(tmp_path):
    # [index.a] and [index.b] both roots = ["~/docs", ...] with distinct second roots
    assert "[index.a]" in msg and "[index.b]" in msg and ".minirag/v2" in msg

@pytest.mark.parametrize("line", ['roots = ["docs"]', 'roots = ["~/d"]\ndb = "./idx"'])
def test_a_relative_path_is_refused(tmp_path, line): ...  # "absolute or start with ~"

def test_home_relative_paths_expand_against_the_given_home(tmp_path): ...

def test_endpoint_tables_reach_the_indexes_that_use_them(tmp_path):
    # api key file "secret\n"; [embedding."intfloat/multilingual-e5-large"]; [rerank]; [classify];
    # index e5 (default model) and index mini (MiniLM)
    assert e5.embed_endpoints == mini.embed_endpoints == {
        E5_LARGE: Endpoint(EMBED_URL, "multilingual-e5-large", api_key="secret")}
    assert make_embedder(e5, configured_scheme(e5)).backend == "endpoint"
    assert make_embedder(mini, configured_scheme(mini)).backend == "in-process"   # no table for MiniLM
    assert e5.rerank_endpoint == mini.rerank_endpoint == Endpoint(RERANK_URL, "bge", api_key="secret")

def test_an_embedding_url_that_is_not_http_is_refused(tmp_path): ...   # '[embedding."intfloat/multilingual-e5-large"] url' in msg

def test_an_endpoint_without_its_name_is_refused(tmp_path): ...    # "[rerank]" and "name" in msg
def test_a_missing_api_key_file_is_refused(tmp_path): ...          # the path in msg

@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com", "::ffff:127.0.0.1"])
def test_a_non_loopback_host_is_refused(tmp_path, host): ...       # "loopback" in msg
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_accepted(tmp_path, host): ...
def test_allow_remote_lifts_the_host_check(tmp_path): ...
def test_a_file_without_indexes_is_refused(tmp_path): ...
def test_a_missing_file_is_refused_naming_it(tmp_path): ...

@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_minirag_config_wins_on_every_platform(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_config_path({"MINIRAG_CONFIG": "~/x.toml"}) == tmp_path / "x.toml"

@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_macos_and_linux_use_dot_config_not_application_support(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_config_path({}) == tmp_path / ".config" / "minirag" / "indexes.toml"
    assert default_config_path({"XDG_CONFIG_HOME": str(tmp_path / "x")}) == tmp_path / "x" / "minirag" / "indexes.toml"

def test_windows_uses_platformdirs(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(indexes.platformdirs, "user_config_path", lambda name: tmp_path / "AppData" / name)
    assert default_config_path({"XDG_CONFIG_HOME": str(tmp_path / "x")}) == tmp_path / "AppData" / "minirag" / "indexes.toml"
```

On a macOS runner, `test_macos_and_linux_use_dot_config_not_application_support[darwin]`
fails against a platformdirs-based implementation (`~/Library/Application Support`). That
is the red this rule needs.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_indexes.py -q`
Expected: collection ERROR, `ModuleNotFoundError: minirag_mcp.indexes`

- [ ] **Step 3: Implement `security.LOOPBACK_NAMES`, `host_name`, `is_loopback_name`, and `indexes.py` as specified above.**

  Use `tomllib.load` on the binary file. Detect a shared directory by grouping on
  `config.db_path` after `load_config`. The `host` check uses `is_loopback_name` (no
  `ipaddress`; see security.py's note on IPv4-mapped addresses varying by patch release).

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_indexes.py -q` then `uv run pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/indexes.py src/minirag_mcp/security.py tests/index_helpers.py tests/conftest.py tests/test_indexes.py
git commit -m "feat(indexes): load indexes.toml into one Config per index"
```

---

### Task 2: Index registry, with both shells on a one-entry registry

**Files:**
- Create: `src/minirag_mcp/registry.py`, `tests/test_registry.py`
- Modify: `src/minirag_mcp/server.py` (the one `IndexHandle` plan 1 left becomes a one-entry registry, no visible change), `src/minirag_mcp/cli/__init__.py` (`_load` → `_open`), `tests/conftest.py` (fixture `index_states`)

**Interfaces:**
- Consumes: Task 1 (`ServerConfig`, `load_server_config`); plan 1 `handle.IndexHandle`,
  `handle.OpenIndex`, `handle.make_embedder`, `handle.EmbedderFactory`,
  `identity.IndexState`, `identity.EmbeddingScheme`, `embedder.UnknownModelError`.
- Produces:
  ```python
  SINGLE_INDEX_NAME = "default"
  CONNECTION_SCOPE_KEY = "minirag.connection"
  class Use(StrEnum): READ = "read"; INGEST = "ingest"; SYNC = "sync"   # delete_file is READ
  @dataclass(frozen=True)
  class Connection:
      default: str | None; allow: frozenset[str] | None   # None: every index reachable
      def reachable(self, name: str) -> bool             # allow is None or name == default or name in allow
  UNRESTRICTED = Connection(default=None, allow=None)
  class IndexRefused(Exception): ...                     # str() is the user-facing message
  class IndexUnavailable(IndexRefused): name: str; reason: str
  RegistryEmbedderFactory = Callable[[Config, EmbeddingScheme], object]
  class EmbedderPool:
      def get(self, config: Config, scheme: EmbeddingScheme) -> Embedder
          # make_embedder(config, scheme), memoized on (scheme, config.cache_dir, config.embed_endpoints.get(scheme.model))
  class IndexEntry:
      name: str; config: Config; description: str
      handle: IndexHandle                                # built with the entry; opens nothing until used
      def runtime(self) -> OpenIndex                     # handle.get(); any exception -> IndexUnavailable(name, reason), retried next call
      def state(self) -> tuple[IndexState, str]          # handle.state(); never raises
  class IndexRegistry:
      @classmethod
      def from_file(cls, path: Path, env: Mapping[str, str], *, embedder_factory: RegistryEmbedderFactory | None = None,
                    model_factory=None) -> "IndexRegistry"  # ConfigError if the first load fails
      @classmethod
      def single(cls, config: Config | None, *, error: str | None = None,
                 embedder_factory: RegistryEmbedderFactory | None = None, model_factory=None) -> "IndexRegistry"
      path: Path | None; server: ServerConfig | None; config_error: str | None
      @property
      def names(self) -> tuple[str, ...]                 # file order
      def reachable(self, connection: Connection) -> tuple[str, ...]
      def resolve(self, connection: Connection, requested: str | None) -> IndexEntry
      def open(self, connection: Connection, requested: str | None, use: Use) -> tuple[IndexEntry, OpenIndex]
  ```
  Each entry's handle is `IndexHandle(config, embedder_factory=partial(factory, config),
  model_factory=model_factory)`. A `None` factory means `self._pool.get`, so indexes of one
  model share one embedder (and one in-process model) in `serve`. Sharing is sound because
  the probe an embedder checks depends only on the scheme and the backend, which the memo
  key fixes.

**What moves and what goes.** Plan 1's `IndexHandle` already builds the embedder, store,
labels, pipeline, `SyncManager` and reranker, checks the identity and picks up a new
generation; the registry builds none of these and keeps no copy of an `OpenIndex`, so
every `open` goes through `handle.get()` and a swap made by the CLI is seen on the next
call. `IndexEntry.runtime()` turns any exception from `handle.get()` (plan 1's
`IndexUnavailableError`, `IndexOpenError`, `UnknownModelError`, an endpoint error) into
`IndexUnavailable` with `reason = str(e)`. When an exception is an `UnknownModelError`, its
reason gains " — see fastembed's supported models" in both shells. That suffix used to be
CLI-only.

**State** is `entry.handle.state()` (plan 1: open failure `UNAVAILABLE` with `str(error)`,
a running job `SYNCING`, else `index_state`). Plan 1's `SyncManager.running` is what
`SYNCING` reads; the registry adds no flag of its own.

**Refusals.** `open` refuses in this order; the copy is exact.
- **No index configured, with a config error:** `f"Configuration error: {config_error}"`
- **No `requested` and no default:** "This connection has no default index, so name one
  with \`index\`. Reachable indexes: {R}."
- **Name not configured:** "Unknown index {name!r}. Reachable indexes: {R}." When the name
  came from the connection's default rather than from `requested`, insert
  " (this connection's default)" right after `{name!r}`.
- **Not reachable:** `f"Index {name!r} is not allowed on this connection (allow={','.join(sorted(allow))}). Reachable indexes: {R}."`
- **Unavailable:** `f"Index {name!r} is unavailable: {reason.rstrip('.')}."` + O (plan 1's
  messages, such as the 1.x refusal naming the v2 path, end in a full stop already)
- **`Use.INGEST` while `REBUILD_PENDING`:** `f"Index {name!r} needs a rebuild ({reason}); ingests wait for it, since the rebuilt index replaces this one. Rebuild it with an unscoped sync_start (CLI: sync)."` + O

`R` is the reachable names in file order, joined by `", "`, or `none`. `O` is
`f" Other reachable indexes: {others}."` when any reachable index other than `name`
exists, and `""` otherwise.

**Locks.** `self._lock` guards only the entries dict and its swap. Opening an index
(seconds, if a model loads) must never hold it, because Task 7 calls `reload()` on the
event-loop thread.

**Shells.** `server.create_app` keeps its current signature in this task. It builds
`IndexRegistry.single(config, error=config_error, embedder_factory=(lambda c, s: embedder)
if embedder is not None else None, model_factory=model_factory)`. Each tool calls
`registry.open(Connection(SINGLE_INDEX_NAME, None), None, use)` where it called the handle,
uses the `OpenIndex` it returns and `entry.handle` (`.sync`, `.notes()`) as plan 1 did, and
`status` keeps plan 1's `status_payload(config, entry.handle)`. `cli._load` becomes
`_open(use, base_dir, db_path, cache_dir, model_name) -> tuple[IndexEntry, OpenIndex]`
over `IndexRegistry.single(...)` with `embedder_factory=_cli_embedder` (late-bound, so the
tests' patch of `_make_embedder` applies): `_cli_embedder(c, s)` returns `_make_embedder(c)`
when that is not `None` (plan 1: patched in tests), else `make_embedder(c, s)`. CLI `sync`
calls `entry.handle.run_sync(scope, on_event)` (plan 1). `IndexRefused` and `ConfigError`
go to `_fail`.

- [ ] **Step 1: Write the failing tests** (`tests/test_registry.py`; fixture `index_states`
  in conftest patches `minirag_mcp.handle.index_state`, the name plan 1's
  `IndexHandle.state()` calls, to read a dict keyed by `config.index_name or "default"`,
  default `(IndexState.READY, "")`)

```python
# in tests/index_helpers.py (Tasks 3, 4, 6, 7, 8 import it from there)
def file_registry(indexes_file, embedder_factory, **kw):
    return IndexRegistry.from_file(indexes_file.path, indexes_file.env, embedder_factory=embedder_factory, **kw)

def test_a_registry_opens_no_index_until_one_is_used(indexes_file, fake_embedder):
    made = []
    reg = file_registry(indexes_file, lambda c, s: made.append(c.index_name) or fake_embedder)
    assert reg.names == ("alpha", "beta", "gamma") and made == []
    reg.open(Connection("alpha", None), None, Use.READ)
    assert made == ["alpha"]

def test_indexes_with_the_same_model_share_one_embedder(indexes_file, monkeypatch):
    built = []
    monkeypatch.setattr(registry, "make_embedder",
                        lambda config, scheme, **kw: built.append(scheme.model) or FakeEmbedder(scheme.model))
    reg = IndexRegistry.from_file(indexes_file.path, indexes_file.env)   # default factory: the pool
    a, b, g = (reg.resolve(UNRESTRICTED, n).runtime().embedder for n in ("alpha", "beta", "gamma"))
    assert a is b and a is not g and built == [E5_LARGE, MINILM]

def test_concurrent_first_use_opens_an_index_once(indexes_file, fake_embedder):
    # factory sleeps 0.2 s and records; 8 threads call resolve(...).runtime()
    assert calls == ["alpha"] and len({id(r) for r in runtimes}) == 1

@pytest.mark.parametrize("conn, requested, message", [
    (Connection("alpha", None), "delta", "Unknown index 'delta'. Reachable indexes: alpha, beta, gamma."),
    (Connection("delta", None), None, "Unknown index 'delta' (this connection's default). Reachable indexes: alpha, beta, gamma."),
    (Connection("alpha", frozenset({"beta"})), "gamma", "Index 'gamma' is not allowed on this connection (allow=beta). Reachable indexes: alpha, beta."),
    (Connection("alpha", frozenset()), "beta", "Index 'beta' is not allowed on this connection (allow=). Reachable indexes: alpha."),
    (UNRESTRICTED, None, "This connection has no default index, so name one with `index`. Reachable indexes: alpha, beta, gamma."),
])
def test_refusals_name_the_index_and_the_reachable_ones(indexes_file, fake_embedder, conn, requested, message):
    with pytest.raises(IndexRefused) as e:
        file_registry(indexes_file, lambda c, s: fake_embedder).open(conn, requested, Use.READ)
    assert str(e.value) == message

def test_a_failed_open_is_unavailable_and_retried(indexes_file, fake_embedder):
    # factory raises RuntimeError("probe mismatch") while fail["on"]
    with pytest.raises(IndexUnavailable) as e:
        reg.open(Connection("alpha", None), None, Use.READ)
    assert str(e.value) == "Index 'alpha' is unavailable: probe mismatch. Other reachable indexes: beta, gamma."
    assert reg.resolve(UNRESTRICTED, "alpha").state() == (IndexState.UNAVAILABLE, "probe mismatch")
    fail["on"] = False
    reg.open(Connection("alpha", None), None, Use.READ)   # no exception

def test_a_rebuild_pending_index_answers_reads_and_syncs_but_refuses_ingests(indexes_file, fake_embedder, index_states):
    index_states["beta"] = (IndexState.REBUILD_PENDING, "chunk budget 300 -> 400 tokens")
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    reg.open(UNRESTRICTED, "beta", Use.READ); reg.open(UNRESTRICTED, "beta", Use.SYNC)
    with pytest.raises(IndexRefused, match=re.escape("Index 'beta' needs a rebuild (chunk budget 300 -> 400 tokens)")):
        reg.open(UNRESTRICTED, "beta", Use.INGEST)

def test_a_running_sync_makes_the_index_syncing(indexes_file):
    # factory returns a FakeEmbedder whose embed_documents waits on an Event;
    # entry.handle.sync.start(None) ingests alpha.md and blocks inside the embedding
    assert entry.state() == (IndexState.SYNCING, ""); release; entry.handle.sync.wait()
    assert entry.state() == (IndexState.READY, "")

def test_every_open_goes_through_the_handle(indexes_file, fake_embedder, monkeypatch):
    # spy on IndexHandle.get: three reg.open(..., "alpha", Use.READ) calls -> three get() calls, one OpenIndex

def test_the_environment_mode_is_one_index_named_default(tmp_path, fake_embedder):
    reg = IndexRegistry.single(load_config({"BASE_DIR": str(tmp_path)}), embedder_factory=lambda c, s: fake_embedder)
    assert reg.names == ("default",)
    broken = IndexRegistry.single(None, error="BASE_DIRS must be a JSON array")
    assert broken.names == ()
    with pytest.raises(IndexRefused, match="Configuration error: BASE_DIRS"):
        broken.open(Connection("default", None), None, Use.READ)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_registry.py -q`
Expected: collection ERROR, `ModuleNotFoundError: minirag_mcp.registry`

- [ ] **Step 3: Implement `registry.py`, then switch `server.create_app` and the CLI's loading onto `IndexRegistry.single` as described.**

- [ ] **Step 4: Run the new tests and the whole suite**

Run: `uv run pytest tests/test_registry.py -q && uv run pytest -q`
Expected: PASS. The existing server, CLI and parity tests pass unchanged. That is the
evidence the move changed nothing visible. If one asserts a whole message that is now
prefixed "Index 'default' is unavailable:", stop and report it. Do not edit it.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/registry.py src/minirag_mcp/server.py src/minirag_mcp/cli/__init__.py tests/conftest.py tests/test_registry.py tests/index_helpers.py
git commit -m "feat(registry): lazy index registry; both shells run on a one-entry registry"
```

---

### Task 3: Reload on a changed file or on request

**Files:**
- Modify: `src/minirag_mcp/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: Task 2 `IndexRegistry`, `IndexEntry`; Task 1 `load_server_config`.
- Produces:
  ```python
  IndexRegistry.refresh(self) -> None   # reloads iff sha256 of the file's bytes changed (an unreadable file counts as changed); never raises
  IndexRegistry.reload(self) -> None    # always re-reads; never raises; the SIGHUP handler (Task 7)
  ```
  Both are no-ops on a `single` registry.

**Reload behaviour.**
- A name whose `Config` and description are unchanged keeps its `IndexEntry`, and with it
  its `IndexHandle` and the index that handle has open.
- A changed or added name gets a new entry, opened lazily.
- A removed name is dropped. Its handle is not touched, so a running sync finishes, and
  its data stays on disk.
- On success, `config_error = None` and `server` is replaced. One line goes to stderr:
  `f"minirag-mcp: reloaded {path}: {', '.join(names)}"`. When `host`, `port` or
  `token_file` changed, it appends `"; host/port/token_file changes apply after a restart"`.
- On `ConfigError` or `OSError`, the entries are kept and `config_error = str(e)`. One line
  goes to stderr: `f"minirag-mcp: could not reload {path}, still serving {', '.join(names)}: {e}"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_an_edited_file_is_reloaded_on_the_next_refresh(indexes_file, fake_embedder):
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    indexes_file.write(indexes_file.text + '\n[index.delta]\nroots = ["~/corpora/alpha/sub"]\n')
    reg.refresh()
    assert reg.names == ("alpha", "beta", "gamma", "delta")

def test_an_unchanged_file_is_not_parsed_again(indexes_file, fake_embedder, monkeypatch):
    # wrap registry.load_server_config with a counter; three refresh() calls
    assert loads == 1

def test_an_unchanged_index_keeps_its_open_runtime(...):     # edit beta's description only
    assert reg.resolve(UNRESTRICTED, "alpha").runtime() is before

def test_a_changed_index_is_reopened_with_its_new_settings(...):   # alpha hybrid_weight = 0.9
    after = reg.resolve(UNRESTRICTED, "alpha").runtime()
    assert after is not before and after.config.hybrid_weight == 0.9

def test_a_removed_index_stops_answering_and_keeps_its_data(...):
    # ingest beta.md through reg.open(UNRESTRICTED, "beta", Use.INGEST)[1].pipeline; drop [index.beta]; refresh
    with pytest.raises(IndexRefused, match="Unknown index 'beta'"):
        reg.open(UNRESTRICTED, "beta", Use.READ)
    assert (indexes_file.roots["beta"] / ".minirag" / "v2").is_dir()

def test_a_broken_edit_keeps_the_previous_indexes_and_says_why(indexes_file, fake_embedder, capsys):
    reg = file_registry(indexes_file, lambda c, s: fake_embedder)
    indexes_file.write("[server]\nport 8765\n"); reg.refresh()
    assert reg.names == ("alpha", "beta", "gamma") and "line 2" in reg.config_error
    assert "still serving alpha, beta, gamma" in capsys.readouterr().err
    indexes_file.write(indexes_file.text); reg.refresh()
    assert reg.config_error is None

def test_a_deleted_file_keeps_the_previous_indexes(...):   # unlink; refresh; names unchanged; path in config_error
def test_reload_reads_the_file_even_when_it_did_not_change(...):   # counter +1, "reloaded" on stderr
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_registry.py -q -k "reload or refresh or edit or removed or deleted or unchanged or changed"`
Expected: FAIL with `AttributeError: 'IndexRegistry' object has no attribute 'refresh'`

- [ ] **Step 3: Implement `refresh` and `reload` as specified.**

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_registry.py -q && uv run pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/registry.py tests/test_registry.py
git commit -m "feat(registry): reload indexes.toml on change or request, keeping it when broken"
```

---

### Task 4: `index` on every tool, `list_indexes`, results that name their index

**Files:**
- Modify: `src/minirag_mcp/server.py`, `src/minirag_mcp/results.py`, `src/minirag_mcp/cli/__init__.py` (wrap every JSON payload; `status` via `registry_status`), `mcpb/manifest.json`
- Test: create `tests/test_server_indexes.py`; modify `tests/test_server.py` (`test_all_twelve_tools_listed` → `test_all_thirteen_tools_listed` plus `list_indexes`; `await_sync(client, job_id, index=None)`), `tests/test_parity.py` (`test_an_ingest_without_labels_keeps_its_reply_shape`: the set gains `"index"`)

**Interfaces:**
- Consumes: Tasks 2 and 3 (`IndexRegistry.open/resolve/refresh/reachable`, `Connection`,
  `Use`, `CONNECTION_SCOPE_KEY`, `IndexEntry.state/handle/runtime`, `IndexUnavailable.reason`);
  plan 1's `results.status_payload(config, handle)`, `results.rerank_status`,
  `OpenIndex.identity/embedder/reranker/labels/store`, `LabelerSource.backend`,
  `identity.read_identity`, `IdentityError`.
- Produces:
  ```python
  # server.py
  def create_app(source: Config | IndexRegistry | None, *, config_error: str | None = None,
                 embedder: object | None = None, env: Mapping[str, str] | None = None,
                 model_factory: Callable | None = None, connection: Connection | None = None) -> FastMCP
  def current_connection(fallback: Connection) -> Connection
  # results.py
  def with_index(payload: dict, entry: IndexEntry) -> dict
  def index_status(entry: IndexEntry) -> dict
  def registry_status(registry: IndexRegistry, connection: Connection, requested: str | None) -> dict
  def list_indexes_payload(registry: IndexRegistry, connection: Connection) -> dict
  ```

**`create_app`.**
- A `Config`/`None` source wraps itself in `IndexRegistry.single(...)`, and its fallback
  connection is `Connection("default", None)`.
- An `IndexRegistry` source must come with `config_error`, `embedder` and `model_factory`
  all `None` (else `ValueError`). Its fallback is `connection or UNRESTRICTED`.
- `current_connection` returns `get_http_request().scope.get(CONNECTION_SCOPE_KEY,
  Connection(None, frozenset()))` (fail closed). On `RuntimeError` (no HTTP request) it
  returns `fallback`.

**Every tool.**
- Each of the 12 tools gains a last parameter
  `index: Annotated[str | None, Field(description=INDEX_ARG_DOC)] = None`.
- Its body starts with `registry.refresh()`, then
  `entry, rt = registry.open(current_connection(fallback), index, USE)`, and ends with
  `return with_index(payload, entry)`.
- `USE` is `INGEST` for the three `ingest_*` tools, `SYNC` for `sync_start`, and `READ`
  for the rest.
- `sync_start`/`sync_status` use `entry.handle.sync` (plan 1's manager, whose `run` rebuilds
  a pending index on an unscoped sync). `sync_status` turns the manager's `KeyError` into
  `f"{e.args[0]} in index {entry.name!r}; a job started in another index is polled with
  that `index`."`.
- `status` returns `registry_status(...)`.

`INDEX_ARG_DOC` = "Which index to use. Omit it for this connection's default; name another
only when the user names it or asks to look elsewhere. list_indexes lists them, and every
result's `index` says which one answered."

**`with_index`** returns `{"index": entry.name, **payload}`, plus
`"indexState": state.value` when the state is not ready. When the state is
`REBUILD_PENDING` it also adds `"indexWarning": f"Index {name!r} differs from its
configuration ({reason}); it answers from its existing vectors until an unscoped
sync_start rebuilds it."` (plan 1: a scoped sync does not rebuild).

**`index_status`** is `{"index": entry.name, **status_payload(entry.config,
entry.handle)}`. Plan 1's payload already carries `indexState` (always), `rebuildReasons`
when pending, and `indexError` (the same `str(error)` `IndexEntry.state()` reports) when
the open fails; this plan adds no state keys of its own to `status`.

**`registry_status`:**
- No entries plus a `config_error`: return `{"version": __version__, "configError": ...}`.
- Otherwise `resolve` (an `IndexRefused` propagates), then `index_status(entry)`.
- Add `configError` when the last reload failed.

**`list_indexes_payload`** returns `{"default": connection.default, "indexes": [row, ...]}`
in file order, plus `configError` when one is set.
- A reachable row is `{name, description, model, language, reachable: True, state,
  documentCount, chunkCount, embeddingBackend, reranker, classifyBackend}`, plus
  `stateReason` when not ready. `state`/`stateReason` come from `entry.state()`.
  - `model` is the recorded model (`read_identity(config.db_path).embedding.model`; `None`
    or `IdentityError` -> the configured `config.model_name`). Reading `index.json` is not
    an open, so an unreachable row uses it too.
  - `language` is `config.fts_language`.
  - `embeddingBackend` is `getattr(rt.embedder, "backend", "in-process")`, `reranker` is
    `rerank_status(rt.reranker)["reranker"]` and `classifyBackend` is `rt.labels.backend`
    (spec section 2: `list_indexes` says which backend is live for all three).
  - Counts come from `rt.store.source_count()` and `chunk_count()`.
  - An unavailable row has no counts and no backends.
- An unreachable row is exactly `{name, description, model, language, reachable: False}`.
  It is never opened.

**The `list_indexes` tool** (13th) takes no arguments. Its docstring:

> List the indexes this server hosts and which of them this connection may use.
>
> Each row has name, description, model, language and reachable. A reachable row also has
> documentCount, chunkCount, state ('ready', 'syncing', 'rebuild pending' or
> 'unavailable'), stateReason when not ready, and embeddingBackend, reranker and
> classifyBackend: which backend computes embeddings ('endpoint', 'in-process', or
> 'in-process (endpoint unreachable)'), reranks ('endpoint', 'unreachable' or 'off') and
> classifies. `default` names this connection's default index, or is null when its URL named
> none. Pass a reachable name as `index` to another tool only when the user asks for that
> index or asks to look elsewhere. Reporting an index opens it, which may load its
> embedding model.

**Also in this task:**
- In the `sync_start` docstring, add: "The reply's `index` is where the job runs; pass it
  to sync_status when it is not the default."
- `mcpb/manifest.json` gains `{"name": "list_indexes", "description": "List the indexes
  this server hosts and which this connection may use"}`.
- Every CLI command's JSON payload goes through `with_index` (CLI `sync --json`, plan 1's
  `{"counts", "errors", "rebuild"}`, included). `cli.status` uses `registry_status`.

- [ ] **Step 1: Write the failing tests** (`tests/test_server_indexes.py`; fixture
  `make(default="alpha", allow=None)` returns `create_app(reg, connection=Connection(...))`
  over `file_registry(indexes_file, lambda c, s: fake_embedder)`, using `index_states`)

```python
async def test_every_tool_takes_an_optional_index(make):
    async with Client(make()) as c:
        tools = {t.name: t for t in await c.list_tools()}
    for name, t in tools.items():
        if name == "list_indexes": continue
        assert "index" in t.inputSchema["properties"] and "index" not in t.inputSchema.get("required", [])
        assert "list_indexes" in t.inputSchema["properties"]["index"]["description"]

async def test_a_call_without_index_uses_the_connection_default(make, indexes_file): ...
    # ingest alpha.md without index -> r["index"] == "alpha"; query "apricots" -> index "alpha",
    # every source under roots["alpha"]
async def test_a_named_index_takes_reads_and_writes(make, indexes_file): ...
    # ingest_file(beta.md, index="beta") -> "beta"; query(index="beta") finds it;
    # query "bananas" on the default finds no beta source
async def test_a_file_outside_the_default_indexs_roots_is_refused(make, indexes_file): ...  # "outside"
async def test_refusals_reach_the_client_naming_the_reachable_indexes(make):
    # Client(make(default="alpha", allow=frozenset({"beta"})))
    r = await c.call_tool("query_documents", {"query": "x", "index": "gamma"}, raise_on_error=False)
    assert r.is_error and r.content[0].text.endswith(
        "Index 'gamma' is not allowed on this connection (allow=beta). Reachable indexes: alpha, beta.")
async def test_a_connection_without_a_default_must_name_an_index(make): ...
    # query and status without index -> "no default index"; status(index="alpha")["index"] == "alpha"
async def test_list_indexes_reports_each_index_and_what_this_connection_may_use(make, indexes_file):
    # make(default="alpha", allow=frozenset()); ingest alpha.md first
    assert got["default"] == "alpha"
    alpha, beta, _ = got["indexes"]
    assert alpha["reachable"] and alpha["state"] == "ready" and alpha["documentCount"] == 1
    assert (alpha["embeddingBackend"], alpha["reranker"], alpha["classifyBackend"]) == ("in-process", "off", "in-process")
    assert beta == {"name": "beta", "description": "Beta documents", "model": beta["model"],
                    "language": "Russian", "reachable": False}
async def test_a_rebuild_pending_index_answers_with_a_warning_and_refuses_ingests(make, index_states): ...
    # beta pending: query(index="beta") has indexState "rebuild pending" and the reason in indexWarning;
    # ingest_data(index="beta") is an error containing "needs a rebuild"; list row stateReason set
async def test_an_unavailable_index_is_refused_listed_and_diagnosable(...):
    # embedder_factory (lambda c, s) raises RuntimeError("probe mismatch") for gamma: query -> error
    # "Index 'gamma' is unavailable: probe mismatch."; list row state "unavailable";
    # status(index="gamma") returns indexError "probe mismatch", indexState "unavailable"
async def test_sync_status_is_polled_in_the_index_the_job_runs_in(make): ...
    # job = sync_start(index="beta") -> job["index"] == "beta"; await_sync(c, id, index="beta") succeeded;
    # sync_status(jobId) without index -> error containing "in index 'alpha'"
async def test_an_edited_indexes_toml_reaches_the_client_without_a_restart(make, indexes_file): ...
async def test_a_broken_edit_is_reported_and_the_old_indexes_keep_answering(make, indexes_file): ...
    # write "[server]\nport 8765\n": list_indexes configError has "line 2"; query on alpha still works
```

In `tests/test_server.py`:

```python
async def test_every_result_names_its_index(app, monkeypatch, public_dns):
    # call each of the 12 tools once (ingest_url with parse_url patched as in test_ingest_data_and_url)
    assert all(payload["index"] == "default" for payload in payloads)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_server_indexes.py tests/test_server.py -q`
Expected: FAIL. `create_app()` rejects the `connection` keyword, and the tool lists lack
`index` and `list_indexes`.

- [ ] **Step 3: Implement `create_app`, `current_connection`, the four results functions, `list_indexes`, the CLI wrapping and the manifest entry, as specified above.**

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_server_indexes.py tests/test_server.py tests/test_mcpb.py tests/test_parity.py -q && uv run pytest -q`
Expected: PASS. Payload-equality parity tests stay green because both shells add
`"index": "default"`.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/server.py src/minirag_mcp/results.py src/minirag_mcp/cli/__init__.py mcpb/manifest.json tests/
git commit -m "feat(server): index argument on every tool, list_indexes, results name their index"
```

---

### Task 5: The index rule in the server instructions

> **Carried in (2026-11-19):** the instructions also get one line for struck-through text
> (plan 2 Task 1, superseded form): `~~...~~` marks deleted text; never cite it as current.
> Keep the total under the pinned budget.

**Files:**
- Modify: `src/minirag_mcp/instructions.py`
- Test: `tests/test_instructions.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SERVER_INSTRUCTIONS` carrying an `<indexes>` block.

**Exact copy.** Insert this block between `</search_first>` and `<do_not_search>`,
separated by blank lines like the others:

```
<indexes>
Use the default index. Pass `index` only when the user names another index or asks to \
look elsewhere, and say which index an answer came from.
</indexes>
```

To stay within 1700, remove two rationale clauses:
- " A search that cannot change the answer costs latency." from `<do_not_search>`
- " — a ranking unit, not a passage meant to stand alone" from `<reading_results>`

Measured: 1669 characters. Add one paragraph to the module docstring. It says index
descriptions travel through `list_indexes`, not here, because of the 2048 cap, and which
two clauses paid for the block.

- [ ] **Step 1: Write the failing test**

```python
async def test_the_index_rule_reaches_the_client(app):
    async with Client(app()) as c:
        received = c.initialize_result.instructions
    block = _flat(received.split("<indexes>", 1)[1].split("</indexes>", 1)[0])
    assert "default index" in block and "`index`" in block
    assert "which index an answer came from" in block
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_instructions.py -q`
Expected: FAIL with `IndexError` (no `<indexes>`). Then add only the block, and confirm
`test_the_builtin_text_leaves_room_for_a_user_append` goes red at 1776 > 1700.

- [ ] **Step 3: Remove the two clauses.**

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_instructions.py -q && uv run pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/instructions.py tests/test_instructions.py
git commit -m "feat(instructions): use the default index and say which index answered"
```

---

### Task 6: CLI `--index`, `--config` and `indexes`

**Files:**
- Modify: `src/minirag_mcp/cli/__init__.py`, `src/minirag_mcp/__main__.py` (`app.meta()` instead of `app()`)
- Test: create `tests/test_cli_index.py`; modify `tests/test_parity.py`

**Interfaces:**
- Consumes: Tasks 1–4 (`default_config_path`, `IndexRegistry.from_file`, `UNRESTRICTED`,
  `registry_status`, `list_indexes_payload`, `with_index`, Task 2's `_cli_embedder`).
- Produces:
  ```python
  @dataclass(frozen=True)
  class Selection: index: str | None = None; config: str | None = None
  _SELECTION: ContextVar[Selection]                  # set and reset around each meta call
  @app.meta.default
  def main(*tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
           index: Annotated[str | None, Parameter(name="--index", help=...)] = None,
           config: Annotated[str | None, Parameter(name="--config", help=...)] = None) -> object
  def _config_path() -> Path        # --config, else default_config_path(os.environ)
  def _file_registry() -> IndexRegistry   # from_file(_config_path(), os.environ, embedder_factory=_cli_embedder, model_factory=_make_model_factory())
  @app.command
  def indexes(*, json: JsonFlag = False) -> None
  ```

**How `--index` selects.**
- With `--index NAME`, `_open` calls `_file_registry().open(UNRESTRICTED, NAME, use)`;
  `sync` then runs `entry.handle.run_sync(...)` as without it.
  `status` calls `registry_status(registry, UNRESTRICTED, NAME)`.
- `--index` given together with any of `--base-dir`, `--db-path`, `--cache-dir` or
  `--model-name` fails: `_fail("--index reads indexes.toml; --base-dir, --db-path,
  --cache-dir and --model-name configure the environment-variable index and cannot be
  combined with it. Set them in the index's [index.NAME] table.")`.
- Without `--index`, behaviour is unchanged (`IndexRegistry.single`).

**`indexes`** always reads the file and prints `list_indexes_payload(registry,
UNRESTRICTED)`. Its human output is one line per row:
`f"{name} [{state}] {documentCount} documents, {chunkCount} chunks, {model}, {language} — {description}"`.
An unreachable row never occurs here. An unavailable row prints `stateReason` in place of
the counts.

- [ ] **Step 1: Write the failing tests** (autouse fixture patches `cli._make_embedder`
  to the fake. `run_meta(tokens)` = `cli.app.meta(tokens, result_action="return_value",
  exit_on_error=False)`, catching `SystemExit` like `run_cli`)

```python
def test_index_selects_a_named_index_from_indexes_toml(indexes_file, capsys):
    cfg = ["--config", str(indexes_file.path)]
    assert run_meta([*cfg, "--index", "beta", "sync", "--json"])[0] == 0
    code, out, _ = run_meta([*cfg, "--index", "beta", "query", "bananas", "--json"])
    payload = json.loads(out)
    assert payload["index"] == "beta"
    assert all(s["source"].startswith(str(indexes_file.roots["beta"])) for s in payload["sources"])

def test_index_may_follow_the_command(...): ...       # ["query", "bananas", "--index", "beta", "--config", p, "--json"]
def test_minirag_config_selects_the_file(...): ...    # monkeypatch.setenv("MINIRAG_CONFIG", p); status --json -> index "beta"
def test_an_unknown_index_is_refused_naming_the_configured_ones(...):
    assert code == 1 and "Unknown index 'delta'. Reachable indexes: alpha, beta, gamma." in err
def test_index_cannot_be_combined_with_the_environment_index_flags(...):
    assert code == 1 and "--db-path" in err and "indexes.toml" in err
def test_without_index_the_cli_reads_the_environment_as_before(...): ...   # payload["index"] == "default"
def test_indexes_lists_every_index_in_the_file(indexes_file, capsys): ...  # names, reachable all True, default None; 3 human lines
def test_a_missing_indexes_toml_is_a_clean_error(tmp_path, capsys): ...    # --index a list: exit 1, path in err
```

In `tests/test_parity.py`:

```python
async def test_a_named_index_answers_the_same_through_both(indexes_file, capsys, monkeypatch, fake_embedder, index_states):
    # CLI --index beta sync; MCP create_app(file_registry(...), connection=Connection("alpha", None))
    # query_documents {"query": "bananas", "index": "beta"}; CLI --index beta query bananas --json
    assert json.loads(out) == mcp_payload and mcp_payload["index"] == "beta"
async def test_an_unknown_index_is_refused_the_same_way_by_both(...):
    assert is_error and code == 1 and text.endswith(_last_error(err))
async def test_list_indexes_matches_the_indexes_command(...):
    # MCP over UNRESTRICTED, after the same CLI sync of alpha
    assert json.loads(out) == mcp_payload
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_cli_index.py tests/test_parity.py -q`
Expected: FAIL. The meta app does not know `--index`, and there is no `indexes` command.

- [ ] **Step 3: Implement the meta app, `_config_path`, `_file_registry`, the `--index` branch of `_open`/`status`, and `indexes`; switch `__main__` to `app.meta()`.**

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_cli_index.py tests/test_parity.py tests/test_cli.py -q && uv run pytest -q && uv run minirag-mcp status --base-dir .`
Expected: PASS, and the smoke command prints a status with `index: default`.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/cli/__init__.py src/minirag_mcp/__main__.py tests/test_cli_index.py tests/test_parity.py
git commit -m "feat(cli): --index and --config read indexes.toml; indexes command"
```

---

### Task 7: `serve` over streamable HTTP, guarded

**Files:**
- Create: `src/minirag_mcp/http_server.py`, `tests/http_helpers.py`, `tests/test_http_server.py`
- Modify: `src/minirag_mcp/cli/__init__.py` (`serve`), `pyproject.toml` + `uv.lock` (direct deps)

**Interfaces:**
- Consumes: Task 1 (`LOOPBACK_NAMES`, `host_name`, `ServerConfig`), Tasks 2–4 (`Connection`,
  `CONNECTION_SCOPE_KEY`, `IndexRegistry.from_file/reload/server`, `create_app`).
- Produces:
  ```python
  def read_token(path: Path) -> str          # ConfigError: group/other mode bits set ("chmod 600 {path}"), or empty
  def ensure_token(path: Path) -> str        # creates parent (0o700) and file via O_CREAT|O_EXCL, 0o600, secrets.token_urlsafe(32)+"\n"; then read_token
  def parse_connection(path: str, query_string: str) -> Connection | None   # None -> 404
  class LoopbackGuard:                       # ASGI; __init__(self, app, *, token: str, allowed_hosts: Collection[str])
  class IndexPathMiddleware:                 # ASGI; __init__(self, app)
  def build_http_app(mcp: FastMCP, *, token: str, allowed_hosts: Collection[str]) -> Starlette
  def serve(config_path: Path, env: Mapping[str, str]) -> None
  ```

**`parse_connection`.**
- Path `"/mcp"` or `"/mcp/"` gives no default.
- `"/mcp/<one segment>"` gives that segment as the default, unvalidated, so a bad name
  gets the registry's loud refusal.
- Anything else is `None`.
- `allow` is the union of every `allow=` value, split on `,`, stripped, empties dropped.
  It is `None` when absent. A present-but-empty `allow=` is `frozenset()`.

**`LoopbackGuard`** handles `http` scopes only, and everything else passes through. Checks,
in order:
1. `host_name(Host)` is not in `allowed_hosts`: 421, body "Misdirected Request: Host must be a loopback name".
2. An `Origin` header is present and `urlsplit(origin).hostname` is not in `allowed_hosts`
   (this includes `"null"`): 403, "Forbidden: Origin is not a loopback origin".
3. `Authorization` is not `Bearer <token>` (scheme case-insensitive; compared with
   `hmac.compare_digest` on bytes): 401, with header `WWW-Authenticate: Bearer` and the
   body "Unauthorized: send Authorization: Bearer <the token in the server's token file>".

**`IndexPathMiddleware`** answers 404 with the body "Not Found: MCP is served at /mcp and
/mcp/<index>" when `parse_connection` gives `None`. Otherwise it forwards
`dict(scope, path="/mcp", raw_path=b"/mcp")` with `CONNECTION_SCOPE_KEY` set.

**`build_http_app`** = `mcp.http_app(path="/mcp", transport="http", middleware=[
Middleware(LoopbackGuard, token=..., allowed_hosts=...), Middleware(IndexPathMiddleware)])`.

**`serve`:**
1. Build `registry = IndexRegistry.from_file(config_path, env)`, then
   `s = registry.server`, `token = ensure_token(s.token_file)`.
2. Build the app with `allowed_hosts = {*LOOPBACK_NAMES, host_name(s.host)}`.
3. Print one stderr line:
   `f"minirag-mcp serve: http://{host}:{port}/mcp/<index> for {', '.join(names)}; token in {s.token_file}"`.
4. Run `asyncio.run(...)` of a coroutine that calls
   `loop.add_signal_handler(signal.SIGHUP, registry.reload)` and then awaits
   `uvicorn.Server(uvicorn.Config(app, host=s.host, port=s.port, lifespan="on",
   access_log=False, timeout_graceful_shutdown=2)).serve()`.

**The CLI `serve`** command reads `_config_path()`. A missing file fails with
`_fail(f"no indexes.toml at {path}; create one (docs/launchd/local.minirag-mcp.plist shows how to
run it)")`, and a `ConfigError` goes to `_fail`.

**`pyproject.toml`** gains `"uvicorn>=0.31.1"` and `"starlette>=0.27"`, with the project's
usual comment: fastmcp already pulls these in, but `serve` runs uvicorn itself and its
guards are Starlette middleware; the floors are fastmcp 3.4's own, so declaring them can
never move the resolution. Then run `uv lock`.

**`tests/http_helpers.py`:**
- `serving(app)` is an async context manager. It binds `socket` `("127.0.0.1", 0)`, runs
  `uvicorn.Server(...).serve(sockets=[sock])` as a task, waits for `server.started`, and
  yields `http://127.0.0.1:<port>`. On exit it sets `should_exit` and awaits the task. A
  `ClosedResourceError` traceback logged at shutdown is noise.
- `INITIALIZE` is a JSON-RPC initialize body.
- `MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_token_file_is_created_once_with_mode_0600(tmp_path):
    p = tmp_path / "cfg" / "token"
    t = ensure_token(p)
    assert len(t) >= 32 and stat.S_IMODE(p.stat().st_mode) == 0o600 and ensure_token(p) == t
def test_a_token_file_other_users_can_read_is_refused(tmp_path): ...   # chmod 0o644 -> "chmod 600"
def test_an_empty_token_file_is_refused(tmp_path): ...

@pytest.mark.parametrize("path, qs, expected", [
    ("/mcp", "", Connection(None, None)), ("/mcp/", "", Connection(None, None)),
    ("/mcp/alpha", "", Connection("alpha", None)),
    ("/mcp/alpha", "allow=beta,%20gamma", Connection("alpha", frozenset({"beta", "gamma"}))),
    ("/mcp/alpha", "allow=beta&allow=gamma", Connection("alpha", frozenset({"beta", "gamma"}))),
    ("/mcp/alpha", "allow=", Connection("alpha", frozenset())),
    ("/mcp/alpha/beta", "", None), ("/api", "", None),
])
def test_parse_connection(path, qs, expected): assert parse_connection(path, qs) == expected

# async fixture `site` in tests/conftest.py (Task 8 reuses it); TOKEN and AUTH in http_helpers:
# registry = file_registry(indexes_file, lambda c, s: fake_embedder);
# app = build_http_app(create_app(registry), token=TOKEN, allowed_hosts=LOOPBACK_NAMES);
# async with serving(app) as base: yield base
async def test_a_request_without_the_token_is_refused(site):
    r = await http.post(f"{site}/mcp/alpha", json=INITIALIZE, headers=MCP_HEADERS)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
async def test_a_wrong_token_is_refused(site): ...                       # 401
async def test_a_foreign_origin_is_refused_even_with_the_token(site): ...  # Origin http://evil.example -> 403
async def test_an_opaque_origin_is_refused(site): ...                    # Origin null -> 403
async def test_a_host_that_is_not_loopback_is_refused(site): ...         # Host evil.example:<port> -> 421
async def test_a_loopback_origin_with_the_token_is_served(site): ...     # Origin http://localhost:5173 -> 200
async def test_an_unknown_path_is_not_found(site): ...                   # POST /api with token -> 404
async def test_the_url_names_the_default_and_the_allow_list(site):
    url = f"{site}/mcp/beta?allow=alpha"
    async with Client(StreamableHttpTransport(url, headers=AUTH)) as c:
        listed = (await c.call_tool("list_indexes", {})).data
        assert listed["default"] == "beta"
        assert {r["name"]: r["reachable"] for r in listed["indexes"]} == {"alpha": True, "beta": True, "gamma": False}
        assert (await c.call_tool("status", {})).data["index"] == "beta"
        r = await c.call_tool("status", {"index": "gamma"}, raise_on_error=False)
        assert r.is_error and "not allowed" in r.content[0].text
async def test_mcp_without_an_index_has_no_default(site): ...            # status at /mcp -> "no default index"

def test_serve_requires_the_token_and_reloads_on_sighup(indexes_file, tmp_path):
    # free port: bind 0, read, close; write it into [server] port; start
    # [sys.executable, "-m", "minirag_mcp", "--config", p, "serve"] with stderr to tmp_path/"serve.log";
    # wait up to 30 s for the port
    assert stat.S_IMODE((indexes_file.path.parent / "token").stat().st_mode) == 0o600
    assert httpx.post(f"{base}/mcp", json=INITIALIZE, headers=MCP_HEADERS).status_code == 401
    # with the token: query_documents without index at /mcp -> error listing "alpha, beta, gamma"
    proc.send_signal(signal.SIGHUP)
    # the same call succeeds again (process alive), then proc.poll() is None
    # append [index.delta]; next call's error lists "alpha, beta, gamma, delta"
    # terminate; "reloaded" in serve.log
def test_serve_refuses_a_non_loopback_host(indexes_file, capsys): ...   # host = "0.0.0.0" -> exit 1, "loopback"
def test_serve_without_indexes_toml_says_where_it_looked(tmp_path, capsys): ...
```

The subprocess test never opens an index: a call refused for "no default" is answered
before any model is needed.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_http_server.py -q`
Expected: collection ERROR, `ModuleNotFoundError: minirag_mcp.http_server`

- [ ] **Step 3: Implement `http_server.py`, the `serve` command and the dependency lines; run `uv lock`.**

- [ ] **Step 4: Run them to verify they pass, then prove the guards are what makes them pass**

Run: `uv run pytest tests/test_http_server.py -q`
Expected: PASS.

Then build the app once without `LoopbackGuard` in `build_http_app` (local edit, not
committed). Run `uv run pytest tests/test_http_server.py -q -k "refused"` and expect
every security test FAIL. Restore the guard. Next, remove only the Origin branch and
confirm both Origin tests go red. Restore, then run `uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/http_server.py src/minirag_mcp/cli/__init__.py pyproject.toml uv.lock tests/http_helpers.py tests/test_http_server.py
git commit -m "feat(serve): streamable HTTP for every index, token, Host and Origin guard, SIGHUP reload"
```

---

### Task 8: `connect` stdio-to-HTTP bridge

**Files:**
- Create: `src/minirag_mcp/bridge.py`, `tests/test_bridge.py`
- Modify: `src/minirag_mcp/cli/__init__.py` (`connect`)

**Interfaces:**
- Consumes: Task 1 (`load_server_config`, `ServerConfig`), Task 7 (`read_token`,
  `serving`), `instructions.build_instructions`.
- Produces:
  ```python
  CONNECT_WAIT_SECONDS = 10.0
  def bridge_url(server: ServerConfig, index: str, allow: Sequence[str] | None) -> str
  def wait_for_server(server: ServerConfig, timeout: float) -> None   # polls socket.create_connection every 0.2 s; ConnectionError naming the URL
  def build_bridge(server: ServerConfig, index: str, allow: Sequence[str] | None,
                   env: Mapping[str, str]) -> FastMCP
  def run_bridge(server: ServerConfig, index: str, allow: Sequence[str] | None,
                 env: Mapping[str, str]) -> None           # wait_for_server, then build_bridge(...).run(show_banner=False)
  ```

**`bridge_url`** is `f"http://{host}:{port}/mcp/{index}"`, with an IPv6 host in brackets,
plus `"?allow=" + ",".join(allow)` when `allow` is not `None`. `allow == []` yields
`?allow=`, which restricts the connection to the default.

**`build_bridge`** = `create_proxy(Client(StreamableHttpTransport(bridge_url(...),
headers={"Authorization": f"Bearer {read_token(server.token_file)}"})),
name="minirag-mcp", instructions=build_instructions(env))`.

**CLI `connect(index: str, *, allow: str | None = None)`:**
- It reads `_config_path()`.
- It refuses, with `_fail`:
  - an `index` or `allow` name not configured: `f"Unknown index {name!r} in {path}. Configured indexes: {...}."`
  - a missing token file: `f"no token at {token_file}; start \`minirag-mcp serve\` once to create it"`
  - `wait_for_server` failing: `f"minirag-mcp serve is not running at {url} ({cause}); start it, e.g. launchctl kickstart gui/$UID/local.minirag-mcp"`
- `allow` is split on `,` and stripped. A given-but-empty value means `[]`.

Nothing may print to stdout. stdout carries the protocol.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("host, allow, url", [
    ("127.0.0.1", None, "http://127.0.0.1:8765/mcp/beta"),
    ("127.0.0.1", ["alpha", "gamma"], "http://127.0.0.1:8765/mcp/beta?allow=alpha,gamma"),
    ("127.0.0.1", [], "http://127.0.0.1:8765/mcp/beta?allow="),
    ("::1", None, "http://[::1]:8765/mcp/beta"),
])
def test_bridge_url(host, allow, url): ...

async def test_the_bridge_carries_the_token_the_default_and_the_allow_list(site, indexes_file):
    # ServerConfig from indexes_file with port = the served port, token file holding TOKEN
    async with Client(build_bridge(server, "beta", ["alpha"], {})) as c:
        assert c.initialize_result.instructions == SERVER_INSTRUCTIONS
        listed = (await c.call_tool("list_indexes", {})).data
        assert listed["default"] == "beta"
        r = await c.call_tool("status", {"index": "gamma"}, raise_on_error=False)
        assert r.is_error and "not allowed on this connection" in r.content[0].text
async def test_the_bridge_without_the_right_token_cannot_reach_the_server(site, ...): ...  # wrong token file -> list_tools raises

def test_connect_refuses_an_index_the_file_does_not_configure(indexes_file, capsys): ...  # exit 1, "Configured indexes: alpha, beta, gamma"
def test_connect_without_a_token_file_says_to_start_serve(indexes_file, capsys): ...
def test_connect_fails_loudly_when_the_server_is_not_running(indexes_file, capsys, monkeypatch):
    monkeypatch.setattr(bridge, "CONNECT_WAIT_SECONDS", 0.3)   # port of a closed socket, token present
    assert code == 1 and "http://127.0.0.1:" in err and "not running" in err and out == ""
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_bridge.py -q`
Expected: collection ERROR, `ModuleNotFoundError: minirag_mcp.bridge`

- [ ] **Step 3: Implement `bridge.py` and the `connect` command.**

- [ ] **Step 4: Run them to verify they pass**

Run: `uv run pytest tests/test_bridge.py -q && uv run pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/bridge.py src/minirag_mcp/cli/__init__.py tests/test_bridge.py
git commit -m "feat(connect): stdio bridge to the running server for Claude Desktop"
```

---

### Task 9: LaunchAgent template and architecture notes

**Files:**
- Create: `docs/launchd/local.minirag-mcp.plist`, `tests/test_launchd.py`
- Modify: `CLAUDE.md` (architecture block)

**Interfaces:**
- Consumes: the `serve` command (Task 7).
- Produces: the one LaunchAgent template, with these placeholders: `__UVX__` (absolute
  path to `uvx`), `__VERSION__` and `__HOME__`. Plan 4's README section and the owner's
  migration copy this file; they add no template of their own.

**The plist.**
- `Label` = `local.minirag-mcp`.
- `ProgramArguments` = `[__UVX__, "--from", "minirag-mcp==__VERSION__", "minirag-mcp", "serve"]`.
- `RunAtLoad` is true; `KeepAlive` is `{SuccessfulExit: false}` (restarted after a crash,
  not after a clean stop).
- `StandardOutPath` and `StandardErrorPath` are `__HOME__/Library/Logs/minirag-mcp.log`.

**Its header XML comment** says:
- replace the placeholders; launchd expands neither `~` nor `$HOME`, so every path must be
  absolute
- copy the file to `~/Library/LaunchAgents/`
- `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.minirag-mcp.plist`
- reload `indexes.toml`: `launchctl kill HUP gui/$(id -u)/local.minirag-mcp`
- stop: `launchctl bootout gui/$(id -u)/local.minirag-mcp`
- the token is in `~/.config/minirag/token`

An XML comment must not contain `--`, so the comment cannot quote `--from`. The test's
parse catches that. The README's `launchctl` section is plan 4's.

**`CLAUDE.md`'s architecture block** becomes "server.py FastMCP app, 13 tools" and
"cli/ 14 subcommands". It gains one line each, for this plan's modules and the ones plans 1
and 2 added (neither of them edits `CLAUDE.md`):
- `indexes.py`: indexes.toml -> one Config per index
- `registry.py`: index registry over IndexHandles, connection rules, refusals, reload
- `http_server.py`: serve: token, Host/Origin guard, /mcp/<index>
- `bridge.py`: connect: stdio proxy to serve
- `handle.py`: opens one index by its recorded scheme; the one place embedders are built
- `identity.py`: index.json, 1.x detection, index state, atomic swap
- `rebuild.py`: re-embed or re-ingest into <db>.next, then swap
- `endpoints.py`: llama.cpp embeddings and rerank clients
- `parallel.py`: sync's worker processes and batched writes

The "Retrieval is hybrid" paragraph gains one sentence: fusion adds a whole-document BM25
bonus (`RAG_DOCUMENT_WEIGHT`) and, when an endpoint is configured, a cross-encoder reranks
the top `RAG_RERANK_TOP` fused chunks.

- [ ] **Step 1: Write the failing test**

```python
def test_the_launch_agent_template_runs_serve():
    raw = (ROOT / "docs" / "launchd" / "local.minirag-mcp.plist").read_text()
    plist = plistlib.loads(raw.replace("__UVX__", "/opt/uvx").replace("__VERSION__", "2.0.0a1")
                              .replace("__HOME__", "/Users/u").encode())
    assert plist["Label"] == "local.minirag-mcp" and plist["RunAtLoad"]
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["ProgramArguments"][-2:] == ["minirag-mcp", "serve"] and "serve" in cli.app
    assert "launchctl kill HUP" in raw and "launchctl bootstrap" in raw
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_launchd.py -q`
Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Write the plist and update `CLAUDE.md`.**

- [ ] **Step 4: Run everything**

Run: `uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests && uv build`
Expected: all green, and the sdist and wheel are built.

- [ ] **Step 5: Commit**

```bash
git add docs/launchd/local.minirag-mcp.plist tests/test_launchd.py CLAUDE.md
git commit -m "docs(serve): LaunchAgent template; architecture notes for the index server"
```
