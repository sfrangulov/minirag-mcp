# minirag v2 core (plan 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give one index its v2 core: model endpoints with a loud in-process fallback, e5 prefixes and per-model chunk windows, an `index.json` identity that rebuilds instead of mixing vectors, whole-document BM25 as a third fusion signal, an optional cross-encoder reranker, a served classify model, and a measured e5 chunk budget.

**Architecture:** All behaviour lands in the core (`endpoints`, `embedder`, `identity`, `store`, `handle`, `rebuild`, `labels/endpoint_model`). `server.py` and the CLI ask an `IndexHandle` for the open index and shape every answer through shared helpers in `results.py`. An index is always searched by the scheme recorded in its `index.json`; a configuration that differs leaves it `rebuild pending` until an explicit unscoped sync rebuilds it into `<db>.next` and swaps it in.

**Tech Stack:** Python 3.11+, LanceDB (vectors + BM25 FTS), fastembed + tokenizers, requests, huggingface_hub, FastMCP, cyclopts, pytest.

**Spec:** `docs/superpowers/specs/2026-11-19-minirag-v2-design.md` (sections 1 single-index mode, 2, 3, 4, 6). Shared names: `docs/superpowers/plans/2026-11-19-minirag-v2-00-contract.md`.

## Global Constraints

- Python >= 3.11; everything through `uv`; CI matrix 3.11-3.14 on Linux and macOS.
- English in code, docs and commits. Tests must fail on unfixed code: run every new test against the code before the change and see it go red (CLAUDE.md traps).
- Every task leaves `uv run pytest -q`, `uv run ruff check src tests` and `uv run ruff format --check src tests` green.
- Behaviour lives in the core; MCP and CLI expose it identically. A user-visible change gets a two-shell test in `tests/test_parity.py`'s style.
- Tests use the fakes in `tests/conftest.py`; endpoint tests use the `fake_llama` server from Task 1 (stdlib `http.server` in a thread), never the network. Nothing outside `-m slow` downloads a model.
- Warnings go through `warnings.warn`, never `print`: stdout is the MCP stdio channel.
- Comments state constraints the code cannot show; never narrate, never argue the change.
- `pyproject.toml` holds the only version number; this plan does not change it. README/docs are plan 4's; this plan edits only `evals/enterpriserag/README.md`.
- Private evaluation material (the Russian ruler, OCR and classify spikes) stays outside the repository; only aggregate numbers go into bd notes.
- Exact values: `DEFAULT_MODEL = "intfloat/multilingual-e5-large"`; windows e5 512, MiniLM 128; `RAG_HYBRID_WEIGHT` 0.6; `RAG_DOCUMENT_WEIGHT` 0.5 in [0, 1], 0 disables; `RAG_RERANK_TOP` 30 in [1, 200]; RRF k = 60; top 100 documents; probe cosine >= 0.999; classify `n_probs` 100; default `db_path` `<first root>/.minirag/v2`.
- Dependencies: `huggingface-hub>=0.20` moves from the `[classify]` extra into core dependencies (fastembed already requires it); nothing else is added. Endpoint HTTP sessions set `trust_env = False`.
- Index state per tool result (`indexState`/`indexWarning`) is plan 3's (`with_index`); plan 1 reports index state in `status` and `identity.index_state` only. Embedding-backend and rerank flags in results are plan 1's.
- Commits: conventional style, message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Text longer than the model window sent to the embedding endpoint** (a pasted paragraph as a query; a code fence the packer keeps whole, up to `FENCE_BUDGET_MULTIPLE` x budget). Expected: cut locally to the window as fastembed truncates in-process, so both backends embed the same tokens; llama.cpp never answers "input too large" and the endpoint is not marked unreachable. Test: Task 3 `test_text_over_the_window_is_cut_before_the_endpoint_sees_it`.
2. **`HTTP_PROXY` / `HTTPS_PROXY` set in the user's environment.** Expected: endpoint calls go straight to the configured host. Test: Task 1 `test_proxy_variables_are_ignored`.
3. **A process killed between the two renames of a swap.** Expected: the next open finishes the swap; it never creates a fresh empty index and answers "nothing indexed". Tests: Task 4 `test_recover_swap_finishes_an_interrupted_swap`, Task 11 `test_an_interrupted_swap_is_finished_on_open`.
4. **An endpoint serving a model of another width** (a MiniLM preset under an e5 index). Expected: a loud error naming both widths and the router model; nothing written, no fallback. Test: Task 3 `test_an_endpoint_of_another_width_is_refused_loudly`.
5. **Something else answering 200 on the endpoint port** (an HTML page, another service on :8080). Expected: `EndpointError` naming the URL and that the reply was not JSON, then the loud fallback. Test: Task 1 `test_a_reply_that_is_not_json_is_an_endpoint_error`.

## Decisions the contract left to plan 1

- Unknown model: window 128 and default budget 110 (today's values). A window set too high truncates silently; too low only costs chunk size.
- `Config.embed_endpoints: Mapping[str, Endpoint]` keyed by fastembed model id replaces the contract's single `embed_endpoint`, so an index whose recorded model differs from the configured one still embeds through the router when that model has an entry. Env mode yields one entry, keyed by the configured model.
- `documents` rows also carry `ingested_at`, the stamp of the chunks they were built from, so staleness is detected the way look-alike signatures detect it.
- A fresh v2 folder gets `index.json` at its first open, with an empty probe; the first write records the probe. So every folder v2 creates is told apart from a 1.x index.
- `reranked` (bool) and `embeddingBackend` are always in a query result; `warnings` (list) only when non-empty. `score` stays the first-stage fused score; a reranked hit adds `rerankScore`.
- A scoped sync on a `rebuild pending` index is refused with `IndexBusyError`-style loud text ("index needs a rebuild: run an unscoped sync"); only an unscoped sync writes to a pending index, and it rebuilds. (Coordinator decision: no writes that would mix two chunking schemes in one index; `delete_file` stays allowed.)
- While a rebuild runs, writes to the old index (any process) are refused with `IndexBusyError` instead of being lost at the swap.
- The sync lock stays at `<db>/.sync.lock`; after the swap the rebuilding sync has nothing left to protect.
- The classify backend is not part of the taxonomy hash (180/180 identical decisions measured).

## File structure

| File | Responsibility |
|---|---|
| `src/minirag_mcp/endpoints.py` (new) | HTTP clients for llama.cpp routes: embeddings, rerank, a JSON request helper |
| `src/minirag_mcp/embedder.py` | model tables, prefixes, endpoint + fallback, window cut, local token counting, probe, pickling |
| `src/minirag_mcp/config.py` | v2 defaults and environment variables |
| `src/minirag_mcp/identity.py` (new) | `index.json`, legacy detection, scheme diff, `index_state`, staging/previous dirs, swap, `RebuildInfo` |
| `src/minirag_mcp/handle.py` (new) | `make_embedder`; opens one index: identity, recorded-scheme embedder, store, labels, pipeline, sync, reranker; state; generation pickup |
| `src/minirag_mcp/rebuild.py` (new) | re-embed or re-ingest into `<db>.next`, then swap |
| `src/minirag_mcp/store.py` | `documents` table, document bonus, reranking, `copy_reembedded`, write guard |
| `src/minirag_mcp/labels/endpoint_model.py` (new) | classify choice model over llama-server |
| `src/minirag_mcp/results.py` | query flags, `status_payload` |
| `sync.py`, `lock.py`, `ingest/pipeline.py`, `labels/labeler.py`, `server.py`, `cli/__init__.py` | wiring |
| `evals/enterpriserag/{common,bulk_index,recall}.py` | the English stand on the production path |

---

### Task 1: Endpoint clients

**Files:**
- Create: `src/minirag_mcp/endpoints.py`
- Modify: `tests/conftest.py` (generalize `FakeEmbedder`; add `fake_llama`, `dead_url`)
- Test: `tests/test_endpoints.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (contract names plus `request_json`, `EMBED_BATCH`, `.endpoint`, `.last_error`):
```python
class EndpointError(Exception): ...            # message names the URL and the cause
@dataclass(frozen=True)
class Endpoint:
    url: str; name: str; api_key: str | None = None; timeout: float = 120.0
EMBED_BATCH = 32
def read_api_key(path: Path | None) -> str | None     # first non-empty line stripped; EndpointError if unreadable or blank
def request_json(endpoint: Endpoint, url: str, body: dict | None = None) -> dict
    # POST json (GET when body is None); adds "model": endpoint.name to a POST body; Bearer header
    # when api_key; EndpointError on connection error, timeout, non-2xx (status + first 300 chars), non-JSON
class EmbeddingEndpoint:
    endpoint: Endpoint
    def __init__(self, endpoint: Endpoint) -> None
    def embed(self, texts: Sequence[str]) -> list[list[float]]    # unit vectors, input order
class RerankEndpoint:
    endpoint: Endpoint
    last_error: str | None                                         # set by a failed call, cleared by a success
    def __init__(self, endpoint: Endpoint) -> None
    def scores(self, query: str, passages: Sequence[str]) -> list[float]   # raw logits, input order
```
- Test fixtures later tasks use: `FakeEmbedder(model_name="fake", dim=8, salt="")`; `fake_llama` -> `FakeLlama`; `dead_url` -> `"http://127.0.0.1:<port nobody listens on>"`.

- [ ] **Step 1: Generalize `FakeEmbedder` and add the fixtures**

`FakeEmbedder.__init__(self, model_name="fake", dim=8, salt="")`. `_vec` hashes `salt + text` exactly as today for the first 8 components (so `FakeEmbedder()` vectors stay byte-identical) and `f"{salt}{text}#{block}"` for each further block of 8; unit length as today.

`FakeLlama`, served by a `ThreadingHTTPServer` on 127.0.0.1:0 in a daemon thread, shut down after the test (the `redirect_server` pattern):
```
url(path="") -> str;  requests: list[Seen]  # Seen(method, path, query, headers, body)
fail: dict[str, tuple[int, str]]            # route -> (status, text body)
html: set[str]                              # routes answering 200 text/html
dim = 8; vector_salt = ""; drop_last = False
top_logprobs: list[dict] = []; props: dict = {}; token_count: int | None = None; split_letters = False
POST /v1/embeddings  -> {"data": [{"index": i, "embedding": 3 * FakeEmbedder(dim=dim, salt=vector_salt)._vec(t)}]}
                        in reverse index order, the last entry dropped when drop_last
POST /v1/rerank      -> {"results": [{"index": i, "relevance_score": overlap(query, doc) - 2.0}]} reversed;
                        overlap = number of query words present in the document
POST /tokenize       -> {"tokens": ...}: a lone letter A..P -> [32 + i] ([32 + i, 1] when split_letters);
                        else token_count ids when set; else one id 1000 + i per whitespace word
POST /completion     -> {"completion_probabilities": [{"top_logprobs": top_logprobs}]}
GET  /props          -> props
```

- [ ] **Step 2: Write the failing tests** (`tests/test_endpoints.py`)

```python
def ep(fake_llama, path="/v1/embeddings", key=None):
    return Endpoint(fake_llama.url(path), "e5", key)

def test_embed_returns_unit_vectors_in_input_order(fake_llama):
    got = EmbeddingEndpoint(ep(fake_llama)).embed(["a", "b", "c"])
    assert got == [pytest.approx(FakeEmbedder()._vec(t)) for t in "abc"]

def test_embed_batches_and_names_the_model(fake_llama):
    EmbeddingEndpoint(ep(fake_llama)).embed([f"t{i}" for i in range(70)])
    assert [len(r.body["input"]) for r in fake_llama.requests] == [32, 32, 6]
    assert {r.body["model"] for r in fake_llama.requests} == {"e5"}

def test_a_bearer_header_is_sent_only_with_a_key(fake_llama): ...
    # key "k" -> headers["Authorization"] == "Bearer k"; no key -> no Authorization header

def test_no_texts_make_no_request(fake_llama):
    assert EmbeddingEndpoint(ep(fake_llama)).embed([]) == [] and fake_llama.requests == []

def test_an_unreachable_endpoint_names_the_url(dead_url):
    with pytest.raises(EndpointError, match=re.escape(dead_url)):
        EmbeddingEndpoint(Endpoint(dead_url + "/v1/embeddings", "e5")).embed(["a"])

def test_an_http_error_carries_status_and_body(fake_llama):
    fake_llama.fail["/v1/embeddings"] = (500, "boom")
    with pytest.raises(EndpointError, match=r"500.*boom"): EmbeddingEndpoint(ep(fake_llama)).embed(["a"])

def test_a_reply_that_is_not_json_is_an_endpoint_error(fake_llama):      # Review Focus 5
    fake_llama.html.add("/v1/embeddings")
    with pytest.raises(EndpointError, match="JSON"): EmbeddingEndpoint(ep(fake_llama)).embed(["a"])

def test_a_short_reply_is_an_endpoint_error(fake_llama):
    fake_llama.drop_last = True
    with pytest.raises(EndpointError): EmbeddingEndpoint(ep(fake_llama)).embed(["a", "b"])

def test_proxy_variables_are_ignored(fake_llama, monkeypatch):            # Review Focus 2
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False); monkeypatch.delenv("no_proxy", raising=False)
    assert len(EmbeddingEndpoint(ep(fake_llama)).embed(["a"])) == 1

def test_rerank_scores_come_back_in_input_order_unnormalized(fake_llama):
    rr = RerankEndpoint(ep(fake_llama, "/v1/rerank"))
    assert rr.scores("x y", ["x y z", "q", "x"]) == [0.0, -2.0, -1.0]
    assert fake_llama.requests[0].body["top_n"] == 3 and rr.last_error is None

def test_a_rerank_failure_is_remembered_until_a_success(fake_llama):
    fake_llama.fail["/v1/rerank"] = (503, "loading")
    rr = RerankEndpoint(ep(fake_llama, "/v1/rerank"))
    with pytest.raises(EndpointError): rr.scores("q", ["a"])
    assert "503" in rr.last_error
    fake_llama.fail.clear(); rr.scores("q", ["a"])
    assert rr.last_error is None

def test_read_api_key(tmp_path):
    (tmp_path / "k").write_text("\n  secret \nother\n")
    assert read_api_key(tmp_path / "k") == "secret" and read_api_key(None) is None
    (tmp_path / "blank").write_text(" \n")
    for path in (tmp_path / "missing", tmp_path / "blank"):
        with pytest.raises(EndpointError, match=re.escape(str(path))): read_api_key(path)
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/test_endpoints.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'minirag_mcp.endpoints'`.

- [ ] **Step 4: Implement `src/minirag_mcp/endpoints.py`**

One `requests.Session` per client, created lazily, with `trust_env = False`. Embeddings: `{"model", "input": batch}` per `EMBED_BATCH`, `data` sorted by `index`, count checked, each vector unit-normalized with numpy (a zero vector stays as is). Rerank: `{"model", "query", "documents", "top_n": len(passages)}`, `results` mapped by `index`; a missing index is an `EndpointError`; no passages, no request.

- [ ] **Step 5: Run the tests and the gate**

Run: `uv run pytest tests/test_endpoints.py -q && uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests`
Expected: all pass.

- [ ] **Step 6: Commit** — `feat(endpoints): clients for llama.cpp embeddings and rerank routes`

---

### Task 2: Model tables and the v2 configuration

**Files:**
- Modify: `src/minirag_mcp/embedder.py` (module-level tables only), `src/minirag_mcp/config.py`
- Test: `tests/test_config.py`, `tests/test_embedder.py` (pin `MINILM` where a test means MiniLM: `test_dim_from_registry_without_download`, `_truncating_embedder`, `test_chunks_fit_the_encoder_at_the_documented_maximum_budget`), and the default-path users `tests/test_server.py:148,162`, `tests/test_cli.py:208,255,297,555` (`.minirag/lancedb` -> `.minirag/v2`)

**Interfaces:**
- Consumes: `Endpoint`, `read_api_key`, `EndpointError` (Task 1).
- Produces:
```python
# embedder.py
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_LARGE = "intfloat/multilingual-e5-large"
MODEL_WINDOWS: dict[str, int] = {E5_LARGE: 512, MINILM: 128}
DEFAULT_BUDGETS: dict[str, int] = {E5_LARGE: 110, MINILM: 110}   # E5_LARGE is set by Task 13's measurement
def model_window(model_name: str) -> int     # unknown -> MODEL_MAX_TOKENS (128)
def default_budget(model_name: str) -> int   # unknown -> DEFAULT_TOKEN_BUDGET (110)
def prefix_rule(model_name: str) -> str      # "e5" when the lowercased id starts "intfloat/" and its name contains "e5"; else "none"
# config.py
DEFAULT_MODEL = E5_LARGE
DEFAULT_DOCUMENT_WEIGHT = 0.5; DEFAULT_RERANK_TOP = 30; MAX_RERANK_TOP = 200
def default_db_path(roots: Sequence[Path]) -> Path         # roots[0] / ".minirag" / "v2"
@dataclass(frozen=True) class Config:   # existing fields unchanged, then, all defaulted:
    document_weight: float = 0.5; rerank_top: int = 30
    embed_endpoints: Mapping[str, Endpoint] = field(default_factory=dict)   # fastembed model id -> endpoint
    rerank_endpoint: Endpoint | None = None; classify_endpoint: Endpoint | None = None
    index_name: str | None = None
```
Plan 3's loader fills `embed_endpoints` from `[embedding."<id>"]`, `index_name` and the other endpoints per index. In env mode `EMBED_URL` + `EMBED_ENDPOINT_MODEL` give one entry keyed by the resolved `model_name` (flag > env > default).

- [ ] **Step 1: Write the failing tests**

```python
def test_defaults_from_cwd(tmp_path):                       # replaces the 1.x version
    cfg = load_config({}, cwd=tmp_path)
    assert cfg.db_path == tmp_path.resolve() / ".minirag" / "v2"
    assert cfg.model_name == "intfloat/multilingual-e5-large"
    assert cfg.token_budget == default_budget(E5_LARGE)
    assert (cfg.hybrid_weight, cfg.document_weight, cfg.rerank_top) == (0.6, 0.5, 30)
    assert cfg.embed_endpoints == {} and cfg.rerank_endpoint is cfg.classify_endpoint is cfg.index_name is None

# test_invalid_numeric_env: drop {"CHUNK_TOKEN_BUDGET": "129"}, add
#   {"CHUNK_TOKEN_BUDGET": "513"}, {"RAG_DOCUMENT_WEIGHT": "1.5"}, {"RAG_DOCUMENT_WEIGHT": "-0.1"},
#   {"RAG_DOCUMENT_WEIGHT": "x"}, {"RAG_RERANK_TOP": "0"}, {"RAG_RERANK_TOP": "201"},
#   {"EMBED_URL": "http://127.0.0.1:8080/v1/embeddings"}, {"EMBED_ENDPOINT_MODEL": "e5"},
#   {"RERANK_URL": "http://127.0.0.1:8080/v1/rerank"}, {"CLASSIFY_MODEL": "plumb-4b"},
#   {"EMBED_URL": "ftp://h/v1/embeddings", "EMBED_ENDPOINT_MODEL": "e5"},
#   {"ENDPOINT_API_KEY_FILE": "/nonexistent/key"}

def test_the_budget_ceiling_is_the_models_window(tmp_path):
    assert load_config({"CHUNK_TOKEN_BUDGET": "512"}, cwd=tmp_path).token_budget == 512
    with pytest.raises(ConfigError, match="512"): load_config({"CHUNK_TOKEN_BUDGET": "513"}, cwd=tmp_path)
    with pytest.raises(ConfigError, match="128"):
        load_config({"MODEL_NAME": MINILM, "CHUNK_TOKEN_BUDGET": "129"}, cwd=tmp_path)
    assert load_config({"MODEL_NAME": "some/unknown"}, cwd=tmp_path).token_budget == 110
    with pytest.raises(ConfigError):
        load_config({"MODEL_NAME": "some/unknown", "CHUNK_TOKEN_BUDGET": "129"}, cwd=tmp_path)

def test_endpoints_come_from_the_environment(tmp_path):
    (tmp_path / "key").write_text("\n  secret  \n")
    env = {"EMBED_URL": "http://127.0.0.1:8080/v1/embeddings", "EMBED_ENDPOINT_MODEL": "multilingual-e5-large",
           "RERANK_URL": "http://127.0.0.1:8080/v1/rerank", "RERANK_MODEL": "bge-reranker-v2-m3",
           "CLASSIFY_URL": "http://127.0.0.1:8080", "CLASSIFY_MODEL": "plumb-4b",
           "ENDPOINT_API_KEY_FILE": str(tmp_path / "key")}
    cfg = load_config(env, cwd=tmp_path)
    assert cfg.embed_endpoints == {E5_LARGE: Endpoint("http://127.0.0.1:8080/v1/embeddings", "multilingual-e5-large", "secret")}
    assert cfg.rerank_endpoint == Endpoint("http://127.0.0.1:8080/v1/rerank", "bge-reranker-v2-m3", "secret")
    assert cfg.classify_endpoint == Endpoint("http://127.0.0.1:8080", "plumb-4b", "secret")
    assert set(load_config({**env, "MODEL_NAME": MINILM}, cwd=tmp_path).embed_endpoints) == {MINILM}

def test_retrieval_tuning_env(tmp_path):
    cfg = load_config({"RAG_DOCUMENT_WEIGHT": "0", "RAG_RERANK_TOP": "50"}, cwd=tmp_path)
    assert (cfg.document_weight, cfg.rerank_top) == (0.0, 50)

def test_prefix_rule_marks_the_e5_family():
    assert prefix_rule(E5_LARGE) == prefix_rule("intfloat/multilingual-e5-small") == "e5"
    assert prefix_rule(MINILM) == prefix_rule("BAAI/bge-base-en") == "none"

def test_model_window_and_default_budget():
    assert (model_window(E5_LARGE), model_window(MINILM), model_window("x/y")) == (512, 128, 128)
    assert default_budget("x/y") == 110
```
The base-dir tests expect `.minirag/v2`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_config.py tests/test_embedder.py -q`
Expected: FAIL (`ImportError: cannot import name 'E5_LARGE'`, then path and model asserts).

- [ ] **Step 3: Implement**

`config.py` imports the tables from `embedder.py`. The model is resolved first, then `CHUNK_TOKEN_BUDGET` with `default_budget(model)` and upper bound `model_window(model)`; the comment above the budget keeps its point (an inclusive bound is sound only because counting is untruncated), now about "the model's window". One `_float(env, key, default, lo, hi)` serves `RAG_HYBRID_WEIGHT` and `RAG_DOCUMENT_WEIGHT`. `_endpoint(env, url_key, name_key, api_key) -> Endpoint | None`: both set or neither (`ConfigError` naming both variables), URL must start `http://` or `https://`. `ENDPOINT_API_KEY_FILE` is expanded and read once; `EndpointError` becomes `ConfigError`.

- [ ] **Step 4: Run the tests and the gate** — expected: all pass.

- [ ] **Step 5: Commit** — `feat(config): v2 defaults, e5 and .minirag/v2, per-model chunk window, endpoint and retrieval variables`

---

### Task 3: Embedder: e5 prefixes, endpoint with a loud fallback, window cut, local token counting, pickling

**Files:**
- Modify: `src/minirag_mcp/embedder.py`, `src/minirag_mcp/endpoints.py` (picklable clients), `src/minirag_mcp/ingest/pipeline.py:205-221` (`_chunk`), `src/minirag_mcp/results.py`, `src/minirag_mcp/sync.py` (`_run_sync_unlocked` tail), `src/minirag_mcp/server.py` (embedder construction, `status`, ingest replies), `src/minirag_mcp/cli/__init__.py` (`_make_embedder`, `status`, ingest replies), `pyproject.toml` (`huggingface-hub>=0.20` in `dependencies`, with the usual "fastembed already pulls this in, but we import it directly" comment), `tests/conftest.py` (`FakeEmbedder` class attributes), `tests/integration/test_real_model.py`, `.github/workflows/ci.yml` (slow-job comment: models ~2.5 GB)
- Test: `tests/test_embedder.py`, `tests/test_endpoints.py`, `tests/test_pipeline.py`, `tests/test_results.py`, `tests/test_sync.py`, `tests/test_parity.py`

**Interfaces:**
- Consumes: Task 1 `EmbeddingEndpoint`, `EndpointError`; Task 2 tables and `Config.embed_endpoints`.
- Produces (plan 2's spawned workers rely on `threads` and pickling):
```python
PASSAGE_PREFIX = "passage: "; QUERY_PREFIX = "query: "
ENDPOINT_RETRY_SECONDS = 60.0
def known_model(model_name: str) -> bool     # in fastembed's registry; no download
class Embedder:
    def __init__(self, model_name: str, cache_dir: Path, *, endpoint: EmbeddingEndpoint | None = None,
                 prefixes: str | None = None, threads: int | None = None)   # prefixes None -> prefix_rule; threads -> TextEmbedding(threads=)
    prefixes: str
    threads: int | None   # public; read when the in-process model is first loaded (plan 2's workers set it on their copy)
    backend: str          # property: "endpoint" | "in-process" | "in-process (endpoint unreachable)"
    warning: str | None   # property: set while fallen back
    max_tokens: int       # property: model_window(model_name)
    prefix_tokens: int    # property: tokens "passage: " adds; 0 when prefixes == "none"
    def count_tokens(self, text: str) -> int          # local tokenizer, without the prefix
    def embed_documents(self, texts) -> list[list[float]]; def embed_query(self, text) -> list[float]
    # picklable: __getstate__ keeps model_name, cache_dir, endpoint, prefixes, threads (and Task 10's expected
    # probe); drops the fastembed model, the token counter, the lock and the fallback state, rebuilt lazily
# EmbeddingEndpoint, RerankEndpoint: picklable; the requests.Session is dropped and rebuilt lazily
# results.py
def embedding_status(embedder) -> dict     # {"embeddingBackend": ..., "embeddingWarning": ... when set}
def embedder_notes(embedder) -> list[str]
def ingest_reply(r: IngestResult, notes: Sequence[str] = ()) -> dict    # + "warnings" when notes
# run_query payload: + "embeddingBackend" (always), "warnings" (list, only when non-empty)
```
Core code reads `prefix_tokens`, `warning`, `backend` with `getattr` defaults (0, None, "in-process"), as `_count_tokens` already does, so test wrappers keep working. `FakeEmbedder` gains class attributes `prefixes = "none"`, `backend = "in-process"`, `warning = None`, `prefix_tokens = 0`, `max_tokens = 128`.

- [ ] **Step 1: Write the failing tests**

Fixtures in `tests/test_embedder.py`: `tokenizer_file` writes a `tokenizers.Tokenizer(WordLevel(...))` with a `Whitespace` pre-tokenizer and a `TemplateProcessing` `[CLS] $A [SEP]` post-processor to `tmp_path/"tokenizer.json"` and monkeypatches `minirag_mcp.embedder.hf_hub_download` to return it. `stub_text_embedding` replaces `embedder.TextEmbedding` with a class that keeps the real `list_supported_models`, records each construction's `model_name` and `threads` and every `embed`/`query_embed` batch, and returns `fake_llama`-style vectors at the registry width.
```python
def e5(tmp_path, fake_llama):
    fake_llama.dim = 1024
    return Embedder(E5_LARGE, tmp_path, endpoint=EmbeddingEndpoint(Endpoint(fake_llama.url("/v1/embeddings"), "e5")))

def test_e5_texts_reach_the_endpoint_prefixed(tmp_path, fake_llama, tokenizer_file):
    emb = e5(tmp_path, fake_llama); emb.embed_documents(["alpha"]); emb.embed_query("beta")
    assert [r.body["input"] for r in fake_llama.requests] == [["passage: alpha"], ["query: beta"]]
    assert emb.backend == "endpoint" and emb.warning is None

def test_minilm_texts_reach_the_endpoint_unchanged(...)        # inputs == [["alpha"], ["beta"]]
def test_in_process_e5_uses_the_same_prefixes_and_threads(tmp_path, stub_text_embedding):
    # Embedder(E5_LARGE, tmp_path, threads=2): batches == [["passage: alpha"], ["query: beta"]], threads == [2]

def test_an_unreachable_endpoint_falls_back_in_process_loudly(tmp_path, dead_url, stub_text_embedding, tokenizer_file):
    emb = Embedder(E5_LARGE, tmp_path, endpoint=EmbeddingEndpoint(Endpoint(dead_url + "/v1/embeddings", "e5")))
    with pytest.warns(RuntimeWarning, match="unreachable"):
        assert len(emb.embed_documents(["alpha"])) == 1
    assert stub_text_embedding.model_names == [E5_LARGE] and stub_text_embedding.batches == [["passage: alpha"]]
    assert emb.backend == "in-process (endpoint unreachable)" and dead_url in emb.warning

def test_the_endpoint_is_tried_again_after_the_interval(...)   # ENDPOINT_RETRY_SECONDS -> 0, endpoint back: backend "endpoint", warning None
def test_an_endpoint_of_another_width_is_refused_loudly(tmp_path, fake_llama, stub_text_embedding, tokenizer_file):  # Review Focus 4
    emb = e5(tmp_path, fake_llama); fake_llama.dim = 8
    with pytest.raises(EmbedderUnavailableError, match=r"(?s)8.*1024|1024.*8") as e: emb.embed_documents(["a"])
    assert "e5" in str(e.value) and stub_text_embedding.batches == []

def test_text_over_the_window_is_cut_before_the_endpoint_sees_it(tmp_path, fake_llama, tokenizer_file, monkeypatch):  # Review Focus 1
    monkeypatch.setitem(MODEL_WINDOWS, E5_LARGE, 8)
    emb = e5(tmp_path, fake_llama); emb.embed_query(" ".join(f"w{i}" for i in range(50)))
    assert fake_llama.requests[-1].body["input"] == ["query: w0 w1 w2 w3"]   # [CLS] query : w0..w3 [SEP] = 8
    assert emb.backend == "endpoint"

def test_the_endpoint_backend_counts_with_the_tokenizer_file_only(tmp_path, fake_llama, tokenizer_file, monkeypatch):
    monkeypatch.setattr(embedder_mod, "TextEmbedding", boom)          # the ONNX model is never built
    assert e5(tmp_path, fake_llama).count_tokens("a b c") == 5        # 3 + 2 specials, prefix not counted

def test_prefix_tokens_is_what_the_prefix_adds(...)   # e5 -> 2 ("passage", ":"); MiniLM -> 0
def test_a_missing_tokenizer_file_is_loud(...)        # hf_hub_download raises -> EmbedderUnavailableError naming "qdrant/multilingual-e5-large-onnx"
def test_max_tokens_is_the_models_window(tmp_path):
    assert Embedder(E5_LARGE, tmp_path).max_tokens == 512 and Embedder(MINILM, tmp_path).max_tokens == 128
def test_known_model_reads_the_registry():
    assert known_model(E5_LARGE) and not known_model("x/unknown")

def test_an_endpoint_embedder_survives_pickling(tmp_path, fake_llama, tokenizer_file):
    emb = e5(tmp_path, fake_llama); before = emb.embed_documents(["alpha"])      # session and counter now exist
    clone = pickle.loads(pickle.dumps(emb))
    assert clone.embed_documents(["alpha"]) == before and clone.prefixes == "e5" and clone.backend == "endpoint"
def test_an_in_process_embedder_survives_pickling_without_its_model(tmp_path, stub_text_embedding):
    emb = Embedder(E5_LARGE, tmp_path, threads=2); before = emb.embed_documents(["alpha"])
    clone = pickle.loads(pickle.dumps(emb))
    assert clone.embed_documents(["alpha"]) == before and stub_text_embedding.threads == [2, 2]  # rebuilt, same threads
```
```python
# tests/test_endpoints.py
def test_clients_survive_pickling(fake_llama)   # EmbeddingEndpoint used, pickled, unpickled -> same vectors
# tests/test_pipeline.py
def test_chunks_leave_room_for_the_passage_prefix(tmp_path):
    class Prefixed(FakeEmbedder): prefixes, prefix_tokens = "e5", 5
    cfg = replace(load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path), token_budget=40)
    p = Pipeline(Store(tmp_path / "db", dim=8), Prefixed(), cfg)
    p.ingest_data(" ".join(f"word{i}." for i in range(300)), source="s")
    assert max(Prefixed().count_tokens(c.text) for c in p.store.all_chunks("s")) <= 35
# tests/test_results.py
def test_a_query_names_its_embedding_backend_and_warns_on_a_fallback(tmp_path, root, store, fake_embedder):
    payload = run_query(store, fake_embedder, make_config(root, tmp_path), "budget review")
    assert payload["embeddingBackend"] == "in-process" and "warnings" not in payload
    fallen = FakeEmbedder(); fallen.backend = "in-process (endpoint unreachable)"; fallen.warning = "W"
    assert run_query(store, fallen, make_config(root, tmp_path), "budget review")["warnings"] == ["W"]
# test_an_unfiltered_query_has_exactly_the_old_keys -> {"results", "sources", "parents", "embeddingBackend"}
# tests/test_sync.py
def test_an_embedding_fallback_is_reported_but_not_a_failure(env)
    # embedder.warning = "W" -> errors contains {"source": str(store.db_path), "error": "embedding: W"}; counts["failed"] == 0
# tests/test_parity.py
async def test_both_shells_report_the_embedding_backend(...)   # EMBED_URL/EMBED_ENDPOINT_MODEL set, real factory:
    # status["embeddingBackend"] == "endpoint" in both shells, equal dicts
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_embedder.py tests/test_endpoints.py tests/test_pipeline.py tests/test_results.py tests/test_sync.py tests/test_parity.py -q`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'endpoint'`, missing keys, `TypeError: cannot pickle '_thread.lock'`).

- [ ] **Step 3: Implement**

`embedder.py`:
- In-process: `TextEmbedding(model_name=..., cache_dir=..., threads=threads)`; documents `embed([PASSAGE_PREFIX + t ...])` and queries `query_embed([QUERY_PREFIX + q])` when `prefixes == "e5"`; otherwise as today.
- Endpoint: each prefixed text is cut to `max_tokens` with the counting tokenizer: encode untruncated; when longer, keep the text up to the end offset of the last non-special token among the first `max_tokens` ids (fastembed truncates to that many ids, specials included). Then `endpoint.embed`; every vector must have `self.dim` components, else `EmbedderUnavailableError` naming the endpoint URL, router model, both widths and the model; no fallback.
- `EndpointError` -> fall back to the in-process model of the same name, prefixes and threads: `backend` becomes `"in-process (endpoint unreachable)"`, `warning` = `f"The embedding endpoint {url} is unreachable ({cause}). {model} runs in-process meanwhile: same vectors, slower. The endpoint is tried again every {ENDPOINT_RETRY_SECONDS:g} s."`, `warnings.warn(..., RuntimeWarning)` once per fallback. The endpoint is retried at most every `ENDPOINT_RETRY_SECONDS` (`time.monotonic`); a success clears the warning. State changes under a `threading.Lock`.
- `count_tokens` with an endpoint: `hf_hub_download(repo_id=<the model's sources["hf"] from TextEmbedding.list_supported_models()>, filename="tokenizer.json", cache_dir=str(cache_dir))`, truncation and padding off; a failed download raises `EmbedderUnavailableError` naming the repo and the cache folder ("needed to size chunks"). Without an endpoint, today's counter. `prefix_tokens = count(PASSAGE_PREFIX + "x") - count("x")`, cached.
- `__getstate__`/`__setstate__` as in the Interfaces block; the endpoint clients do the same for their session.

Elsewhere:
- `Pipeline._chunk` packs to `config.token_budget - prefix_tokens`.
- `_run_sync_unlocked`, last: a non-empty `pipeline.embedder.warning` appends `{"source": str(store.db_path), "error": f"embedding: {warning}"}` (not a failure).
- server `ctx()` and CLI `_make_embedder` build `Embedder(cfg.model_name, cfg.cache_dir, endpoint=EmbeddingEndpoint(ep) if (ep := cfg.embed_endpoints.get(cfg.model_name)) else None)` (Task 5 moves this into `make_embedder`); both `status` add `embedding_status(embedder)`; both ingest paths pass `embedder_notes(embedder)` to `ingest_reply`; `query_documents`'s docstring documents `embeddingBackend` and `warnings`.
- `tests/integration/test_real_model.py`: the three existing tests use `MINILM` explicitly; add (all `slow`):
```python
def test_real_e5_vectors_prefixes_and_window(model_cache):
    emb = Embedder(E5_LARGE, cache_dir=model_cache)
    v = emb.embed_documents([ru_auth, en_auth, ru_cook])
    assert all(len(x) == 1024 and abs(math.hypot(*x) - 1) < 1e-6 for x in v)
    assert cosine(v[0], v[1]) > cosine(v[0], v[2])
    assert emb.max_tokens == 512 and emb.prefix_tokens > 0
    assert emb._load().token_count("слово " * 1000) == 512
def test_real_e5_tokenizer_file_counts_like_the_loaded_model(model_cache):   # every corpus_samples line: equal counts
def test_e5_chunks_fit_the_window_with_the_prefix(model_cache):  # budgets default_budget(E5_LARGE) and 512: count + prefix_tokens <= 512
```

- [ ] **Step 4: Run the tests and the gate** — expected: all pass. Then `uv run pytest -m slow -q` once (downloads e5, ~2.24 GB): pass.

- [ ] **Step 5: Commit** — `feat(embedder): e5 prefixes, llama.cpp endpoint with loud in-process fallback, local token counting`

---

### Task 4: Index identity module

**Files:**
- Create: `src/minirag_mcp/identity.py`
- Modify: `src/minirag_mcp/lock.py` (`is_held`), `tests/conftest.py` (`stamp_identity`)
- Test: `tests/test_identity.py`, `tests/test_lock.py`

**Interfaces:**
- Consumes: `lock.SyncLock`; lazily (inside functions, because `embedder.py` imports this module's probe constants): `embedder.known_model`, `config.default_db_path`.
- Produces (contract, plus `PROBE_COSINE`, `IdentityError`, `legacy_error`, `index_state`, `rebuild_kind`, `recover_swap`, `rebuild_running`, `RebuildInfo`, `lock.is_held`; plan 3 calls `index_state`):
```python
INDEX_FILE = "index.json"; FORMAT = 2; PROBE_COSINE = 0.999
PROBE_TEXT = ("minirag index probe. The quick brown fox jumps over the lazy dog. "
              "Съешь же ещё этих мягких французских булок, да выпей чаю.")   # never changed
EmbeddingScheme, ChunkingScheme, IndexIdentity, IndexState, LegacyIndexError   # as in the contract; probe == () until recorded
class IdentityError(Exception): ...           # unreadable, malformed, or a format newer than FORMAT
@dataclass(frozen=True)
class RebuildInfo:
    kind: str; reasons: tuple[str, ...]; generation: int; reembedded: int; restored: int
    def to_dict(self) -> dict                  # {"kind", "reasons", "generation", "reembedded", "restored"}
def read_identity(db_path: Path) -> IndexIdentity | None
def write_identity(db_path: Path, identity: IndexIdentity) -> None
def is_legacy_index(db_path: Path) -> bool    # (db_path / "chunks.lance").is_dir() and no index.json
def legacy_error(db_path: Path, roots: Sequence[Path]) -> LegacyIndexError
def differences(recorded: IndexIdentity, embedding: EmbeddingScheme, chunking: ChunkingScheme) -> list[str]
def rebuild_kind(recorded, embedding, chunking) -> str        # "none" | "reembed" | "reingest"
def index_state(config: Config, embedder) -> tuple[IndexState, str]   # "" reason when READY
def staging_dir(db_path: Path) -> Path; def previous_dir(db_path: Path) -> Path
def swap_in(db_path: Path) -> None
def recover_swap(db_path: Path) -> bool
def rebuild_running(db_path: Path) -> bool    # staging dir exists and the sync lock is held
# lock.py
def is_held(db_path: Path) -> bool            # tries the lock without blocking and releases it at once
# tests/conftest.py
def stamp_identity(config: Config, embedder) -> IndexIdentity   # writes what a handle writes for that embedder
```
File layout (spec section 3; the probe sits under `embedding`):
```json
{"format": 2, "minirag": "<__version__>", "generation": 1,
 "embedding": {"model": "...", "dim": 1024, "prefixes": "e5", "probe": [...]},
 "chunking": {"scheme_version": 3, "token_budget": 110},
 "fts_language": "Russian", "document_table": true}
```
`differences` texts: `f"model {a} -> {b}"`, `f"query/passage prefixes {a} -> {b}"`, `f"vector width {a} -> {b}"` (only when the model is the same), `f"chunking scheme {a} -> {b}"`, `f"chunk budget {a} -> {b} tokens"`; `fts_language` is never one. `rebuild_kind`: a chunking difference -> `"reingest"`, else an embedding one -> `"reembed"`, else `"none"`.

`legacy_error` message: `f"{db_path} holds a minirag 1.x index (a LanceDB table and no {INDEX_FILE}). minirag 2 never writes to a 1.x index. Leave DB_PATH unset to use {default_db_path(roots)}, or point it at an empty folder."`

`index_state(config, embedder)`: `embedder` is the one built for the configured model; the configured schemes are `EmbeddingScheme(embedder.model_name, embedder.dim, embedder.prefixes)` and `ChunkingScheme(SCHEME_VERSION, config.token_budget)`. Cheap: reads `index.json`, stats the folder, consults fastembed's registry; never loads a model or calls an endpoint. In order: `recover_swap(db)`; `IdentityError` -> `(UNAVAILABLE, str(error))`; no identity -> `(UNAVAILABLE, legacy message)` for a legacy index, else `(READY, "")` (a fresh folder is stamped at its first open); recorded model neither the embedder's model, nor a key of `config.embed_endpoints`, nor `known_model` -> `(UNAVAILABLE, f"this index was built with {model}, which cannot be loaded here: no embedding endpoint is configured for it and fastembed does not know it")`; differences -> `(REBUILD_PENDING, "; ".join(differences))`; else `(READY, "")`. `FakeEmbedder` on a fresh or fake-built index is READY.

- [ ] **Step 1: Write the failing tests**

```python
def test_identity_round_trips_in_the_spec_layout(tmp_path)   # json["embedding"]["probe"] is a list; read == written
def test_a_missing_file_reads_as_none(tmp_path)
@pytest.mark.parametrize("raw", ["{", "[]", '{"format": 2}', NEWER])    # NEWER: a valid file with "format": 3
def test_an_unreadable_or_newer_identity_is_an_error(tmp_path, raw)
def test_a_failed_write_leaves_the_old_file(tmp_path, monkeypatch)        # os.replace raises -> old content intact, no tmp left
def test_a_lancedb_table_without_identity_is_legacy(tmp_path):
    Store(tmp_path / "db", dim=8)
    assert is_legacy_index(tmp_path / "db")
    write_identity(tmp_path / "db", IDENT); assert not is_legacy_index(tmp_path / "db")
    assert not is_legacy_index(tmp_path / "empty")
def test_differences_name_what_changed()          # model, prefixes, budget, scheme; a changed fts_language is not listed
def test_rebuild_kind()                           # reingest wins over reembed

def test_index_state_reads_files_and_never_loads_a_model(tmp_path, fake_embedder, monkeypatch):
    monkeypatch.setattr(embedder_mod, "TextEmbedding", RegistryOnly)     # list_supported_models works, construction raises
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    assert index_state(cfg, fake_embedder) == (IndexState.READY, "")            # fresh folder
    stamp_identity(cfg, fake_embedder)
    assert index_state(cfg, fake_embedder) == (IndexState.READY, "")
    assert index_state(replace(cfg, token_budget=200), fake_embedder) == (IndexState.REBUILD_PENDING, "chunk budget 110 -> 200 tokens")
def test_index_state_refuses_a_legacy_index_naming_the_v2_path(tmp_path, fake_embedder)
def test_index_state_needs_a_way_to_load_the_recorded_model(tmp_path, fake_embedder):
    # recorded "x/unknown": UNAVAILABLE; with replace(cfg, embed_endpoints={"x/unknown": ep}): REBUILD_PENDING

def test_swap_in_bumps_generation_and_keeps_one_previous(tmp_path):
    # db gen 1; staged gen 1 -> swap -> db gen 2, previous gen 1; stage again -> swap -> db gen 3, previous gen 2, one previous
def test_swap_in_refuses_a_staging_without_identity(tmp_path)
def test_a_failed_second_rename_puts_the_index_back(tmp_path, monkeypatch)
def test_recover_swap_finishes_an_interrupted_swap(tmp_path):              # Review Focus 3
    # previous holds gen 1, staging holds gen 2, db missing -> recover_swap -> True, db gen 2; nothing to do -> False
def test_rebuild_running_needs_both_the_staging_dir_and_the_lock(tmp_path)
# tests/test_lock.py
def test_is_held_reports_a_holder_without_taking_the_lock(tmp_path)        # held by another SyncLock -> True; free -> False and still free
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_identity.py tests/test_lock.py -q`; expected: `ModuleNotFoundError: minirag_mcp.identity`.

- [ ] **Step 3: Implement**

`write_identity`: create the folder, write `index.json.tmp` in it, fsync, `os.replace`. `swap_in`: the staging identity is required (`IdentityError`); its generation becomes `max(staging, current or 0) + 1`; remove `previous`; rename `db -> previous` when `db` exists; rename `staging -> db`; when the second rename raises, rename `previous` back and re-raise. `recover_swap`: `db` missing, `previous` present, `staging` holding an identity -> rename `staging -> db`.

- [ ] **Step 4: Run the tests and the gate** — expected: all pass.

- [ ] **Step 5: Commit** — `feat(identity): index.json, legacy detection, index state and atomic swap`

---

### Task 5: One place opens an index: `IndexHandle` and `make_embedder`

**Files:**
- Create: `src/minirag_mcp/handle.py`
- Modify: `src/minirag_mcp/server.py` (`_Ctx`/`ctx()` replaced; `status`), `src/minirag_mcp/cli/__init__.py` (`_make_embedder`, `_label_source`, `_load`, `status`), `src/minirag_mcp/results.py` (`status_payload`)
- Test: `tests/test_handle.py`, `tests/test_parity.py`

**Interfaces:**
- Consumes: Task 3 `Embedder`, `embedder_notes`, `embedding_status`; Task 4 `EmbeddingScheme`.
- Produces (plan 3's registry holds one `IndexHandle` per index and builds its shared embedders with `make_embedder`; plan 2's sync workers never build an embedder: they receive the pipeline's own by pickle (Task 3), so a pending index's scoped sync and a rebuild embed with the scheme the pipeline was opened with, and carry its expected probe (Task 10)):
```python
EmbedderFactory = Callable[[EmbeddingScheme], object]
def configured_scheme(config: Config) -> EmbeddingScheme   # (model_name, registry width, prefix_rule(model_name)); UnknownModelError for an unknown model
def make_embedder(config: Config, scheme: EmbeddingScheme, *, threads: int | None = None) -> Embedder
    # endpoint = config.embed_endpoints.get(scheme.model); prefixes = scheme.prefixes; cache_dir = config.cache_dir
@dataclass(frozen=True)
class OpenIndex:
    config: Config; store: Store; embedder: object; pipeline: Pipeline; labels: LabelerSource
class IndexHandle:
    config: Config
    def __init__(self, config: Config, *, embedder_factory: EmbedderFactory | None = None,
                 model_factory: ModelFactory | None = None) -> None    # factory default: lambda s: make_embedder(config, s)
    def get(self) -> OpenIndex        # opens once under a lock; a failed open raises, the next call retries
    @property
    def sync(self) -> SyncManager     # built with the first successful get()
    def notes(self) -> list[str]      # result "warnings" owned here: embedder_notes(...)
def status_payload(config: Config, handle: IndexHandle) -> dict   # results.py; the whole status for a valid config
```
`create_app` keeps its signature; with `embedder=` it builds `IndexHandle(config, embedder_factory=lambda s: embedder, ...)`. CLI: `_make_embedder(cfg) -> object | None` returns None ("patched in tests; None builds through make_embedder"), `_make_model_factory()` unchanged, `_load(...) -> tuple[Config, IndexHandle, OpenIndex]`. Degraded `status` (configuration error) stays in the shells.

- [ ] **Step 1: Write the failing tests**

```python
def test_get_returns_one_open_index(tmp_path, fake_embedder):
    h = IndexHandle(load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path), embedder_factory=lambda s: fake_embedder)
    a, b = h.get(), h.get()
    assert a is b and a.pipeline.store is a.store and a.embedder is fake_embedder

def test_a_failed_open_is_retried_by_the_next_call(tmp_path, fake_embedder):
    (tmp_path / "db").write_text("not a folder")
    h = IndexHandle(load_config({"BASE_DIR": str(tmp_path), "DB_PATH": str(tmp_path / "db")}, cwd=tmp_path),
                    embedder_factory=lambda s: fake_embedder)
    with pytest.raises(IndexOpenError): h.get()
    (tmp_path / "db").unlink()
    assert h.get().store.chunk_count() == 0

def test_make_embedder_takes_each_models_own_endpoint(tmp_path):
    ep = Endpoint("http://127.0.0.1:9/v1/embeddings", "e5")
    cfg = replace(load_config({}, cwd=tmp_path), embed_endpoints={E5_LARGE: ep})
    assert configured_scheme(cfg) == EmbeddingScheme(E5_LARGE, 1024, "e5")
    assert make_embedder(cfg, configured_scheme(cfg)).backend == "endpoint"
    assert make_embedder(cfg, EmbeddingScheme(MINILM, 384, "none")).backend == "in-process"
    both = replace(cfg, embed_endpoints={E5_LARGE: ep, MINILM: ep})
    assert make_embedder(both, EmbeddingScheme(MINILM, 384, "none")).backend == "endpoint"

# tests/test_parity.py
async def test_status_is_identical_in_both_shells(root, tmp_path, capsys, monkeypatch, fake_embedder):
    # after a CLI sync: the MCP status dict == json.loads(CLI `status --json`)
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_handle.py tests/test_parity.py -q`; expected: `ModuleNotFoundError: minirag_mcp.handle` (the parity test may already pass; it guards the refactor).

- [ ] **Step 3: Implement** `handle.py`, `status_payload`, and switch both shells to them. `server.py` and the CLI keep no construction logic of their own; the handle builds its embedder with `embedder_factory(configured_scheme(config))`.

- [ ] **Step 4: Run the tests and the gate** — expected: all pass, no other test changed.

- [ ] **Step 5: Commit** — `refactor(core): IndexHandle opens an index for both shells; one status payload`

---

### Task 6: Classify served by llama-server

**Files:**
- Create: `src/minirag_mcp/labels/endpoint_model.py`
- Modify: `src/minirag_mcp/labels/labeler.py` (`needs_extra`, `backend`), `src/minirag_mcp/handle.py` (model factory choice), `src/minirag_mcp/results.py` (`_labels_status` adds `"backend"`)
- Test: `tests/test_endpoint_model.py`, `tests/test_labeler.py`, `tests/test_handle.py`

**Interfaces:**
- Consumes: Task 1 `Endpoint`, `request_json`, `EndpointError`; `labels.model` (`LETTERS`, `N_CTX`, `build_prompt`, `calibrate`, `ModelError`, `_sha256`).
- Produces:
```python
DEFAULT_N_PROBS = 100
class EndpointChoiceModel:                     # implements labels.model.ChoiceModel
    def __init__(self, endpoint: Endpoint, temperature: float, *,
                 n_probs: int = DEFAULT_N_PROBS, expected: ModelSpec | None = None) -> None
    def probabilities(self, evidence: str, question: str, options: Sequence[tuple[str, str]]) -> dict[str, float]
    def close(self) -> None
def endpoint_model_factory(endpoint: Endpoint) -> ModelFactory
    # (spec, cache_dir) -> EndpointChoiceModel(endpoint, spec.temperature, expected=spec)
Labeler(taxonomy, roots, cache_dir, model_factory, *, needs_extra: bool = True)
LabelerSource(roots, cache_dir, model_factory, *, needs_extra: bool = True); LabelerSource.backend -> "endpoint" | "in-process"
```
`endpoint.url` is the server base (`http://127.0.0.1:8080`); routes `/tokenize`, `/completion`, `GET /props?model=<name>`. The handle picks `model_factory` when injected; else `endpoint_model_factory(cfg.classify_endpoint)` with `needs_extra=False` when set; else `default_model_factory`.

- [ ] **Step 1: Write the failing tests**

`pinned` fixture: a file `tmp_path/"plumb.gguf"` with bytes `b"gguf"`, `ModelSpec(temperature=2.07, path=<it>)`, and `fake_llama.props = {"model_path": str(<it>)}`.
```python
OPTS = [("a", "A"), ("b", "B"), ("c", "C")]
def model(fake_llama, pinned, **kw): return EndpointChoiceModel(Endpoint(fake_llama.url(), "plumb", "k"), 2.07, expected=pinned, **kw)

def test_probabilities_equal_the_in_process_readout(fake_llama, pinned):
    fake_llama.top_logprobs = [{"id": 32, "token": "A", "logprob": -0.1}, {"id": 33, "token": "B", "logprob": -2.5},
                               {"id": 7, "token": "x", "logprob": -3.0}, {"id": 34, "token": "C", "logprob": -4.0}]
    got = model(fake_llama, pinned).probabilities("ev", "q?", OPTS)
    assert got == pytest.approx(dict(zip("abc", calibrate([-0.1, -2.5, -4.0], 2.07))))

def test_the_prompt_goes_as_token_ids_with_the_measured_settings(fake_llama, pinned):
    model(fake_llama, pinned).probabilities("ev", "q?", OPTS)
    tok = [r for r in fake_llama.requests if r.path == "/tokenize"][-1].body
    assert tok == {"content": build_prompt("ev", "q?", OPTS), "add_special": False, "parse_special": True, "model": "plumb"}
    done = next(r for r in fake_llama.requests if r.path == "/completion").body
    assert {k: v for k, v in done.items() if k != "prompt"} == {"n_predict": 1, "n_probs": 100,
        "post_sampling_probs": False, "cache_prompt": False, "temperature": 0, "samplers": [], "model": "plumb"}
    assert all(isinstance(t, int) for t in done["prompt"])

def test_letters_are_matched_by_id_not_by_text(...)        # {"id": 999, "token": "A", -0.01} ignored; id 32 used
def test_a_letter_outside_the_top_n_gets_zero(...)         # C absent -> got["c"] == 0.0
def test_no_letter_in_the_top_n_is_a_model_error(...)      # match "raise n_probs"
def test_a_prompt_over_the_context_is_refused_before_completion(fake_llama, pinned):
    m = model(fake_llama, pinned); fake_llama.token_count = N_CTX
    with pytest.raises(ModelError, match=str(N_CTX)): m.probabilities("ev", "q?", OPTS)
    assert "/completion" not in [r.path for r in fake_llama.requests]
def test_every_request_names_the_router_model_and_carries_the_key(...)
def test_the_served_file_must_be_the_pinned_one(fake_llama, tmp_path):
    # props -> a file with other bytes; ModelSpec(sha256=<sha of b"gguf">) -> ModelError naming both; same bytes -> constructs
def test_props_without_a_model_path_is_refused(...)
def test_an_unreachable_server_is_a_model_error(dead_url, pinned)
def test_a_letter_that_is_two_tokens_is_refused(...)      # fake_llama.split_letters = True
# tests/test_labeler.py
def test_a_served_model_needs_no_classify_extra(monkeypatch, ...):
    # labels.model.available -> False; LabelerSource(..., needs_extra=False).get()[0].model_status() == ("on demand", None)
# tests/test_handle.py
def test_classify_url_selects_the_served_model(tmp_path, fake_embedder, ...):  # h.get().labels.backend == "endpoint"; status labels block "backend" == "endpoint"
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_endpoint_model.py tests/test_labeler.py tests/test_handle.py -q`; expected: `ModuleNotFoundError: minirag_mcp.labels.endpoint_model`.

- [ ] **Step 3: Implement**

Startup, in order: (1) the served-model check: `GET /props?model=<name>` -> `model_path`; with `expected.path` the resolved paths must be equal, with `expected.sha256` `labels.model._sha256(Path(model_path))` must equal it; a missing `model_path`, an unreadable file or a mismatch raises `ModelError` naming the served file and the pinned one. (2) Letters A..P tokenized with `parse_special: false`: one token each and 16 distinct ids. Per decision: `/tokenize {content: build_prompt(...), add_special: false, parse_special: true}`; `len >= N_CTX` -> `ModelError` before `/completion`; `/completion {prompt: ids, n_predict: 1, n_probs, post_sampling_probs: false, cache_prompt: false, temperature: 0, samplers: []}`; letters by id from `completion_probabilities[0].top_logprobs`; a letter absent -> `-inf`; all absent -> `ModelError(f"no option letter is among the top {n} next tokens; raise n_probs")`; `calibrate(picked, temperature)`. Every `EndpointError` and malformed reply becomes `ModelError`. `close()` does nothing. `Labeler._problem` skips the `available()` check when `needs_extra` is False.

- [ ] **Step 4: Run the tests and the gate** — expected: all pass.

- [ ] **Step 5: Commit** — `feat(labels): classify through llama-server, checked against the pinned model file (g7q.3)`

---

### Task 7: The `documents` table and the document-BM25 bonus

**Files:**
- Modify: `src/minirag_mcp/store.py`, `src/minirag_mcp/results.py` (`run_query` passes `document_weight`; `documents_status`; `status_payload`), `src/minirag_mcp/sync.py` (backfill)
- Test: `tests/test_store_documents.py` (new), `tests/test_sync.py`, `tests/test_results.py`

**Interfaces:**
- Consumes: `join_document`, the signature-table patterns in `store.py`.
- Produces (plan 2's `Store.replace_sources(batch, *, indexed)` builds rows with `_document_row` and writes them with `_write_documents`, exactly as `replace_source` does; anything left without a row is filled by the backfill every sync runs):
```python
DOCUMENT_TABLE = "documents"     # source, title, text, file_hash, ingested_at
DOCUMENT_FETCH = 100
Store._document_row(self, source: str, records: Sequence[ChunkRecord]) -> dict   # join_document of the records in chunk order
Store._write_documents(self, rows: Sequence[dict]) -> None
    # one merge on ["source", "ingested_at"] that also deletes the other rows of those sources; warns, never raises
Store._delete_documents(self, sources: Sequence[str]) -> None    # warns, never raises
Store.search(..., document_weight: float = 0.0)
Store.backfill_documents(self, scopes: tuple[str, ...] = (), on_event: Callable[[str], None] | None = None) -> tuple[int, list[dict]]
Store.undocumented_sources(self) -> list[str]
results.documents_status(store) -> dict  # {"undocumentedSourceCount": n, "documentsWarning": "... a sync builds them"} when n > 0
```

- [ ] **Step 1: Write the failing tests**

```python
def test_replace_source_writes_one_row_with_the_documents_joined_text(store)   # text == join_document(records), title, file_hash, ingested_at
def test_reingest_delete_and_move_keep_one_row_per_source(store)
def test_one_write_of_several_rows_matches_one_replace_each(tmp_path)          # _write_documents([row_a, row_b]) == two replace_source calls' rows
def test_document_rows_match_with_the_index_stemmer(tmp_path)                  # fts_language="Russian": "договоры" finds the row "договор поставки"
def test_a_language_change_rebuilds_the_document_index_too(tmp_path)

def test_the_bonus_follows_the_formula(tmp_path):
    s = Store(tmp_path / "db", dim=8); s.replace_source("/a.md", [rec("/a.md", 0, "alpha", V(0.0))])
    got = s.search("alpha", V(0.0), top_k=1, hybrid_weight=0.6, document_weight=0.5)
    assert got[0].score == pytest.approx((0.4 + 0.6 + 0.5) / 61)

def test_a_document_matching_as_a_whole_lifts_its_chunk(tmp_path):
    # /b.md: one chunk "gamma delta" at V(1.0); /a.md: "alpha one" at V(0.0) and "beta two" at V(0.1)
    assert s.search("alpha beta", V(1.0), top_k=3, hybrid_weight=0.0)[0].source == "/b.md"
    assert s.search("alpha beta", V(1.0), top_k=3, hybrid_weight=0.0, document_weight=0.5)[0].source == "/a.md"

def test_weight_zero_ranks_as_before(store)
def test_the_bonus_respects_scope(tmp_path)          # a document outside the scope takes no rank from the scoped one
def test_backfill_fills_rows_for_chunks_written_without_them(tmp_path):
    # rows added with s._table.add(...) -> undocumented_sources() == ["/a.md"]; backfill -> (1, []); again -> (0, [])
def test_update_indices_folds_new_document_rows(store)  # num_unindexed_rows == 0 on the documents FTS indices
def test_a_failed_document_write_never_fails_the_chunk_write(store, monkeypatch)  # warns "document"; chunks stored
def test_a_failing_document_query_degrades_to_no_bonus(store, monkeypatch)
# tests/test_sync.py
def test_sync_builds_missing_document_rows(env)
# tests/test_results.py
def test_status_counts_sources_without_a_document_row(...)
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_store_documents.py -q`; expected: `TypeError: ... 'document_weight'` and missing table.

- [ ] **Step 3: Implement**

- The table is created with the chunks table; an index opened without it gets it best-effort, as the signature table (warn, never fatal). FTS on `text` and `title` in `fts_language`; `_ensure_fts_indices` covers both tables and shares the `.fts-language` record, so a language change rebuilds both.
- `replace_source` calls `_write_documents([_document_row(source, records)])` after the chunk merge, or `_delete_documents([source])` for no records. `move_source` renames the row, `delete_source` deletes it. `update_indices` folds both tables. `backfill_documents` mirrors `backfill_signatures` (sources whose chunks carry one stamp and have no row of that stamp; batches of 64; errors prefixed `"documents:"`).
- Fusion, ranks 1-based, ties by id:
```
score(c) = (1-w)/(60 + r_vec(c)) + w/(60 + r_fts(c)) + wd/(60 + r_doc(source(c)))
```
`r_doc` comes from a documents FTS query over `text` and `title`, top `DOCUMENT_FETCH`, filtered by the scope clause only (the table has no label columns). A chunk whose document is not in that list, or a failing documents query, adds nothing. The vector-only branch is taken only when `hybrid_weight <= 0` and `document_weight <= 0`.
- `run_query` passes `config.document_weight`. `_run_sync_unlocked` runs `store.backfill_documents(scope)` after `backfill_signatures` and before `update_indices`, its errors appended. `status_payload` adds `documents_status`.

- [ ] **Step 4: Run the tests and the gate** — expected: all pass.

- [ ] **Step 5: Commit** — `feat(store): whole-document BM25 as a third fusion signal (g7q.4)`

---

### Task 8: Reranking the top chunks

**Files:**
- Modify: `src/minirag_mcp/store.py` (`SearchResult.rerank_score`, `search`), `src/minirag_mcp/results.py` (`run_query`, `result_dict`, `rerank_status`, `status_payload`), `src/minirag_mcp/handle.py` (`OpenIndex.reranker`), `src/minirag_mcp/server.py` (`query_documents` passes it; docstring), `src/minirag_mcp/cli/__init__.py` (`query`)
- Test: `tests/test_store_search.py`, `tests/test_results.py`, `tests/test_parity.py`

**Interfaces:**
- Consumes: Task 1 `RerankEndpoint`; Task 7 search.
- Produces:
```python
SearchResult.rerank_score: float | None = None
Store.search(..., reranker: Callable[[str, list[str]], list[float]] | None = None, rerank_top: int = 30)
OpenIndex.reranker: RerankEndpoint | None          # from config.rerank_endpoint
run_query(..., reranker: RerankEndpoint | None = None)   # payload "reranked": bool, always
result_dict: "rerankScore" only on a reranked hit
results.rerank_status(reranker: RerankEndpoint | None) -> dict
    # {"reranker": "off" | "endpoint" | "unreachable", "rerankWarning": last_error when unreachable}
```

- [ ] **Step 1: Write the failing tests**

```python
def last_highest(query, texts): return [float(i) for i in range(len(texts))]

def test_the_reranker_reorders_only_the_top_n(store12):      # 12 chunks
    first = store12.search("q", V(0.0), top_k=12, hybrid_weight=0.6)
    got = store12.search("q", V(0.0), top_k=12, hybrid_weight=0.6, reranker=last_highest, rerank_top=5)
    assert [g.text for g in got[:5]] == [f.text for f in first[:5]][::-1]
    assert [g.text for g in got[5:]] == [f.text for f in first[5:]]
    assert [g.score for g in got[:5]] == [f.score for f in first[:5]][::-1]
    assert all(g.rerank_score is not None for g in got[:5]) and all(g.rerank_score is None for g in got[5:])

def test_the_reranker_sees_the_query_and_the_chunk_texts(store)
def test_a_failing_reranker_keeps_the_first_stage_order(store)
def test_a_wrong_number_of_scores_keeps_the_first_stage_order(store)
def test_distance_filters_apply_before_reranking(hybrid_store)   # max_distance: the FTS-only hit never reaches the reranker
def test_the_fetch_window_covers_rerank_top(tmp_path)            # 80 chunks, top_k=5, rerank_top=60 -> the reranker got 60 texts
def test_max_files_applies_after_reranking(store)
# tests/test_results.py
def test_a_served_reranker_reorders_and_says_so(fake_llama, ...)       # payload["reranked"] is True, results[0]["rerankScore"]
def test_an_unreachable_reranker_is_flagged_in_the_result_and_in_status(dead_url, ...):
    # payload["reranked"] is False; a warning starts "Reranking was skipped";
    # rerank_status(rr) == {"reranker": "unreachable", "rerankWarning": rr.last_error}
def test_without_a_reranker_reranked_is_false(...)
# test_an_unfiltered_query_has_exactly_the_old_keys -> + "reranked"
# tests/test_parity.py
async def test_reranked_queries_are_identical_in_both_shells(fake_llama, ...)   # RERANK_URL/RERANK_MODEL -> fake_llama
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_store_search.py tests/test_results.py -q`; expected: `TypeError: ... 'reranker'`.

- [ ] **Step 3: Implement**

`fetch = max(top_k * 4, 50, rerank_top if reranker else 0)`. After `max_distance` and `grouping`: `head = results[:rerank_top]`, `scores = reranker(query_text, [r.text for r in head])`; any exception or a wrong count keeps the fused order; otherwise `head` is stable-sorted by descending score (ties keep fused order) and carries `rerank_score`; the rest follow unchanged; then `max_files`, `top_k`, parents. `run_query` wraps the endpoint in a per-call recorder (used, error) so concurrent queries share no state; an error adds `f"Reranking was skipped ({error}); results are in first-stage order."` to `warnings`. `query_documents`'s docstring: results are in reranked order when a reranker is configured, `rerankScore` is the cross-encoder's score, `score` the first-stage score, `reranked` says which happened.

- [ ] **Step 4: Run the tests and the gate** — expected: all pass.

- [ ] **Step 5: Commit** — `feat(search): rerank the top fused chunks with a served cross-encoder (g7q.7)`

---

### Task 9: An index opens by its recorded identity

**Files:**
- Modify: `src/minirag_mcp/handle.py`, `src/minirag_mcp/results.py` (`status_payload`), `src/minirag_mcp/server.py` and `src/minirag_mcp/cli/__init__.py` (`IndexUnavailableError` -> `ToolError` / `_fail`), `src/minirag_mcp/sync.py` (`SyncManager.running`), and every test that builds an index with `Store(...)` or `store_helpers._older_table` at a path a shell opens afterwards without a shell having opened it first (known: `tests/test_ignore.py:219`): call `stamp_identity` first
- Test: `tests/test_handle.py`, `tests/test_parity.py`

**Interfaces:**
- Consumes: Task 4 identity, `index_state`; Task 5 handle.
- Produces:
```python
class IndexUnavailableError(Exception): ...     # handle.py; says why and what to do
OpenIndex.identity: IndexIdentity
IndexHandle.state(self) -> tuple[IndexState, str]   # never raises: open failure -> (UNAVAILABLE, str(error)); job running -> (SYNCING, "");
                                                    # else index_state(config, configured embedder), the name imported into handle.py
                                                    # (plan 3's tests patch minirag_mcp.handle.index_state)
IndexHandle.configured(self) -> tuple[EmbeddingScheme, ChunkingScheme]
SyncManager.running -> bool                    # a job is pending or running
status_payload: + "indexState", "indexGeneration", "embedding": {"model", "prefixes", "dim", "backend"},
                "chunkBudget", "rebuildReasons" (pending only); when unavailable: "indexState" + "indexError"
```
No per-result index state here (plan 3's `with_index`). Open, in this order:
1. The configured embedder: `embedder_factory(configured_scheme(config))`; `configured()` = `EmbeddingScheme(e.model_name, e.dim, e.prefixes)` and `ChunkingScheme(SCHEME_VERSION, config.token_budget)`. An unknown configured model raises `UnknownModelError` as today.
2. `recover_swap(db)`; `read_identity(db)`; `IdentityError` -> unavailable. None: a legacy index -> `legacy_error(db, config.roots)`, nothing written; otherwise write a fresh identity: configured schemes, generation 1, `probe=()`, `config.fts_language`, `document_table=True`.
3. The embedder for the RECORDED scheme: the configured embedder when the schemes are equal, else `embedder_factory(recorded)`; a failure to build it (`UnknownModelError`) or a width other than the recorded `dim` -> unavailable, naming the model. Same verdict as `index_state`.
4. `Store(db, dim=recorded.dim, fts_language=config.fts_language)`; a changed `fts_language` is today's in-place FTS rebuild, and `index.json`'s `fts_language` is rewritten (best effort, warn on `OSError`).

- [ ] **Step 1: Write the failing tests**

```python
def test_a_fresh_folder_is_stamped_with_the_configured_scheme(tmp_path, fake_embedder):
    h = handle(tmp_path, fake_embedder); h.get()
    ident = read_identity(h.config.db_path)
    assert (ident.generation, ident.embedding, ident.probe) == (1, EmbeddingScheme("fake", 8, "none"), ())
    assert ident.chunking == ChunkingScheme(SCHEME_VERSION, h.config.token_budget) and h.state() == (IndexState.READY, "")

def test_a_1x_index_is_refused_and_left_untouched(tmp_path, fake_embedder):
    db = tmp_path / "db"; Store(db, dim=8).replace_source("/a.md", [rec("/a.md", 0, "x")])
    before = sorted(p.relative_to(db) for p in db.rglob("*"))
    h = handle(tmp_path, fake_embedder, DB_PATH=str(db))
    with pytest.raises(IndexUnavailableError, match=re.escape(str(default_db_path(h.config.roots)))): h.get()
    assert h.state()[0] is IndexState.UNAVAILABLE
    assert sorted(p.relative_to(db) for p in db.rglob("*")) == before

async def test_both_shells_refuse_a_1x_index_naming_the_v2_path(...)     # MCP ToolError and CLI exit 1, same message
def test_an_unreadable_identity_is_unavailable_and_never_overwritten(...)
def test_a_changed_model_is_pending_and_queries_use_the_recorded_model(tmp_path):
    models = {MINILM: FakeEmbedder(MINILM, salt="a"), E5_LARGE: FakeEmbedder(E5_LARGE, salt="b")}
    factory = lambda s: models[s.model]
    a = IndexHandle(cfg(tmp_path, MODEL_NAME=MINILM), embedder_factory=factory)
    a.get().pipeline.ingest_data("budget review", source="s")
    b = IndexHandle(cfg(tmp_path), embedder_factory=factory)                    # default model: e5
    state, why = b.state()
    assert state is IndexState.REBUILD_PENDING and f"model {MINILM} -> {E5_LARGE}" in why
    assert b.get().embedder is models[MINILM]
    assert run_query(b.get().store, b.get().embedder, b.config, "budget review")["sources"]
def test_a_pending_index_embeds_through_the_recorded_models_endpoint(tmp_path, fake_llama, tokenizer_file):
    # recorded E5_LARGE, configured MINILM, embed_endpoints={E5_LARGE: fake_llama}: get().embedder.backend == "endpoint";
    # a query reaches fake_llama with "query: " (the recorded prefixes)
def test_a_changed_budget_is_pending_with_a_chunking_reason(...)
def test_a_changed_fts_language_is_rebuilt_in_place_and_not_pending(...)
def test_a_recorded_model_that_cannot_be_built_is_unavailable(...)
async def test_status_reports_the_identity_the_same_in_both_shells(...)  # indexState, indexGeneration, embedding, chunkBudget
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_handle.py tests/test_parity.py -q`; expected: no `index.json` written, `AttributeError: 'IndexHandle' object has no attribute 'state'`.

- [ ] **Step 3: Implement** as above. Then run the whole suite and stamp the tests that seed an index before a shell opens it (they now fail as legacy).

- [ ] **Step 4: Run the tests and the gate** — expected: all pass.

- [ ] **Step 5: Commit** — `feat(core): an index is searched by its recorded scheme; 1.x indexes are refused (g7q.1)`

---

### Task 10: The served model is verified by a probe vector

**Files:**
- Modify: `src/minirag_mcp/embedder.py`, `src/minirag_mcp/handle.py`, `src/minirag_mcp/ingest/pipeline.py` (`before_store`), `tests/conftest.py` (`FakeEmbedder.probe_vector()` = `_vec(PROBE_TEXT)`, `expect_probe` stores, `verify()` no-op, `probe_error = None`)
- Test: `tests/test_embedder.py`, `tests/test_handle.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 4 `PROBE_TEXT`, `PROBE_COSINE`, `write_identity`; Task 3 fallback and pickling.
- Produces (the handle builds its pipeline with `before_store=self.ensure_probe`; plan 2's parallel sync writes through `Store.replace_sources`, not `replace_source`, so it calls `pipeline.before_store()` once before it starts its workers, which then inherit the armed expectation by pickle):
```python
class ProbeMismatchError(EmbedderUnavailableError): ...
Embedder.probe_vector(self) -> list[float]            # PROBE_TEXT embedded as a document, prefix included, by the live backend
Embedder.expect_probe(self, probe: Sequence[float]) -> None   # each backend is checked before its first embedding; kept across pickling
Embedder.verify(self) -> None                         # checks a configured, reachable endpoint now; never loads the in-process model
Embedder.probe_error -> str | None
Pipeline(store, embedder, config, labels=None, *, before_store: Callable[[], None] | None = None)  # called before every replace_source
Pipeline.before_store: Callable[[], None] | None   # the constructor argument, kept as a public attribute
IndexHandle.ensure_probe(self) -> None                # records the probe when index.json has none, then expect_probe
```
A live probe with `cosine(live, recorded) < PROBE_COSINE` raises `ProbeMismatchError`: `f"{served} does not serve {recorded_model}: its vector for the probe text differs from the one recorded when this index was built (cosine {c:.4f} < 0.999). Point it at {recorded_model}; until then this index is unavailable."`, where `served` is `f"The endpoint {url} (router model {name!r})"` or `f"fastembed's {model}"`. The check runs before the first embedding through a backend, after a fall-back and when the endpoint comes back. The handle calls `verify()` at open when the backend is the endpoint and a probe is recorded; a `probe_error` makes `state()` UNAVAILABLE and `get()` raise `IndexUnavailableError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_first_ingest_records_the_probe(tmp_path, fake_embedder):
    h = handle(tmp_path, fake_embedder); h.get().pipeline.ingest_data("x", source="s")
    assert read_identity(h.config.db_path).probe == pytest.approx(fake_embedder._vec(PROBE_TEXT))
def test_a_matching_endpoint_is_ready(fake_llama, tmp_path, tokenizer_file)
def test_an_endpoint_serving_another_model_is_unavailable_naming_both(fake_llama, tmp_path, tokenizer_file):
    # index built through fake_llama (probe recorded); fake_llama.vector_salt = "other"; a new handle:
    state, why = h.state()
    assert state is IndexState.UNAVAILABLE and E5_LARGE in why and "'e5'" in why
    with pytest.raises(IndexUnavailableError): h.get()
def test_the_probe_is_checked_again_when_the_endpoint_comes_back(...)  # fallback (stub matches) -> endpoint back with another salt -> ProbeMismatchError, probe_error set
def test_a_fallback_model_is_verified_before_use(...)                  # stub with other vectors -> ProbeMismatchError
def test_the_expected_probe_survives_pickling(...)                     # clone of a mismatching embedder still raises ProbeMismatchError
def test_the_store_hook_runs_before_every_write(tmp_path, fake_embedder)  # counts calls: ingest_data twice -> 2
```

- [ ] **Step 2: Run to verify they fail** — expected: `AttributeError: ... 'probe_vector'`.

- [ ] **Step 3: Implement** as specified; `ensure_probe` writes with `write_identity(replace(identity, probe=tuple(vector)))`.

- [ ] **Step 4: Run the tests and the gate** — expected: all pass.

- [ ] **Step 5: Commit** — `feat(core): verify the served embedding model with a recorded probe vector`

---

### Task 11: Rebuild into `<db>.next`, swap, and pick up a new generation

**Files:**
- Create: `src/minirag_mcp/rebuild.py`
- Modify: `src/minirag_mcp/store.py` (`copy_reembedded`, `IndexBusyError` guard), `src/minirag_mcp/ingest/pipeline.py` (`reingest_stored`), `src/minirag_mcp/sync.py` (`SyncManager(run=)`, `SyncJob.rebuild`), `src/minirag_mcp/handle.py` (`run_sync`, generation pickup), `src/minirag_mcp/server.py` (`sync_start`/`sync_status` docstrings), `src/minirag_mcp/cli/__init__.py` (`sync` via `handle.run_sync`, prints the rebuild)
- Test: `tests/test_rebuild.py`, `tests/test_handle.py`, `tests/test_parity.py`

**Interfaces:**
- Consumes: Tasks 4, 7, 9, 10.
- Produces:
```python
# rebuild.py
def rebuild(current: OpenIndex, make: EmbedderFactory,
            on_event: Callable[[str], None] | None) -> tuple[dict[str, int], list[dict], RebuildInfo]  # caller holds the sync lock
# store.py
class IndexBusyError(Exception): ...
Store.copy_reembedded(self, source: Store, embed: Callable[[list[str]], list[list[float]]], *,
                      batch_rows: int = 1024, on_event: Callable[[str], None] | None = None) -> int
# pipeline
Pipeline.reingest_stored(self, info: SourceInfo, markdown: str) -> IngestResult
# sync.py
SyncRun = Callable[[Path | None], tuple[dict[str, int], list[dict], RebuildInfo | None]]
SyncManager(pipeline, store, config, *, run: SyncRun | None = None)
SyncJob.rebuild: RebuildInfo | None = None        # to_dict()["rebuild"]: dict | None
# handle.py
IndexHandle.run_sync(self, scope: Path | None, on_event=None) -> tuple[dict[str, int], list[dict], RebuildInfo | None]
```
`rebuild`:
1. Remove a leftover `staging_dir(db)`.
2. `new = make(configured_scheme(config))`; write the staging identity: `EmbeddingScheme(new.model_name, new.dim, new.prefixes)`, the configured chunking, `probe = new.probe_vector()`, the current generation; `new.expect_probe(probe)`.
3. `target = Store(staging, dim=new.dim, fts_language=config.fts_language)`; `Pipeline(target, new, config, labels=current.labels)` without `before_store`.
4. `"reembed"`: `target.copy_reembedded(current.store, new.embed_documents)` copies every chunk column but the vector, which is the stored text re-embedded, plus the document rows (through `_write_documents`) and signatures as they are. Labels are untouched; nothing is parsed. `"reingest"`: nothing copied.
5. `_run_sync_unlocked(pipeline, target, roots, max_file_size, None, on_event)` reconciles with the disk (for `"reingest"` that reads every file).
6. Every source of `current.store` missing from `target` that is not a file, or is a file still on disk and not excluded by a rule, is re-chunked from `join_document(current.store.all_chunks(source))` via `reingest_stored`; a file among them adds the error `"kept from its stored text: this installation could not read it again"`. A file gone from disk is not restored.
7. `target.update_indices()`; `swap_in(db)`; return `RebuildInfo(kind, tuple(differences), new_generation, reembedded, restored)`.
A failure before `swap_in` propagates; the old index keeps serving.

`handle.run_sync` holds `sync_lock(db)` and first removes a leftover `staging_dir(db)` (only a rebuild holding this lock writes there, so one found now is debris of a crashed rebuild; left in place, `rebuild_running` would be true for the lock this sync holds and refuse its own writes): REBUILD_PENDING and no scope -> `rebuild`, then reopen; REBUILD_PENDING with a scope -> today's sync of the scope plus the error `{"source": str(db), "error": "rebuild: an unscoped sync rebuilds this index (<differences>); this one synced its scope under the recorded scheme"}`; otherwise today's sync. The handle's `SyncManager` gets `run=` the same logic without taking the lock (the manager already holds it). Generation pickup in `get()`: stat `index.json` (inode, mtime_ns); on change read it, and a different generation reopens everything but the labels source and the sync manager; a missing or unreadable file mid-swap keeps the open index until the next call. `Store.replace_source`, `move_source`, `delete_source`, `set_document_labels` first raise `IndexBusyError` when `rebuild_running(self.db_path)`: `f"{db} is being rebuilt into {staging}; writes to it are refused until the rebuild finishes, since the rebuilt index replaces it. Retry once sync_status reports the sync finished."`

- [ ] **Step 1: Write the failing tests**

```python
def test_an_unscoped_sync_reembeds_on_a_model_change_without_reading_files(corpus, factory, parses, model_factory):
    # built with MINILM; a handle configured E5_LARGE (FakeEmbedders keyed by model):
    counts, errors, info = b.run_sync(None)
    assert (info.kind, info.generation, info.reembedded) == ("reembed", 2, chunks_before)
    assert parses.count == 0 and model_factory.calls == []
    assert stored_vectors(b) == models[E5_LARGE].embed_documents(stored_texts(b))
    assert read_identity(db).embedding.model == E5_LARGE and read_identity(previous_dir(db)).embedding.model == MINILM
    assert b.state() == (IndexState.READY, "")
def test_reembedding_keeps_document_rows_signatures_and_labels(...)
def test_a_new_vector_width_is_rebuilt_at_that_width(...)          # the E5_LARGE fake with dim 16
def test_a_budget_change_reingests_from_disk(...)                   # kind "reingest", parses > 0, recorded budget changed
def test_a_reingest_keeps_what_the_disk_cannot_give_back(...)
    # an ingest_data source and a file whose parser now raises are restored (restored == 2, one error);
    # a file deleted from disk is not
def test_a_scoped_sync_on_a_pending_index_is_refused_and_writes_nothing(...)   # chunk count and generation unchanged; error names the unscoped sync
def test_the_old_index_answers_until_the_swap(...)                 # a query from inside the new embedder's hook sees generation 1 and the old results
def test_a_failed_rebuild_leaves_the_old_index_serving(...)        # the new embed raises mid-copy -> run_sync raises; generation 1, still pending; the next run_sync succeeds
def test_writes_are_refused_while_a_rebuild_runs(...)              # staging dir + lock held elsewhere -> ingest_data raises IndexBusyError; released -> works
def test_a_leftover_staging_dir_never_blocks_a_sync(...)            # <db>.next left by a crash, READY index: run_sync ingests a new file, staging dir gone
def test_a_swap_by_another_process_is_picked_up_before_the_next_request(...)  # handle S open; handle C rebuilds; S.get() sees generation 2 and the new embedder
def test_an_interrupted_swap_is_finished_on_open(...)              # Review Focus 3: db missing, previous + staging present -> get() serves the staged index
async def test_both_shells_report_the_rebuild(...)                  # CLI `sync --json`["rebuild"] == MCP sync_status["rebuild"] on equal setups
```

- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_rebuild.py -q`; expected: `ModuleNotFoundError: minirag_mcp.rebuild`.

- [ ] **Step 3: Implement** as specified. CLI `sync --json` prints `{"counts", "errors", "rebuild"}`; the human line appends `; rebuilt (<kind>): <reasons>`. `sync_status`'s docstring describes `rebuild` and that only an unscoped sync rebuilds.

- [ ] **Step 4: Run the tests and the gate** — expected: all pass.

- [ ] **Step 5: Commit** — `feat(core): rebuild into <db>.next on an explicit sync and swap atomically (g7q.1)`

---

### Task 12: The English stand measures through the production path

**Files:**
- Modify: `evals/enterpriserag/common.py`, `evals/enterpriserag/bulk_index.py`, `evals/enterpriserag/recall.py`, `evals/enterpriserag/test_common.py`, `evals/enterpriserag/README.md`

**Interfaces:**
- Consumes: `make_embedder`, `configured_scheme` (Task 5), `RerankEndpoint` (Task 1), `run_query(..., reranker=)` (Task 8), `Store.backfill_documents` (Task 7).
- Produces: `common.CORE_ENV: tuple[str, ...]`, `common.core_env(**fixed: str) -> dict[str, str]`; `recall.py --subset --budget N`.

- [ ] **Step 1: Write the failing test** (`evals/enterpriserag/test_common.py`; delete `test_e5_prefixes_wrap_documents_and_queries`)

```python
def test_core_env_passes_through_only_core_variables(monkeypatch):
    monkeypatch.setenv("EMBED_URL", "http://127.0.0.1:8080/v1/embeddings"); monkeypatch.setenv("HOME", "/x")
    env = core_env(BASE_DIR="/c", DB_PATH="/d")
    assert env == {"BASE_DIR": "/c", "DB_PATH": "/d", "EMBED_URL": "http://127.0.0.1:8080/v1/embeddings"}
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest evals/enterpriserag -q`; expected: `ImportError: cannot import name 'core_env'`.

- [ ] **Step 3: Implement**

- `common.py`: delete `apply_e5_prefixes` and `apply_embedding_endpoint` (prefixes and endpoints are native; the patches would now prefix twice). `CORE_ENV = ("MODEL_NAME", "CHUNK_TOKEN_BUDGET", "RAG_HYBRID_WEIGHT", "RAG_DOCUMENT_WEIGHT", "RAG_RERANK_TOP", "RAG_FTS_LANGUAGE", "EMBED_URL", "EMBED_ENDPOINT_MODEL", "RERANK_URL", "RERANK_MODEL", "ENDPOINT_API_KEY_FILE")`.
- `bulk_index.py`: `_env` -> `core_env(BASE_DIR=..., DB_PATH=...)`; workers build `make_embedder(cfg, configured_scheme(cfg), threads=threads)` (the `functools.partial` on `TextEmbedding` goes); after the last flush `store.backfill_documents()` before `update_indices()`. `--verify` unchanged.
- `recall.py`: `_open` builds the config from `core_env` and also returns `RerankEndpoint(cfg.rerank_endpoint)` or None, which `measure` and `_deep` pass on; `--budget N` (subset only) sets `CHUNK_TOKEN_BUDGET` for the build and the measurement; `--model` and `--e5-prefixes` go (use `MODEL_NAME`); a run is an experiment when any `CORE_ENV` variable or `--budget` is set; tag `f"{model}-b{budget}-dw{document_weight}" + "-rr" (reranker) + "-endpoint" (EMBED_URL)`; `fingerprint` adds `token_budget`, `document_weight`, `rerank_model`.
- README: the experiment commands use `EMBED_URL`/`EMBED_ENDPOINT_MODEL`/`ENDPOINT_API_KEY_FILE`/`RERANK_URL`/`RERANK_MODEL` instead of `ERB_*`, plus two lines on `--budget`. `baseline.json` is not re-taken here (spec section 6: after the joint check of the final pipeline).

- [ ] **Step 4: Run** — `uv run pytest evals/enterpriserag -q` and the main gate; `uv run python evals/enterpriserag/recall.py --help` lists `--budget`. Expected: all pass.

- [ ] **Step 5: Commit** — `feat(evals): the English stand runs the production retrieval path (endpoints, document BM25, reranker)`

---

### Task 13: Measure and set the e5 default chunk budget

> **Widened (owner, 2026-11-19):** also measure `BAAI/bge-m3` as a candidate default model
> before the re-index (a model change after migration would mean re-indexing twice). Serve it
> through llama.cpp (GGUF converted from the official weights at the router's commit, as e5
> and the reranker were; verify against the reference implementation), embed the private
> ruler's chunks at the winning budget, and compare with e5 on both fields and on the English
> 20k subset, with document BM25 and the reranker on. Also try bge-m3 whole-document vectors
> (one vector per document up to 8192 tokens) as an extra fusion signal. Decision rule: switch
> the default only on a significant gain on the Russian ruler (sign test p < 0.05 on hit@10 or
> bootstrap CI of MRR excluding 0) without a loss on English; then add it to MODEL_WINDOWS /
> DEFAULT_BUDGETS, check fastembed supports it for the in-process fallback (if not, record that
> bge-m3 requires an endpoint), and record all numbers in bd g7q.1.

**Files:**
- Modify: `src/minirag_mcp/embedder.py` (`DEFAULT_BUDGETS[E5_LARGE]`), `tests/test_config.py`
- Outside the repository (private): `<ruler>/scripts/v2_measure.py`, its indexes and runs under `<ruler>`, the private Russian ruler (its location is in the bd memory about the ruler: `bd memories ruler`)
- Record: `bd update minirag-mcp-g7q.1 --append-notes "..."`, `bd update minirag-mcp-g7q.3 --append-notes "..."`

**Interfaces:**
- Consumes: everything above, through the CLI and `IndexHandle`.
- Produces: the measured `DEFAULT_BUDGETS[E5_LARGE]`; numbers in bd notes only.

- [ ] **Step 1: Live check against the owner's router.** With e5, `bge-reranker-v2-m3` and the classify model served, export `EMBED_URL`, `EMBED_ENDPOINT_MODEL=multilingual-e5-large`, `RERANK_URL`, `RERANK_MODEL=bge-reranker-v2-m3`, `ENDPOINT_API_KEY_FILE` (the router's key file). Over a scratch copy of three private documents (in the scratchpad, never the repo): `status --json` shows `embeddingBackend: "endpoint"`; a `query` shows `reranked: true`; `classify try` on each document with and without `CLASSIFY_URL`/`CLASSIFY_MODEL` gives the same choice per facet (the spike left the router `model` field, `/props?model=` and the Bearer key untested). A failure is fixed in the core with a test first. Append the outcome (counts only) to g7q.3.

- [ ] **Step 2: English, three budgets.** For N in 110, 250, 400: `uv run python evals/enterpriserag/recall.py --subset --budget N` with the Step 1 environment. Read overall `recall@10`, `mrr`, `recall_default_call`, chunk count and build time from `evals/enterpriserag/runs/subset_<tag>.json` (gitignored).

- [ ] **Step 3: Russian ruler, three budgets.** `v2_measure.py` (private, modelled on the ruler's `measure.py`): for each N, `minirag-mcp sync` with `BASE_DIRS=[<corpus root from the ruler README>]`, `DB_PATH=<ruler>/v2-b<N>`, `RAG_FTS_LANGUAGE=Russian`, `CHUNK_TOKEN_BUDGET=N` and the Step 1 environment; then for every gold question, both fields (`query`, `question`), `run_query` through `IndexHandle(load_config(env)).get()` with its reranker at `top_k=100` and `top_k=8`, rows in `measure.py`'s format to `runs/v2-b<N>.json`, plus the median query time.

- [ ] **Step 4: Judge and score.** `pool.py` over the three runs; judge any unjudged pairs as earlier rounds did (`judge/BRIEF.md`, verdicts under `judge/`); then `score.py v2-b110 --field query`, `--field question`, and `--raw`.

- [ ] **Step 5: Decide.** Drop a budget whose English `recall@10` is more than 1.0 point, or MRR more than 0.02, below the best English budget (the regression gate's thresholds). Of the rest, take the highest Russian MRR on the `query` field; if a smaller budget is within 0.01 of it, take the smaller (smaller reranker inputs and citations).

- [ ] **Step 6: Record.** `bd update minirag-mcp-g7q.1 --append-notes` with one row per budget: Russian query MRR / default-window hit / hit@10 (judged and raw), question MRR, English recall@10 / MRR / default-call recall, chunk count, index time, median query time with the reranker, and the choice with its reason. Note that the Russian corpus was read with today's OCR (plan 2 changes it). No document names, questions or paths.

- [ ] **Step 7: Set the default, test first**

```python
def test_e5_default_budget_is_the_measured_one(tmp_path):
    assert load_config({}, cwd=tmp_path).token_budget == <chosen N>    # g7q.1 notes, 2026-11
```
Run it (FAIL unless N is 110), set `DEFAULT_BUDGETS[E5_LARGE] = N`, run the gate (`uv run pytest -q`, ruff check, ruff format --check) and `uv run pytest -m slow -q`: all pass.

- [ ] **Step 8: Commit** — `feat(config): e5 default chunk budget <N> tokens, measured on both stands (g7q.1)`
