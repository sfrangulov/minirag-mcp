# minirag v2 — cross-plan contract

Spec: `docs/superpowers/specs/2026-11-19-minirag-v2-design.md`. This file is the index of the
interfaces one plan defines and another consumes, as the plans finally agreed them. Each name
is defined, with its signature and tests, in the task named; when this file and a plan
disagree, the plan's Interfaces block wins and this file is the one to fix.

Plans, executed in this order (each leaves `uv run pytest -q`, `ruff check src tests` and
`ruff format --check src tests` green):

1. `-01-core.md` — endpoints, embedder, index identity and rebuilds, `IndexHandle`, document
   BM25, reranking, classify endpoint, v2 config, chunk-budget measurement. g7q.1, .3, .4, .7.
2. `-02-ingest.md` — struck-through text (kept, marked `~~...~~` across chunk splits),
   Tesseract OCR, parallel sync. g7q.6, .9, .12 (that item), .14.
3. `-03-server.md` — `indexes.toml`, the registry, `serve`, `index` argument, `list_indexes`,
   `connect`, CLI `--index`, instructions, LaunchAgent template. g7q.2.
4. `-04-release.md` — release mechanics, README, joint check and baseline (g7q.5), `2.0.0a1`,
   the owner's migration (g7q.13).

## Who owns what

| Module | Plan, task | Names other plans use |
|---|---|---|
| `endpoints.py` | 1.1 | `Endpoint(url, name, api_key=None, timeout=120.0)`, `EndpointError`, `read_api_key(path)`, `EmbeddingEndpoint.embed`, `RerankEndpoint.scores`, `.last_error` |
| `config.py` | 1.2; 2.3, 2.6 | `DEFAULT_MODEL` (e5-large), `default_db_path(roots)` = `<first root>/.minirag/v2`; `Config` + `document_weight`, `rerank_top`, `embed_endpoints: Mapping[str, Endpoint]` (fastembed id -> endpoint), `rerank_endpoint`, `classify_endpoint`, `index_name` (plan 1); `sync_workers`, `ocr_lang` in Tesseract spelling (plan 2) |
| `embedder.py` | 1.2, 1.3, 1.10 | `Embedder(model_name, cache_dir, *, endpoint=None, prefixes=None, threads=None, allow_fallback=True)`: picklable, public `threads`, `allow_fallback`, `prefixes`, `backend`, `warning`, `max_tokens`, `prefix_tokens`, `expect_probe`, `probe_vector`, `verify`, `probe_error`; `known_model`, `prefix_rule`, `model_window`, `default_budget` |
| `identity.py` | 1.4 | `IndexIdentity`, `EmbeddingScheme`, `ChunkingScheme`, `IndexState` (`ready`, `rebuild pending`, `syncing`, `unavailable`), `read_identity`, `index_state`, `differences`, `rebuild_kind`, `staging_dir`, `previous_dir`, `swap_in`, `recover_swap`, `rebuild_running`, `RebuildInfo` |
| `handle.py` | 1.5, 1.9-1.11 | `make_embedder(config, scheme, *, threads=None)`, `configured_scheme(config)`, `EmbedderFactory`, `OpenIndex(config, store, embedder, pipeline, labels, identity, reranker)`, `IndexHandle(config, *, embedder_factory, model_factory)` with `get()`, `state()`, `.sync`, `run_sync(scope, on_event)`, `ensure_probe()`, `notes()`; `IndexUnavailableError` |
| `rebuild.py` | 1.11 | `rebuild(current, make, on_event)` |
| `store.py` | 1.7, 1.8, 1.11; 2.5 | `documents` table; `_document_row`, `_write_documents`, `_delete_documents`, `backfill_documents`; `search(..., document_weight, reranker, rerank_top)`; `copy_reembedded`; `IndexBusyError`; `replace_sources(batch, *, indexed)` (plan 2) |
| `ingest/pipeline.py` | 1.3, 1.10, 1.11; 2.4 | `Pipeline(store, embedder, config, labels=None, *, before_store=None)`, `.before_store`, `reingest_stored`; `PreparedDocument` (no `embedding_warning`: workers never fall back, ruling C10), `prepare`, `prepare_file`, `records`, `store_prepared` (plan 2) |
| `sync.py` | 1.3, 1.7, 1.9, 1.11; 2.9 | `_run_sync_unlocked(pipeline, store, roots, max_file_size, scope, on_event)`, `SyncManager(db_path, run)`, `.running`, `SyncJob.rebuild` |
| `results.py` | 1.3-1.9; 3.4 | `embedding_status`, `embedder_notes`, `ingest_reply(r, notes)`, `rerank_status`, `documents_status`, `status_payload(config, handle)` (plan 1); `with_index`, `index_status`, `registry_status`, `list_indexes_payload` (plan 3) |
| `labels/endpoint_model.py` | 1.6 | `EndpointChoiceModel`, `endpoint_model_factory`; `LabelerSource.backend` |
| `parallel.py` | 2.6-2.8 | `prepare_files`, `worker_pipeline`, `BatchWriter`, `default_workers`, `worker_count`, `physical_cores`, `total_memory`, `worker_footprint`, `worker_threads` (sync.py only) |
| `indexes.py` | 3.1 | `ServerConfig`, `load_server_config(path, env)`, `default_config_path(env=None)`, `INDEX_KEYS` |
| `registry.py` | 3.2, 3.3 | `Connection`, `UNRESTRICTED`, `Use`, `IndexRegistry` (`from_file`, `single`, `open`, `resolve`, `refresh`, `reload`), `IndexEntry` (`.handle`, `runtime()`, `state()`), `EmbedderPool`, `IndexRefused`, `IndexUnavailable` |
| `http_server.py`, `bridge.py` | 3.7, 3.8 | `serve`, `ensure_token`, `parse_connection`; `run_bridge` |
| `docs/launchd/local.minirag-mcp.plist` | 3.9 | label `local.minirag-mcp`, placeholders `__UVX__`, `__VERSION__`, `__HOME__` (plan 4 copies it) |

## Rules the plans agreed

- **Opening an index.** Only `IndexHandle` builds an index's embedder, store, labels,
  pipeline, sync manager and reranker, checks `index.json` and follows a new generation.
  Both shells, single-index mode included, reach it through plan 3's registry
  (`IndexEntry.handle`); the registry keeps no `OpenIndex` of its own, so every call goes
  through `handle.get()`.
- **Building embedders.** `make_embedder(config, scheme)` is the one constructor; it picks
  `config.embed_endpoints[scheme.model]` when present. Plan 3's `EmbedderPool` memoizes it on
  `(scheme, cache_dir, endpoint)` so indexes of one model share one embedder in `serve`. Sync
  workers never build one: they get the pipeline's embedder by pickle, so they embed with the
  recorded scheme and carry its expected probe.
- **The probe.** The handle passes each generation's pipeline a `_BeforeStore` as
  `before_store` (`IndexHandle.ensure_probe()` calls the open one); `store_prepared` calls
  it before each `replace_source`; the parallel sync calls `pipeline.before_store()` once
  before starting workers. A rebuild arms its new embedder itself.
- **Batched writes.** `Store.replace_sources` writes chunk rows in one commit, document rows
  with one `_write_documents([_document_row(...) ...])`, signatures with one
  `_put_signatures`, and raises `IndexBusyError` while a rebuild runs, like `replace_source`.
- **Sync and rebuild.** Only an unscoped sync rebuilds a `rebuild pending` index
  (`IndexHandle.run_sync` / the manager's `run`); it re-ingests through `_run_sync_unlocked`
  on `<db>.next`, so it runs in workers too. A sync first removes a leftover `<db>.next`.
  A scoped sync on a pending index is refused (only the unscoped rebuild writes there).
  Workers never fall back in-process: an unreachable endpoint fails the file, and 3
  consecutive failures stop dispatch with one loud error; the serial path keeps the fallback.
  `parallel.prepare_files` yields each file as it finishes, not in input order (2026-11-20,
  ruling I1).
- **Status and result flags.** Plan 1's `status_payload(config, handle)` is the per-index
  status (`indexState`, `indexGeneration`, `embedding`, `embeddingBackend`, `reranker`,
  `chunkBudget`, `rebuildReasons`, `indexError`, labels `backend`); plan 3's `index_status`
  adds only `index`, and `registry_status` adds `configError`. Per result: plan 1 owns
  `embeddingBackend`, `reranked`, `rerankScore`, `warnings`; plan 3 owns `index`,
  `indexState`, `indexWarning`. While a rebuild is pending: reads answer with the warning,
  `sync_start` and `delete_file` are allowed, `ingest_*` are refused.
- **`indexes.toml` keys.** `[index.NAME]` keys map onto `load_config`'s variables
  (`model`, `db`, `fts_language`, `token_budget`, `hybrid_weight`, `document_weight`,
  `rerank_top`, `grouping`, `max_distance`, `max_files`, `max_file_size`,
  `allow_private_urls`, `ocr_lang`, `ocr_min_chars_per_page`, `sync_workers`); `[server]
  cache_dir`, `api_key_file`, `[rerank]` and `[classify]` feed the endpoint variables; every
  `[embedding."<id>"]` table becomes one entry of every index's `embed_endpoints`.
- **Registry.** `publish-mcp.yml` registers only final `vMAJOR.MINOR.PATCH` tags.

## Rules every plan follows

- CLAUDE.md applies: TDD; each fix's test must fail on the unfixed code; MCP and CLI stay
  identical; English everywhere; comments state constraints only.
- Tests use the fakes in `tests/conftest.py`; endpoint tests use plan 1's `fake_llama`
  (stdlib `http.server` in a thread), never the network.
- No private data in the repository: no corpus or client names, no machine paths, no corpus
  text. The private Russian ruler lives outside the repo (its location is in the bd memory
  about the ruler); only aggregate numbers go into issues.
- Commit after each task, conventional-commit style, ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
