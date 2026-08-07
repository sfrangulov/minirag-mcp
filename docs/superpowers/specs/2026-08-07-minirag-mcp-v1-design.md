# minirag-mcp v1 — Design

- **Date:** 2026-08-07
- **Status:** Approved (brainstorm with user, this session)
- **Reference:** [shinpr/mcp-local-rag](https://github.com/shinpr/mcp-local-rag) (TypeScript)
- **Tracking:** beads `minirag-mcp-jpw` (design), `minirag-mcp-01m` (plan); `minirag-mcp-l7s` closed — `ingest_url` folded into v1

## Overview

minirag-mcp is a local-first RAG MCP server in Python — a functional analog of
mcp-local-rag. It indexes local documents, searches them with hybrid retrieval
(semantic vector search + keyword boost for exact technical terms), and sends
nothing to external services after the embedding model is downloaded.

Differences from the reference:

- **markitdown** is the single conversion path for everything — files, HTML,
  and URLs (PDF, DOCX, PPTX, XLSX, HTML, CSV, EPub, ipynb, …). No separate
  readability layer: markitdown's HtmlConverter converts the whole `<body>`
  without boilerplate removal — accepted trade-off (user decision 2026-08-07);
  revisit with a readability pass only if index noise hurts in practice.
- **Two tools beyond parity**: `ingest_url` — the server fetches a page
  itself via markitdown's `convert_url` (YouTube/Wikipedia/RSS get special
  handling for free); `read_file` — return a source's full indexed Markdown
  (chunks joined in order; original file bytes are not stored).
- **Multilingual embedding model by default** (Russian + English corpora work
  out of the box).
- **Smarter default storage**: index lives next to the documents, model cache
  is global (see Configuration).
- Python stack: fastmcp, fastembed, LanceDB Python bindings.

All documentation, tool descriptions, and code are in English.

## Goals

1. Tool-for-tool parity with the reference's 9 MCP tools and CLI commands,
   plus `ingest_url` and `read_file` (11 tools total).
2. Same environment-variable contract (names) where applicable; defaults
   deviate deliberately for `DB_PATH`/`CACHE_DIR` (see Configuration).
3. Local-first: file/data ingestion and search never touch the network after
   the one-time model download. `ingest_url` is the only network operation and
   runs only on explicit request — documentation must state this clearly.
4. Distributable as a PyPI package runnable via `uvx minirag-mcp`.

## Non-goals (deferred)

- PDF visual mode (vision-model captions for figures).
- `RAG_DEVICE` / `RAG_DTYPE` (fastembed manages ONNX providers itself).
- Multi-writer coordination. Single writer per `DB_PATH`, as in the reference.

## Stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python ≥ 3.11 | markitdown needs ≥ 3.10 |
| Package/dev manager | uv | `uv build`, `uv run pytest` |
| MCP framework | fastmcp (v3) | stdio transport |
| File/HTML/URL → Markdown | markitdown + extras `[pdf, docx, pptx, xlsx]` | everything becomes Markdown; `convert_url` for `ingest_url` |
| Paths | platformdirs | global model cache location |
| Embeddings | fastembed | ONNX, no torch; default model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~220 MB quantized), override via `MODEL_NAME` |
| Store | LanceDB (embedded) | vectors + BM25 FTS + native hybrid search |
| CLI | cyclopts | already a transitive dependency of fastmcp (verified: `fastmcp-slim[server]` requires `cyclopts>=4.0.0`) — zero added deps |
| Tests | pytest | fake embedder via DI for fast tests |

Verified facts (2026-08-07, sources checked live):

- `defuddle` does not exist on PyPI; `pydefuddle` 0.1.0 is a one-release port
  (rejected as a core dependency). A readability layer was considered
  (trafilatura) and dropped by user decision — markitdown only.
- markitdown `HtmlConverter` converts the whole `<body>` (only strips
  `<script>`/`<style>`) — no boilerplate removal; accepted trade-off.
- markitdown has `convert_url()`/`convert_uri()` out of the box; `convert_uri`
  also accepts `file:`/`data:` URIs, hence the scheme whitelist in Security.
- fastembed's registry id for the default model is
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dim=384,
  0.22 GB, quantized ONNX weights from `qdrant/…-onnx-Q`).
- fastmcp v3 depends on `cyclopts` (CLI framework) via `fastmcp-slim[server]`,
  so the CLI uses cyclopts at no dependency cost.

## Entry point

One console script, `minirag-mcp` (parity with `npx mcp-local-rag`):

- **No arguments** → MCP server on stdio.
- **Subcommand** → CLI action executed in-process against the same index,
  without starting the server.

## Modules

```
src/minirag_mcp/
  server.py       — FastMCP app; 9 thin tool wrappers
  config.py       — env + CLI-flag resolution (flags > env > defaults)
  security.py     — root containment checks, symlink-escape rejection
  ingest/
    parser.py     — markitdown wrapper: file/HTML/URL → Markdown + title
    pipeline.py   — parse → chunk → embed → store (replace by source)
    scanner.py    — recursive root walk, extension whitelist, sha256 diff
  chunker/
    structural.py — Markdown structure split; code fences atomic
    semantic.py   — embedding-similarity merge of adjacent small chunks
  embedder.py     — fastembed wrapper; model cache in CACHE_DIR
  store.py        — LanceDB: schema, FTS index, hybrid search, upsert/delete,
                    neighbors, distinct-source listing
  sync.py         — single in-memory sync job, background thread
  cli/            — command parser + one module per subcommand (thin wrappers)
```

Each module is independently testable; `server.py` and `cli/` share the same
core (`pipeline`, `store`, `embedder`) and contain no business logic.

## MCP tools (11 = 9 parity + ingest_url + read_file)

| Tool | Input | Output |
|---|---|---|
| `sync_start` | `path?` (absolute, inside a root) | `{jobId}` immediately; work runs in background thread |
| `sync_status` | `jobId` | `{state: pending\|running\|succeeded\|failed, counts, errors}` |
| `ingest_file` | `filePath` (absolute, inside a root) | `{source, chunkCount, title}` — re-ingest replaces chunks |
| `ingest_data` | `data`, `source` (stable id), `format: text\|markdown\|html`, `title?` | same; html goes through markitdown HtmlConverter |
| `ingest_url` | `url` (http/https only), `source?` (default: the URL), `title?` | fetch + convert via markitdown `convert_url`; re-ingest replaces |
| `query_documents` | `query`, `topK?` (default 8), `scope?` (absolute path prefix or list) | `results`: `{text, source, title, chunkIndex, score}`; plus `sources`: distinct matched sources in rank order with `{source, title, hits}` |
| `read_chunk_neighbors` | `chunkIndex` + `filePath` or `source`, `before?`, `after?` | surrounding chunks in order |
| `read_file` | `filePath` or `source` | the source's full indexed Markdown: `{source, sourceType, title, chunkCount, text}` (chunks joined in order) |
| `list_files` | `scope?` | files on disk under roots with state `ingested` \| `not_ingested` \| `stale` (hash/mtime mismatch), plus indexed `data`/`url` sources |
| `delete_file` | `filePath` or `source` | removes all chunks for that source |
| `status` | — | config summary, model, chunk/source counts, index health; **works even when config is invalid** and explains the error |

Sync semantics (parity): ingests new and changed files (sha256 diff), skips
byte-identical ones, removes index entries for files that no longer exist. Only
the latest job is retained; a server restart discards it.

## CLI commands

```
minirag-mcp ingest <path...>       # files or directories (recursive)
minirag-mcp sync [path]            # synchronous, progress to stderr
minirag-mcp ingest-url <url> [--source S]
minirag-mcp query "..." [--scope P]...
minirag-mcp read-neighbors (--file-path P | --source S) --chunk-index N
minirag-mcp read <path> | --source <id>
minirag-mcp list
minirag-mcp status
minirag-mcp delete <path> | --source <id>
```

Global options before the subcommand: `--db-path`, `--cache-dir`,
`--model-name`; `--base-dir` is repeatable on `ingest` and `list`. CLI flags
take precedence over env vars; env var names are shared with the server.
Default document root for the CLI is the current directory. Output is
human-readable text; `--json` switches to machine-readable.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `BASE_DIR` | current directory | one document root; also the security boundary |
| `BASE_DIRS` | unset | JSON array of roots; takes precedence over `BASE_DIR`; invalid value is a hard config error (no fallback), `status` stays available |
| `DB_PATH` | `<first root>/.minirag/lancedb` | LanceDB directory; lives next to the documents (deviation from the reference's cwd-relative `./lancedb/` — cwd is unpredictable under MCP clients) |
| `CACHE_DIR` | platformdirs user cache, e.g. `~/Library/Caches/minirag-mcp/models` (macOS) | embedding model cache; global so the ~220 MB model is never duplicated per corpus (deviation from `./models/`) |
| `MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | fastembed model id |
| `MAX_FILE_SIZE` | `104857600` (100 MB) | per-file limit |
| `CHUNK_MIN_LENGTH` | `50` | minimum chunk length, chars |
| `RAG_HYBRID_WEIGHT` | `0.6` | keyword boost 0.0–1.0 (0 = pure vector) |
| `RAG_GROUPING` | unset | `similar` = first relevance group; `related` = up to two groups (vector-distance gap boundaries) |
| `RAG_MAX_DISTANCE` | unset | drop results above this distance |
| `RAG_MAX_FILES` | unset | keep results from top-N files only |

Root resolution order: CLI `--base-dir` flags > `BASE_DIRS` > `BASE_DIR` >
current directory. Each level replaces (not merges with) the lower one.
Changing `MODEL_NAME` makes existing vectors incompatible — document "new
`DB_PATH` or re-ingest", as the reference does.

## Data model

One LanceDB table, `chunks`:

| Column | Type | Notes |
|---|---|---|
| `id` | str | `"{source}#{chunk_index}"` |
| `source` | str | absolute file path, logical id for `ingest_data`, or URL |
| `source_type` | str | `file` \| `data` \| `url` |
| `title` | str | converter metadata → first H1 → basename/source id |
| `chunk_index` | int | 0-based, contiguous per source |
| `text` | str | chunk content (Markdown) |
| `vector` | vector[384] | fastembed output |
| `file_hash` | str | sha256; empty for `data` sources |
| `mtime` | float | file mtime; 0 for `data` sources |
| `ingested_at` | str | ISO timestamp |

BM25 FTS index on `text`. File listing and sync diff derive from this table
(distinct `source` + `file_hash`) — no separate registry table.

FTS tokenizer note: default (English) stemming. The keyword boost primarily
targets identifiers and error codes, which no stemmer touches; Russian recall
is carried by the multilingual vector side. Revisit only if real-world Russian
keyword queries underperform.

## Pipelines

**Ingest (file):** security check (root containment, `MAX_FILE_SIZE`) →
markitdown → title → chunk → embed (batch) → atomic replace of the source's
chunks (delete by source + add).

**Ingest (data):** `text`/`markdown` pass through; `html` → markitdown
HtmlConverter → same pipeline. Re-using a `source` id replaces its chunks.

**Ingest (url):** scheme check (`http`/`https` only) → markitdown
`convert_url` (network fetch; special converters for YouTube/Wikipedia/RSS
apply automatically) → same pipeline. `source` defaults to the URL string;
re-ingesting the same source replaces its chunks.

**Sync:** `sync_start` validates the optional path, creates the job record,
returns `jobId`, and spawns one background thread (onnxruntime releases the
GIL during embedding). The thread: recursive scan of roots (or the one path) →
extension whitelist → sha256 diff against the index → ingest new/changed,
delete vanished → update job counts `{scanned, ingested, skipped, deleted,
failed}` and per-file errors. `sync_status` reads the in-memory record.
Starting a new sync while one is running is rejected. Sources of type
`data`/`url` are never touched by sync — they are not files under the roots
and persist until deleted explicitly via `delete_file`.

Scanner whitelist: `.md .markdown .txt .pdf .docx .pptx .xlsx .html .htm .csv
.epub .ipynb`. Hidden entries (dot-prefixed) and `node_modules`,
`__pycache__`, `.venv`, `venv` directories are skipped.

**Query:** embed query → LanceDB hybrid search (explicit query vector + query
text; no embedding function registered on the table) → keyword weight from
`RAG_HYBRID_WEIGHT` via linear-combination reranking (exact reranker choice
verified at implementation; fallback: RRF + manual term-boost rerank) →
optional `RAG_MAX_DISTANCE` filter → optional relevance-gap grouping →
optional `RAG_MAX_FILES` → optional `scope` prefix filter on `source`.

**Neighbors:** fetch by `source` + `chunk_index` range `[i−before, i+after]`
(defaults: `before=1`, `after=1`), ordered.

**Read (full source):** all chunks of a source in `chunk_index` order, joined
with a blank line. This is the indexed Markdown reconstruction, not the
original file bytes.

**List:** scan roots (recursive, whitelist) and join with the index: state is
`ingested` (hash or mtime matches), `stale` (both differ), or `not_ingested`
(absent from the index); indexed `data`/`url` sources are appended as
`ingested`.

## Chunking

Two-stage, operating on Markdown (the universal intermediate):

1. **Structural split:** parse into blocks — heading-bounded sections,
   paragraphs, fenced code blocks (atomic, never split), tables, lists.
   Sections longer than the max chunk size split at paragraph boundaries.
2. **Semantic merge:** embed blocks; merge adjacent blocks while cosine
   similarity stays above a threshold and the merged size stays under the max.
   Chunks shorter than `CHUNK_MIN_LENGTH` merge into a neighbor rather than
   being dropped.

Exact size/threshold constants are implementation-time decisions, tuned on the
test corpus; they are internal (not env-configurable) in v1.

## Security

- Every file operation resolves the real path (symlinks followed) and requires
  containment in a configured root; escapes are rejected with a clear error.
- MCP file paths must be absolute (parity). CLI accepts relative paths and
  resolves them against the current directory.
- `MAX_FILE_SIZE` enforced before parsing.
- `ingest_url` accepts only `http`/`https` schemes. `file:` and `data:` URIs
  are rejected — markitdown's `convert_uri` would otherwise read arbitrary
  local files, bypassing the document-root boundary.
- No other network I/O: only `ingest_url` (explicit) and the one-time model
  download by fastembed.
- Single local user; no auth (parity). One writer per `DB_PATH`; concurrent
  read-while-sync allowed. Backup = copy the `DB_PATH` directory while no
  writer is active.

## Error handling

- Tool failures raise fastmcp `ToolError` with actionable messages
  (path outside roots, file too large, unsupported format, model download
  failure, invalid `BASE_DIRS`, unknown `jobId`, source not found, disallowed
  URL scheme, URL fetch failure — network error, timeout, non-2xx).
- Config errors do not crash the server: tools that need the config fail with
  the config error; `status` always answers and explains what is wrong.
- Per-file sync errors do not abort the job; they are collected in the job
  record.
- CLI maps the same errors to non-zero exit codes and stderr messages.

## Testing

- **Unit (fast, fake embedder via DI):** chunker (code fences intact,
  boundaries, `CHUNK_MIN_LENGTH` behavior), security (symlink escape, path
  containment, `BASE_DIRS` validation, URL scheme whitelist), config
  precedence (flags > env > defaults), result post-processing (distance filter, gap grouping, max-files,
  scope matching), scanner diff logic.
- **Integration (slow marker, real model, cached locally):** full
  ingest → query on a tmpdir corpus (RU + EN, one PDF/DOCX fixture, code-block
  markdown), hybrid ranking sanity (exact identifier query finds its chunk).
- **E2E:** fastmcp in-memory client exercising all 11 tools over the MCP
  protocol (`ingest_url` with a mocked fetch — tests stay offline); CLI smoke
  via runner (ingest → query → delete; flag-over-env precedence).
