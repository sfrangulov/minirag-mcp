# minirag-mcp v1 — Design

- **Date:** 2026-08-07
- **Status:** Approved (brainstorm with user, this session)
- **Reference:** [shinpr/mcp-local-rag](https://github.com/shinpr/mcp-local-rag) (TypeScript)
- **Tracking:** beads `minirag-mcp-jpw` (design), `minirag-mcp-01m` (plan), `minirag-mcp-l7s` (deferred ingest_url)

## Overview

minirag-mcp is a local-first RAG MCP server in Python — a functional analog of
mcp-local-rag. It indexes local documents, searches them with hybrid retrieval
(semantic vector search + keyword boost for exact technical terms), and sends
nothing to external services after the embedding model is downloaded.

Differences from the reference:

- **markitdown** replaces per-format parsers — more input formats (PDF, DOCX,
  PPTX, XLSX, HTML, CSV, EPub, ipynb, …).
- **trafilatura** replaces Mozilla Readability for HTML main-content extraction.
- **Multilingual embedding model by default** (Russian + English corpora work
  out of the box).
- Python stack: fastmcp, fastembed, LanceDB Python bindings.

All documentation, tool descriptions, and code are in English.

## Goals

1. Tool-for-tool parity with the reference's 9 MCP tools and CLI commands.
2. Same environment-variable contract (names and defaults) where applicable.
3. Fully local: no network access after model download (trafilatura is used as
   an HTML *cleaner*, never as a fetcher, in v1).
4. Distributable as a PyPI package runnable via `uvx minirag-mcp`.

## Non-goals (deferred)

- `ingest_url` (server-side web fetching) — beads `minirag-mcp-l7s`. Verified
  2026-08-07: markitdown ships `convert_url()` and YouTube/Wikipedia/RSS
  converters, so this is cheap to add later.
- PDF visual mode (vision-model captions for figures).
- `RAG_DEVICE` / `RAG_DTYPE` (fastembed manages ONNX providers itself).
- Multi-writer coordination. Single writer per `DB_PATH`, as in the reference.

## Stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python ≥ 3.11 | markitdown needs ≥ 3.10 |
| Package/dev manager | uv | `uv build`, `uv run pytest` |
| MCP framework | fastmcp (v3) | stdio transport |
| File → Markdown | markitdown + extras `[pdf, docx, pptx, xlsx]` | everything becomes Markdown |
| HTML main content | trafilatura | markdown output, metadata; fallback → markitdown HtmlConverter |
| Embeddings | fastembed | ONNX, no torch; default model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~220 MB quantized), override via `MODEL_NAME` |
| Store | LanceDB (embedded) | vectors + BM25 FTS + native hybrid search |
| CLI | cyclopts | already a transitive dependency of fastmcp (verified: `fastmcp-slim[server]` requires `cyclopts>=4.0.0`) — zero added deps |
| Tests | pytest | fake embedder via DI for fast tests |

Verified facts (2026-08-07, sources checked live):

- `defuddle` does not exist on PyPI; `pydefuddle` 0.1.0 is a one-release port
  (rejected as a core dependency). trafilatura 2.2.0 chosen instead.
- markitdown `HtmlConverter` converts the whole `<body>` (only strips
  `<script>`/`<style>`) — it does **no** boilerplate removal, which is why
  trafilatura stays in the pipeline.
- markitdown has `convert_url()`/`convert_uri()` out of the box (relevant only
  for the deferred `ingest_url`).
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
    parser.py     — markitdown wrapper (file → Markdown + title);
                    trafilatura wrapper (HTML → Markdown, markitdown fallback)
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

## MCP tools (parity: 9)

| Tool | Input | Output |
|---|---|---|
| `sync_start` | `path?` (absolute, inside a root) | `{jobId}` immediately; work runs in background thread |
| `sync_status` | `jobId` | `{state: pending\|running\|succeeded\|failed, counts, errors}` |
| `ingest_file` | `filePath` (absolute, inside a root) | `{source, chunkCount, title}` — re-ingest replaces chunks |
| `ingest_data` | `data`, `source` (stable id), `format: text\|markdown\|html`, `title?` | same; html is cleaned with trafilatura first |
| `query_documents` | `query`, `topK?` (default 8), `scope?` (absolute path prefix or list) | results: `{text, source, title, chunkIndex, score}` |
| `read_chunk_neighbors` | `chunkIndex` + `filePath` or `source`, `before?`, `after?` | surrounding chunks in order |
| `list_files` | `scope?` | supported files under roots + ingestion state (ingested / not ingested / stale) + `ingest_data` sources |
| `delete_file` | `filePath` or `source` | removes all chunks for that source |
| `status` | — | config summary, model, chunk/source counts, index health; **works even when config is invalid** and explains the error |

Sync semantics (parity): ingests new and changed files (sha256 diff), skips
byte-identical ones, removes index entries for files that no longer exist. Only
the latest job is retained; a server restart discards it.

## CLI commands

```
minirag-mcp ingest <path...>       # files or directories (recursive)
minirag-mcp sync [path]            # synchronous, progress to stderr
minirag-mcp query "..." [--scope P]...
minirag-mcp read-neighbors (--file-path P | --source S) --chunk-index N
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
| `DB_PATH` | `./lancedb/` | LanceDB directory |
| `CACHE_DIR` | `./models/` | embedding model cache |
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
| `source` | str | absolute file path, or logical id for `ingest_data` |
| `source_type` | str | `file` \| `data` |
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

**Ingest (data):** `text`/`markdown` pass through; `html` → trafilatura
(markdown output; on empty result fall back to markitdown HtmlConverter) →
same pipeline. Re-using a `source` id replaces its chunks.

**Sync:** `sync_start` validates the optional path, creates the job record,
returns `jobId`, and spawns one background thread (onnxruntime releases the
GIL during embedding). The thread: recursive scan of roots (or the one path) →
extension whitelist → sha256 diff against the index → ingest new/changed,
delete vanished → update job counts `{scanned, ingested, skipped, deleted,
failed}` and per-file errors. `sync_status` reads the in-memory record.
Starting a new sync while one is running is rejected.

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
- No network I/O in v1 except the one-time model download by fastembed.
- Single local user; no auth (parity). One writer per `DB_PATH`; concurrent
  read-while-sync allowed. Backup = copy the `DB_PATH` directory while no
  writer is active.

## Error handling

- Tool failures raise fastmcp `ToolError` with actionable messages
  (path outside roots, file too large, unsupported format, model download
  failure, invalid `BASE_DIRS`, unknown `jobId`, source not found).
- Config errors do not crash the server: tools that need the config fail with
  the config error; `status` always answers and explains what is wrong.
- Per-file sync errors do not abort the job; they are collected in the job
  record.
- CLI maps the same errors to non-zero exit codes and stderr messages.

## Testing

- **Unit (fast, fake embedder via DI):** chunker (code fences intact,
  boundaries, `CHUNK_MIN_LENGTH` behavior), security (symlink escape, path
  containment, `BASE_DIRS` validation), config precedence (flags > env >
  defaults), result post-processing (distance filter, gap grouping, max-files,
  scope matching), scanner diff logic.
- **Integration (slow marker, real model, cached locally):** full
  ingest → query on a tmpdir corpus (RU + EN, one PDF/DOCX fixture, code-block
  markdown), hybrid ranking sanity (exact identifier query finds its chunk).
- **E2E:** fastmcp in-memory client exercising all 9 tools over the MCP
  protocol; CLI smoke via runner (ingest → query → delete; flag-over-env
  precedence).
