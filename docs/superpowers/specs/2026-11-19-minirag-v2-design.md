# minirag v2 — design

Date: 2026-11-19. Epic `minirag-mcp-g7q`. Status: written for the owner's approval.

v2 is built for one user first — the owner, on an Apple-silicon Mac that already runs a
llama.cpp router — and for everyone else second. It ships as a **test (pre-release)
version**; 1.1.1 stays what `pip install minirag-mcp` and `uvx minirag-mcp` resolve to.

## Why

Two stands measured where 1.1.1 loses answers:

- **English** — EnterpriseRAG-Bench (`evals/enterpriserag/`): 511,962 synthetic enterprise
  documents, 470 questions with gold documents. 1.1.1 finds a gold document in its top 10 for
  54.9% of questions (MRR 0.413). Its vectors (MiniLM) add almost nothing: keyword-only scores
  54.1%, vector-only 21.5%. Chunk-level BM25 trails the leaderboard's whole-document BM25 by
  ~14 points.
- **Russian** — the owner's working corpus (952 documents, 239k chunks, already on
  `multilingual-e5-large`), with 150 questions written from the documents (51 lookup, 51
  paraphrase, 24 title, 24 multi-document) and 2,343 pooled hits judged by independent
  judges. Kept outside this public repository; scripts and numbers are summarised here.

What the measurements say (details in the issues named):

| change | owner's Russian corpus (agent-style queries) | English, full corpus |
|---|---|---|
| e5 `query:`/`passage:` prefixes (g7q.1) | hybrid unchanged within noise; vector side +3-4 points | — |
| fusion keyword weight 0.3-0.75 (g7q.5) | all within 3 of 150 questions: keep 0.6 | to re-tune with the rest |
| whole-document BM25 as a third signal (g7q.4) | first-ranked hit 59% -> 67% (p 0.03), MRR +0.05 | top-10 54.8% -> 71.1%, MRR 0.41 -> 0.60 (document-list fusion) |
| document vectors (mean of chunk vectors) | hurts (-16 questions) | — |
| cross-encoder over the top 30-50 chunks (g7q.7) | with document BM25: MRR 0.757 -> 0.825, hit in the default 8-chunk window 90% -> 93%; natural-language questions MRR 0.71 -> 0.75 | MRR 0.413 -> 0.486 alone, 0.630 with document BM25 |
| classify model served by llama.cpp (g7q.3) | 180/180 identical decisions, same speed | — |
| OCR engine (g7q.9) | RapidOCR drops 12 of 83 hand-checked lines (one a contract sum); Tesseract 83/83, CER 0.001 | — |

Beyond retrieval quality: the owner runs three corpora through three separately configured
stdio instances, each loading its own embedding model; indexing a 512k-file corpus with stock
`sync` takes ~40 h on one core (the stand's parallel indexer: 9.3 h, proven equivalent); and a
llama.cpp endpoint embeds e5 ~5x faster than all CPU cores with vectors equal to fastembed's
(cosine 0.99999).

## Goals

1. One long-running local server hosting several named indexes. A project connects by URL; the
   URL names its default index; a tool call may name another index when the user asks.
2. Every model minirag uses — embeddings, reranker, classify — can be served by an external
   OpenAI-style llama.cpp endpoint, with nothing loaded in minirag's process.
3. Retrieval gains measured above: e5 prefixes, whole-document BM25 as a third fusion signal,
   an optional cross-encoder reranker.
4. An index records how it was built; a mismatch rebuilds it instead of mixing vectors.
5. Faster ingest for large corpora: parallel workers, batched writes.
6. OCR through Tesseract, in Tesseract's own reading order.
7. The owner's corpora re-indexed with v2 and the owner's clients switched to it — last.

## Non-goals

- Searching several indexes in one call, or fusing their rankings. Indexes may use different
  models and stemmers; their scores are not comparable, and the owner wants no mixing.
- Remote access, multi-user serving, authentication beyond a local token.
- Making v2 the default install. 1.x keeps working and keeps its indexes.
- Classifying document versions or claiming which is newer (settled in an earlier spike).
- Apple Vision OCR as the primary engine (measured; revisit only as a second reader).

## Anti-goals (failure even if it technically works)

- **A v2 trial that breaks 1.x.** 1.1.1 cannot tell a prefixed index from its own and would
  answer from incompatible vectors without any error. v2 must never rewrite a 1.x index.
- **Silent cross-index reads.** A session must not see another index's documents unless an
  index was named explicitly; a refused index must say so, not return nothing.
- **Silent degradation.** An unreachable endpoint, a skipped reranker, an index needing a
  rebuild: each shows in `status` and in the affected tool's result.
- **MCP and CLI drifting apart.** Every behaviour lives in the core; both shells expose it
  identically (project rule).
- **A local HTTP server any web page can read.** A browser on the same machine must not be
  able to query client documents through DNS rebinding or a permissive origin.

## Constraints

- Python >= 3.11; everything through `uv`; CI matrix 3.11-3.14 on Linux and macOS.
- English in code, docs and commits. Tests must fail on unfixed code (CLAUDE.md traps).
- Claude Code truncates server instructions at 2048 characters
  (`instructions.py`, `MAX_BUILTIN_INSTRUCTIONS_CHARS`): the index rules must fit.
- `pyproject.toml` holds the only version number; release through `bump-my-version`.
- Private evaluation material (the Russian ruler, OCR and classify spikes) stays outside the
  repository; only aggregate numbers appear here.

## Approaches considered

### Several corpora

1. **Index fixed per connection** (one URL per index; a project needing two indexes adds two
   connections). Hard isolation, no tool changes; each extra connection repeats the 12 tool
   descriptions in the agent's context.
2. **One connection, every call names its index.** Least configuration; the boundary is only
   the agent's discipline.
3. **Default index per connection plus explicit override** — chosen by the owner. The URL sets
   the default; any tool takes an optional `index`; an optional allow-list in the URL makes a
   hard boundary where a project needs one.
4. **One index over everything with restructured folders** (the owner's first idea). Dropped by
   the owner in favour of 3: one index forces one model and one stemmer on Russian and English
   corpora and makes isolation a matter of path filters.

### Retrieval

Measured and kept: prefixes, document BM25 as a per-chunk bonus, reranker. Measured and
dropped: document vectors from mean-pooled chunk vectors (hurt), first-chunk vectors (neutral),
re-weighted fusion (noise). Fusing whole document lists instead of a per-chunk bonus gained a
little more on short queries but leaned worse on natural-language questions and does not fit
a chunk-returning API. Not measured: whole-document vectors from a long-context model (bge-m3).

### OCR

RapidOCR (today), Apple Vision, Tesseract — measured on the owner's scans; Tesseract chosen
(see table above). Vision is 7x faster but drops lines and short words, orders two-column
pages block by block and loses small text in wide images without tiling.

## Decided with the owner

- v2 is a test release; 1.1.1 stays the default; the owner's corpora move to v2 only at the
  very end (g7q.13), keeping the 1.x index until v2 is confirmed, so rolling back is a config
  edit.
- Indexes stay separate and are never mixed in one call; option 3 above.
- An index other than the default may be named for reads **and writes** when the user asks.
  Any index is reachable by explicit name unless the connection's allow-list says otherwise.
- The owner's working index is not switched to a local checkout before the release.
- Prefixes ship with v2's re-index; the keyword weight stays 0.6 on the owner's corpus.

## Design

### 1. The index server

**Configuration** — `indexes.toml` in the user config directory
(`$XDG_CONFIG_HOME` or `~/.config`, then `minirag/`, on macOS too, matching `~/.config/llama/`;
`platformdirs.user_config_path("minirag")` on Windows). `MINIRAG_CONFIG` overrides the path.

```toml
[server]
host = "127.0.0.1"
port = 8765
api_key_file = "~/.config/llama-server/api-key"      # bearer key for the endpoints

# Which embedding models are served by an endpoint, keyed by the fastembed model id an index
# names. A model without an entry runs in-process (fastembed), as today.
[embedding."intfloat/multilingual-e5-large"]
url = "http://127.0.0.1:8080/v1/embeddings"
name = "multilingual-e5-large"                       # the router's model name

[rerank]                                             # optional; no reranking otherwise
url = "http://127.0.0.1:8080/v1/rerank"
name = "bge-reranker-v2-m3"

[classify]                                           # optional; in-process GGUF otherwise
url = "http://127.0.0.1:8080"
name = "plumb-4b"

[index.client-a]
description = "Client A engagement documents"
roots = ["~/Documents/client-a", "~/vault/Inbox"]
fts_language = "Russian"
# model, db, token_budget, hybrid_weight, ocr_lang, ... : every per-index setting that is an
# environment variable today, with v2's defaults (model: intfloat/multilingual-e5-large)

[index.reference]
description = "Reference library: frameworks and methods"
roots = ["~/Documents/reference"]
```

Each `[index.NAME]` maps onto today's `Config` (roots, db_path, model_name, fts_language,
token_budget, hybrid_weight, ...). Names: lowercase letters, digits, `-`; they appear in URLs
and tool arguments. The default `db` is `<first root>/.minirag/v2/` — a different directory
from 1.x's `<first root>/.minirag/lancedb`, so both versions coexist untouched.
`classify.toml` stays where it is today: `<first root>/.minirag/classify.toml`.

**Process** — `minirag-mcp serve` runs streamable-HTTP MCP (FastMCP >= 3.4) for every
configured index. A LaunchAgent template ships in the docs (a llama-server is commonly
run that way); the README shows `launchctl bootstrap`. On SIGHUP or a changed `indexes.toml`
the server reloads the index list; an index removed from the file stops answering, its data
stays on disk.

**URLs** — `http://127.0.0.1:8765/mcp/<index>` makes `<index>` the connection's default.
`?allow=a,b` restricts the connection to its default plus `a`, `b`; without `allow`, every
configured index is reachable by explicit name. `/mcp` without an index has no default: every
call must name one.

**Tools** — the 12 tools keep their names and contracts; each gains an optional `index`
argument. With no `index`, the call uses the connection's default (or fails loudly when there
is none). One call always touches exactly one index. Every result carries `index`, so a
follow-up `read_file` / `read_chunk_neighbors` can pass it back. New tool `list_indexes`:
name, description, document and chunk counts, language, model, whether this connection may
use it, and its state (`ready`, `syncing`, `needs rebuild`, `unavailable`). Writes
(`ingest_*`, `delete_file`, `sync_start`) follow the same rule as reads: allowed in any
reachable index when named.

**Refusals are loud** — an unknown index, an index outside the allow-list, an index needing a
rebuild: the tool returns an error naming the index and the reachable ones, never an empty
result.

**Instructions** — the server instructions gain one short block: use the default index; name
another only when the user names it or asks to look elsewhere; say which index an answer came
from. Index descriptions reach the agent through `list_indexes`, not the instructions (the
2048-character cap).

**Security** — binds 127.0.0.1 only (refuses a non-loopback `host` unless an explicit
`allow_remote = true`, which v2 does not document). Requires a bearer token: generated on first
start into `~/.config/minirag/token` (mode 0600); clients pass it in the `Authorization`
header (Claude Code's HTTP MCP config supports `headers`). Validates `Host` (loopback names
only) and rejects requests carrying a non-loopback `Origin`, per the MCP transport guidance on
DNS rebinding.

**Clients** — Claude Code and Codex connect over HTTP directly. Claude Desktop's local config
runs stdio only: `minirag-mcp connect <index> [--allow a,b]` is a thin stdio-to-HTTP bridge to
the running server (token read from the same file).

**Single-index mode** — `minirag-mcp` configured by environment variables, one index per
process over stdio, stays as a *way of configuring*, not as 1.x behaviour: it runs the same v2
core as `serve`, for one index. Consequences: its default `DB_PATH` becomes
`<first root>/.minirag/v2/` (a 1.x index next to it is left alone, and the first v2 sync builds
a fresh index); the endpoints get environment variables (`EMBED_URL`, `EMBED_ENDPOINT_MODEL`,
`RERANK_URL`, `RERANK_MODEL`, `CLASSIFY_URL`, `CLASSIFY_MODEL`, `ENDPOINT_API_KEY_FILE`) next
to the new retrieval ones (`RAG_DOCUMENT_WEIGHT`, `RAG_RERANK_TOP`); an explicit `DB_PATH`
pointing at a 1.x index is refused with a message naming the v2 path. One core, one index
format, one retrieval path.

**CLI** — `minirag-mcp --index NAME <command>` reads `indexes.toml`; without `--index` it uses
the environment variables, exactly as the single-index server does. Same core calls, same
output, as the project requires.

### 2. Model endpoints

All three are optional; each has today's in-process behaviour as the fallback, and `status`
and `list_indexes` say which backend is live.

- **Embeddings** — per model, `[embedding."<fastembed id>"]` gives the endpoint `url` and the
  router's `name`; an index whose model has no entry embeds in-process. OpenAI
  `/v1/embeddings`. Chunk sizes are still
  counted locally with the model's own `tokenizer.json`, untruncated, as minirag counts them
  today (prototype: `evals/enterpriserag/common.py: apply_embedding_endpoint`). Vectors are
  unit-normalized on arrival. Measured: llama.cpp f16 e5 vs fastembed, cosine >= 0.99999, so an
  index built by one backend is served by the other and the backend is not part of the
  embedding scheme. Unreachable endpoint: fall back to in-process fastembed for the same
  model, loudly (status flag + a warning in results), never to a different model.
  **The served model is verified**, since a router preset pointing at another model of the
  same dimension would mix vectors silently: `index.json` records the vector of a fixed probe
  text at build time; on open (and when the endpoint comes back) the probe is embedded through
  the live backend and must match with cosine >= 0.999, or the index is `unavailable` with a
  message naming both models.
- **Reranker** — `[rerank]` `url` + `name`: llama.cpp `/v1/rerank` (raw logits). Model:
  `BAAI/bge-reranker-v2-m3` (Apache-2.0), converted from the official weights with
  `convert_hf_to_gguf.py` at the router's own llama.cpp commit and served with
  `--rerank --pooling rank`; verified against the reference CrossEncoder (Pearson 0.999998,
  identical order). Unreachable or unset: no reranking, flagged in results.
- **Classify** — `[classify]` `url` + `name`: an `EndpointChoiceModel` with
  `LlamaChoiceModel`'s interface (prototype verified 180/180): `/tokenize`
  (`add_special: false, parse_special: true`), `/completion` with the token ids,
  `n_predict 1, n_probs 100, post_sampling_probs false, cache_prompt false, temperature 0,
  samplers []`; option letters matched by token id; same calibration and `N_CTX` guard. At
  startup it checks that the served model is the sha256-pinned file of `classify.toml`
  (llama.cpp `/props` exposes the model path). Unset: in-process as today (needs the
  `[classify]` extra).
- **Router presets** — the owner's router needs the reranker as a fourth resident model
  (`--models-max 4`: e5 1.1 GB, reranker 1.2 GB, classify 4.5 GB, a chat model) and >= 8192
  tokens per slot for classify. Documented in the README; applied on the owner's machine in
  g7q.13.

### 3. Index identity and rebuilds

Each v2 index directory holds `index.json`:

```json
{"format": 2, "minirag": "2.0.0a1", "generation": 3,
 "embedding": {"model": "intfloat/multilingual-e5-large", "dim": 1024, "prefixes": "e5",
               "probe": [0.0132, -0.0418, ...]},
 "chunking": {"scheme_version": 4, "token_budget": 300},
 "fts_language": "Russian", "document_table": true}
```

- **Queries always follow the recorded scheme**, not the configured one: an index built with
  model M and prefix rule P is searched by embedding the query with M and P. So a changed
  configuration never breaks search; it makes the index `rebuild pending` (shown in `status`,
  `list_indexes` and each result), and the index keeps answering from its existing vectors.
  If M itself cannot be loaded (no endpoint entry and no in-process model), the index is
  `unavailable` and says why.
- The rebuild starts on the next explicit `sync` (agent, CLI or `sync_start`), never by
  itself at server start — a 9-hour rebuild is the user's call. Different `fts_language`:
  today's in-place FTS rebuild (no re-embedding).
- A rebuild writes a complete new index into a sibling directory, then swaps it in atomically
  and bumps `generation`; the old one keeps serving until the swap and is kept as
  `<db>.previous` until the next successful rebuild. A running server notices a swap made by
  another process (the CLI) through `generation` before each request and reopens the index.
  An embedding-only change re-embeds the stored chunk text (no re-parse, no OCR, no relabel);
  a chunking change re-ingests.
- A directory without `index.json` that contains a LanceDB table (a 1.x index) is never opened
  for writing by v2: v2 refuses with a message naming the v2 default path.
- `prefixes: "e5"` applies to the `intfloat/*e5*` family: documents embedded as
  `passage: <text>`, queries as `query: <text>`. Other models: `none`. The chunk budget
  subtracts the prefix's tokens.
- **Default model** — v2's default becomes `intfloat/multilingual-e5-large` with the e5
  prefixes (g7q.1). Costs for a user without a router: the first in-process download is
  2.24 GB (fastembed's ONNX; MiniLM was 0.22 GB) and CPU embedding is ~28 chunks/s on 18
  cores. The slow test, the README and the English stand's baseline move with it. A user who
  wants MiniLM sets `MODEL_NAME` (or `model` per index) as before.
- **Chunk budget per model** — today `MODEL_MAX_TOKENS = 128` caps every model at MiniLM's
  window, so the owner's e5 index is cut at 110 tokens although e5 reads 512. v2 takes the cap
  from the model (e5: 512) and picks the default budget for e5 **by measurement before the
  re-index**: 110 vs ~250 vs ~400 on the Russian ruler and the English 20k subset, together
  with the reranker (its inputs grow with the chunks). The table above was measured at 110.

### 4. Retrieval

Per query, within one index:

1. Chunk vector search and chunk BM25 (text + title), fetch window as today.
2. **Document BM25**: a `documents` table in the index — one row per source (source, title,
   full parsed text joined from its chunks, file hash) with FTS on text and title in the
   index's language — top 100 documents.
3. Fusion: today's weighted RRF over chunks plus a document bonus:
   `score(c) = (1-w)/(60+r_vec(c)) + w/(60+r_fts(c)) + wd/(60+r_doc(source(c)))`,
   `w = RAG_HYBRID_WEIGHT` (0.6), `wd = RAG_DOCUMENT_WEIGHT` (0.5; 0 disables). Ties by id.
4. **Rerank** (when configured): the top `RAG_RERANK_TOP` (30; 30 and 50 measured equal)
   fused chunks are rescored by the cross-encoder on (query, chunk text) and reordered; the
   rest keep their fused order. Distance-based filters (`max_distance`, `grouping`) apply to
   the first stage before reranking.
5. Everything after (aggregation per source, parents, labels, scopes, date filters) unchanged.

The `documents` table is maintained wherever chunks are: ingest replaces a source's row,
delete removes it, a move renames it, `sync` folds new rows into its FTS index the same way
920e241 does for chunks. Cost: one more FTS query (milliseconds) and the reranker (~0.1-0.3 s
on the owner's GPU). `find_mentions` (literal search) is unchanged.

### 5. Ingest

- **Parallel sync** — `sync` parses, chunks and embeds in worker processes and writes in
  batches from the parent, as `evals/enterpriserag/bulk_index.py` does (its `--verify` proves
  identical rows, signatures and query results to stock sync). Worker count defaults to
  physical cores minus one, `RAG_SYNC_WORKERS` overrides; with an embedding endpoint the
  workers' requests run concurrently against it. Per-file failures still never abort a sync.
- **OCR** — the `[ocr]` extra becomes `tesserocr>=2.13` (wheels bundle libtesseract 5.5.4 for
  macOS 15+, Linux, Windows) instead of rapidocr, rapid-orientation and opencv-python
  (+171.5 MB -> ~+9 MB). `rus`, `eng`, `osd` traineddata (tessdata_fast, 18.5 MB) download to
  `cache_dir/ocr` on first use, pinned by commit and sha256 like today's models.
  `PSM.AUTO_OSD`, PDF pages rendered at 300 dpi with `SetSourceResolution(300)`. The text is
  taken in Tesseract's own reading order; `assemble_lines`, which interleaves the columns of
  two-column pages for every engine (g7q.14), is no longer on this path. `ocr_engine` becomes
  `tesseract`. `RAG_OCR_LANG` maps to Tesseract language codes.
- **Parser noise** (g7q.12) — struck-through `.docx` text is dropped at parse time (deleted
  requirements are searchable as current today: a correctness fix). The other items —
  boilerplate and table-header titles, transcript block headers, PDF leader dots, spreadsheet
  `Unnamed: N`/`NaN` — are fixed only if the ruler shows they move retrieval.

### 6. Measurement gates

- Before the re-index: the chunk-budget measurement (section 3) and one joint check of the
  final pipeline on both stands.
- The Russian ruler and its scripts stay private; the English stand is in the repo. A
  retrieval change lands with its numbers in the issue.
- `evals/enterpriserag/baseline.json` is re-taken after the retrieval changes (it predates
  2ee521f's tie order).

### 7. Release and migration

- **Where v2 lives**: on `main`, through feature branches and pull requests as usual;
  versions `2.0.0aN` are tagged from `main`. There is no long-lived v2 branch. A 1.x patch
  (the three fixes merged in #31) is not released now (owner's call); if one is ever needed,
  it is cut on a `release/1.x` branch from the `v1.1.1` tag with those commits cherry-picked,
  and tagged from there (`release.yml` already exists on `main`, which is what GitHub
  requires).
- Version `2.0.0a1` onwards. `bump-my-version` learns pre-release parts. `release.yml`
  already publishes pre-releases to PyPI (pip skips them without `--pre`).
- **MCP Registry**: `publish-mcp.yml` runs on every `v*` tag. Before the first pre-release tag,
  verify how the registry treats a pre-release version; if it could become what registry
  clients install by default, skip pre-release tags in that workflow.
- The `.mcpb` Desktop bundle is built from whatever branch it runs on (`bundle.yml` builds,
  nothing publishes it), so bundles built from `main` are v2. v2's bundle wraps the
  single-index mode; nobody is pointed at it until v2 is the default.
- g7q.13, last: router presets on the owner's machine, `indexes.toml` for the owner's three
  corpora (each re-embedded with e5 and prefixes), clients switched to HTTP URLs (Desktop via
  `connect`), regenerated constraints for the pinned `uvx` installs; 1.x indexes kept until
  v2 is confirmed on them.

## Calls made during design (owner to check)

- Port 8765; config `~/.config/minirag/indexes.toml`; token file `~/.config/minirag/token`.
- Default model `intfloat/multilingual-e5-large` with prefixes for every new index (g7q.1);
  MiniLM only when configured.
- A bearer token is required even on loopback.
- `/mcp` without an index is allowed but has no default.
- Reranking top 30 by default; document bonus weight 0.5; keyword weight 0.6.
- Endpoints fall back to in-process models loudly rather than failing searches.
- v2 indexes default to `<first root>/.minirag/v2/`.
- Rebuilds keep one previous copy (disk cost: one index).
- Struck-through text is dropped in v2 without a ruler measurement (correctness).
- Sync stays triggered as today (agent or CLI); no file watcher in v2. A rebuild after a
  configuration change also waits for an explicit sync; until then the index answers from
  its recorded scheme.
- The env-var single-index mode runs the v2 core (new default path, new variables), not
  frozen 1.x behaviour.
- Embedding endpoints are mapped per model id; the served model is verified with a recorded
  probe vector.
- v2 is developed on `main`; a 1.x patch, if ever, comes from a `release/1.x` branch.

## Left to the builder

- How FastMCP serves several defaults (one app reading the path, or one mounted app per
  index sharing a registry), as long as the behaviour above holds.
- The `index.json` schema details and the atomic swap mechanics on macOS and Linux.
- Worker/batch sizes for parallel sync; how workers share the embedding endpoint.
- The `connect` bridge's implementation.
- Test design, following CLAUDE.md: each fix's test must fail on the unfixed code.

## Deferred

- g7q.8 end-to-end benchmark run and README section (after v2).
- g7q.10 English 5k dev tier (full-corpus checks through saved runs proved cheap enough).
- Apple Vision as a macOS second reader for low-confidence Tesseract pages; Kazakh OCR.
- Whole-document vectors from a long-context model.
- Gold for the 107 real agent queries on the Russian ruler.
