# minirag-mcp

[![CI](https://github.com/sfrangulov/minirag-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/sfrangulov/minirag-mcp/actions/workflows/ci.yml)

A local-first RAG (retrieval-augmented generation) MCP server. Point it at a
folder of documents and it gives your MCP client (Claude Code, Cursor, Codex,
...) hybrid search — semantic vector similarity plus a keyword boost for
exact terms — over that content.

Nothing leaves your machine except two things: the one-time embedding-model
download on first use, and the explicit `ingest_url` call when you ask it to
fetch a web page. Ingesting local files, indexing, and querying never touch
the network.

It is a Python, MCP-native analog of
[shinpr/mcp-local-rag](https://github.com/shinpr/mcp-local-rag) (TypeScript),
built on [fastmcp](https://github.com/jlowin/fastmcp),
[fastembed](https://github.com/qdrant/fastembed), and
[LanceDB](https://github.com/lancedb/lancedb).

## Features

- **Hybrid search** — vector similarity (fastembed/ONNX) fused with keyword
  ranking (LanceDB BM25 full-text search) by weighted Reciprocal Rank Fusion,
  so exact identifiers and error codes surface alongside semantically similar
  passages.
- **Filenames are searchable** — keyword search covers document titles as well
  as body text, and an informative filename becomes the document's title when
  the document's own heading is boilerplate. In many real document sets the
  filename is the only place the document code and subject appear at all.
  See [Titles and filenames](#titles-and-filenames).
- **Multilingual by default** — the default embedding model covers 50+
  languages, so English and Russian corpora both work out of the box.
- **12 file formats** ingested via `markitdown` (PDF, DOCX, PPTX, XLSX,
  HTML, CSV, EPUB, Jupyter notebooks, Markdown, and plain text), plus direct
  text/markdown/HTML ingestion and URL fetching.
- **MCP server and CLI over the same index** — inspect and manage the index
  from a terminal without going through an MCP client.
- **Degrades gracefully** — a broken configuration doesn't crash the server;
  every tool reports the error and `status` always answers.
- **No hidden network calls** — see [Security and Operation](#security-and-operation).

## Quick Start

Not on PyPI yet — install straight from this repository. Every client below
launches the same process; only the config format differs. Replace
`/absolute/path/to/docs` with the folder you want indexed.

The invocation is `uvx --from git+https://github.com/sfrangulov/minirag-mcp minirag-mcp`.
It resolves and caches the package on first run, so start-up is slow once and
fast afterwards. To pin a revision, append `@<tag-or-sha>` to the URL.

### Claude Code

```bash
claude mcp add minirag --scope user --env BASE_DIR=/absolute/path/to/docs \
  -- uvx --from git+https://github.com/sfrangulov/minirag-mcp minirag-mcp
```

### Cursor (`~/.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "minirag": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/sfrangulov/minirag-mcp",
        "minirag-mcp"
      ],
      "env": {
        "BASE_DIR": "/absolute/path/to/docs"
      }
    }
  }
}
```

### Codex (`~/.codex/config.toml`)

```toml
[mcp_servers.minirag]
command = "uvx"
args = ["--from", "git+https://github.com/sfrangulov/minirag-mcp", "minirag-mcp"]

[mcp_servers.minirag.env]
BASE_DIR = "/absolute/path/to/docs"
```

### From a clone

For development, or to use the CLI without the `--from` prefix everywhere:

```bash
git clone https://github.com/sfrangulov/minirag-mcp
cd minirag-mcp
uv sync
uv run minirag-mcp status --base-dir /absolute/path/to/docs
```

### First use

The index starts empty — nothing is scanned until you ask for it:

1. Ask your client to sync: "sync minirag" (calls `sync_start`, then poll
   `sync_status` until it reports `succeeded`). From a terminal you can do
   the same thing synchronously: `minirag-mcp sync --base-dir /absolute/path/to/docs`.
2. Then query: "search minirag for ..." (calls `query_documents`).

The first sync (or the first ingest of any kind) downloads the embedding
model — see [Requirements](#requirements).

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (provides `uvx`)
- ~220 MB of disk space and a network connection the first time a document is
  ingested — fastembed downloads the quantized ONNX weights for the default
  model and caches them; every ingestion after that is fully offline.

## Supported Content

Files under the document root(s) with one of these 12 extensions are picked
up by `sync_start`/`sync` and `ingest_file`/`ingest`, converted to Markdown by
`markitdown`:

`.md` `.markdown` `.txt` `.pdf` `.docx` `.pptx` `.xlsx` `.html` `.htm` `.csv`
`.epub` `.ipynb`

Embedded pictures are not indexed. `markitdown` inlines each one as an
`![alt](data:image/png;base64,…)` placeholder — 8.5% of chunks on a measured
558-document corpus carried one — so the placeholder is removed before
chunking and only its alt text is kept. Image links that point at a path or an
http URL are references, not inlined pictures, and stay as written, as does a
`data:` URI inside a fenced code block.

Two more ways to get content in without a file on disk:

- **`ingest_data`** — hand the server text, Markdown, or HTML content
  directly (`format: text|markdown|html`), under a `source` id you choose.
- **`ingest_url`** — the server fetches an `http`/`https` URL itself via
  `markitdown`'s `convert_url` (YouTube, Wikipedia, and RSS get
  format-specific handling automatically). This is the one tool that reaches
  the network. Private and local hosts are refused unless
  `ALLOW_PRIVATE_URLS` says otherwise — see
  [Security and Operation](#security-and-operation).

## MCP Tools

11 tools, all backed by the same index:

| Tool | Purpose |
|---|---|
| `sync_start` | Reconcile the index with the document roots (or one path inside them). Returns a `jobId`; the work runs in a background thread. |
| `sync_status` | Poll a sync job started by `sync_start`. |
| `ingest_file` | Ingest or re-ingest one file, replacing any content already indexed for it. |
| `ingest_data` | Ingest text/markdown/html content the client holds, under a source id you choose. |
| `ingest_url` | Fetch an http(s) URL, convert it to Markdown, and index it. |
| `query_documents` | Hybrid search: semantic similarity plus a keyword boost for exact terms. |
| `read_chunk_neighbors` | Read the chunks immediately before and after a search result, for context. |
| `read_file` | Read a source's entire indexed content as Markdown (all chunks joined). |
| `list_files` | List files found on disk under the document roots, plus indexed data/url sources. |
| `delete_file` | Delete an indexed file, data item, or url item from the index. |
| `status` | Report configuration and index status. Works even when configuration is invalid. |

MCP tool file paths (`filePath`) must be absolute and inside a configured
document root.

## CLI

`minirag-mcp` with no arguments starts the MCP server on stdio; a subcommand
runs a one-shot CLI action against the same index instead.

Every subcommand accepts the same option quartet, given **after** the
subcommand, plus `--json` for machine-readable output:

| Flag (repeatable where noted) | Env var equivalent | Effect |
|---|---|---|
| `--base-dir` (repeatable) | `BASE_DIR` / `BASE_DIRS` | Document root(s); overrides the env vars entirely when given. |
| `--db-path` | `DB_PATH` | Index directory. |
| `--cache-dir` | `CACHE_DIR` | Embedding model cache directory. |
| `--model-name` | `MODEL_NAME` | fastembed model id. |

CLI-relative paths (for `ingest`, `read`, `delete`, `--file-path`, ...)
resolve against the current directory, unlike MCP tool paths, which must be
absolute. With no `--base-dir`/`BASE_DIR`/`BASE_DIRS`, the document root
defaults to the current directory.

```bash
# Index everything under a folder (recursive; also accepts individual files)
minirag-mcp ingest ~/docs

# Reconcile the index with what's on disk: ingest new/changed files,
# skip unchanged ones, drop entries for files that were deleted
minirag-mcp sync

# Fetch and index a web page
minirag-mcp ingest-url https://example.com/release-notes --source release-notes

# Hybrid search
minirag-mcp query "connection timeout error" --top-k 5

# Search only under one subtree
minirag-mcp query "changelog" --scope ~/docs/releases

# Read the chunks around a known hit, for context
minirag-mcp read-neighbors --file-path ~/docs/notes.md --chunk-index 3 --before 2 --after 2

# Read a whole indexed document back as Markdown
minirag-mcp read ~/docs/notes.md
minirag-mcp read --source release-notes   # for data/url sources

# List every file under the roots with its ingestion state
minirag-mcp list

# Config + index health, as JSON
minirag-mcp status --json

# Remove a file from the index (the file itself is untouched on disk)
minirag-mcp delete ~/docs/old-notes.md
```

The 9 subcommands: `ingest`, `ingest-url`, `sync`, `query`, `read-neighbors`,
`read`, `list`, `status`, `delete`.

Each subcommand's `--json` output carries the same fields as the matching MCP
tool. Exit status is `0` on success and `1` on failure; `ingest` and `sync`
both count any per-file failure as a failure of the run, while still printing
the full counts and a `warn:` line per file. The one exception is `status`,
which is the command you reach for when the configuration is broken: on a
configuration error it reports `{version, configError}` and exits `0`, exactly
like the `status` MCP tool. Every other command exits `1` on the same error.

## Search Tuning

Four environment variables shape `query_documents`/`minirag-mcp query`
results; none of them are exposed as MCP tool arguments.

`topK` (`--top-k` on the CLI) must be at least 1 and is capped at **100**.
Search fetches a multiple of `topK` candidates from each of the vector and
keyword sides, so an unbounded `topK` is an unbounded scan. A larger value is
clamped to the cap rather than rejected — asking for too much context is a bad
guess, not an error — while `0` or a negative value is refused outright.

### `RAG_HYBRID_WEIGHT` (default `0.6`, range `0.0`–`1.0`)

`query_documents` runs a vector search and a BM25 full-text search in
parallel, then fuses the two ranked lists with **weighted Reciprocal Rank
Fusion (RRF)**: for each candidate, `score = (1 − weight) / (k + vector_rank
+ 1) + weight / (k + keyword_rank + 1)`, where `weight` is
`RAG_HYBRID_WEIGHT` and `k = 60` is the standard RRF damping constant.

Fusing by *rank position* rather than blending raw scores is deliberate: L2
vector distance and BM25 relevance live on incomparable scales, so a
raw-score blend (or LanceDB's built-in `LinearCombinationReranker`, which
was tried first) lets a strong vector match bury an exact keyword hit no
matter how the weight is tuned. RRF sidesteps the scale mismatch entirely by
only looking at each side's ranking.

- `0.0` — pure vector search (keyword ranking ignored, FTS isn't even run).
- `1.0` — pure keyword ranking (BM25 order wins ties completely).
- `0.6` (default) — leans slightly toward exact-term matches while still
  benefiting from semantic recall.

<a id="titles-and-filenames"></a>
**Titles and filenames.** The BM25 side indexes the `title` column as well as
the chunk text, so a query matching a document's title finds it even when the
term never appears in the body. For files the title is chosen as: converter
metadata (only formats like HTML and EPUB carry it) → the first `# H1`, unless
it is **boilerplate** → the **filename stem**, when it is informative → the
first `# H1` → the stem.

A heading the author wrote is the best title available, so it wins by default.
It steps aside when it names a section rather than the document — office
document sets share their opening section ("1. Общие положения", "Лист
изменений", "Introduction", "Table of contents"), so that heading is identical
across the whole set — or when it holds no words at all, as a heading that is
only a picture does. Then the filename takes over: a stem is informative
unless it is shorter than 4 characters or, once pure-digit tokens are dropped,
consists only of generic words (`untitled`, `document`, `new`, `copy`, `scan`,
`img`, `dsc`, `screenshot`, `копия`, `документ`, …). That rejects the names
machines hand out — `Untitled-1`, `IMG_20260807_123456`, `Копия документа
(2)` — while keeping real names that merely contain such a word. Underscores
become spaces and the rest is kept as-is, so `И-112_ЗПС_Хранение ТМЗ.docx`
gives the title `И-112 ЗПС Хранение ТМЗ`.

The title is also prepended as a `# Title` line to the first chunk's text
before embedding, so it reaches semantic search too — later chunks are
untouched, and chunk boundaries, ids and counts are unaffected. Data and URL
sources are seeded only when they have a title of their own (given explicitly
or found in the content): a source id or a bare URL identifies a document
without describing it, and injecting it would only add noise to the vector.

Both are ingest-time decisions: **already-indexed files keep the title they
were ingested with until they are re-ingested.** `sync` will not do it for
you — it treats a file whose content hash is unchanged as already ingested —
so use `ingest_file` per file, or `delete_file` and re-sync. Keyword search
over the `title` column, by contrast, needs no re-ingest: an index built by an
earlier version gains the title index the next time it is opened. That upgrade
is best-effort — a read-only index directory, or a second process racing for
the same commit, leaves the index as it was and warns instead of failing, so
the database still opens and still searches (titles simply stay out of keyword
results until an index can be built).

<a id="hits-without-a-distance"></a>
**Hits without a distance.** The vector side only fetches a bounded window of
candidates, so at any weight above `0.0` the keyword side can surface a chunk
the vector side never scored. Such a hit is returned with `distance: null` —
it was ranked by BM25 alone. The two distance-based settings below each say
explicitly what they do with those hits, because "no distance" cannot be
compared against a distance threshold.

### `RAG_GROUPING` (unset by default; `similar` or `related`)

Cuts the result list at a natural relevance boundary instead of returning a
fixed `topK`. A boundary is any gap between two consecutive distances — taken
over the results **sorted by distance, ascending** — that exceeds the **mean
gap across the whole list by a factor of 2**. This ignores small jitter and
only reacts to a materially significant jump in relevance.

- `similar` — keep only the first relevance group (everything before the
  first boundary).
- `related` — keep up to two relevance groups (everything before the second
  boundary, if one exists).
- Unset — no grouping; return up to `topK` results regardless of gaps.

Only results that *have* a distance are judged, and at least 3 of them are
needed for a boundary to exist at all. [Hits without a
distance](#hits-without-a-distance) are **kept unconditionally** — a
distance-gap rule has nothing to measure them by. Surviving results keep
their fused-rank order; grouping changes which results come back, never the
order they come back in.

### `RAG_MAX_DISTANCE` (unset by default)

Drops results whose vector distance exceeds this value. Distance is
LanceDB's raw metric distance for the table (lower is more similar); it is
not normalized to `0.0`–`1.0`. Run a query without this set first to see the
distance range typical for your corpus and embedding model before picking a
cutoff.

Setting this also **drops every [hit without a
distance](#hits-without-a-distance)**: you asked for results within a
distance bound, and a chunk that was never scored by the vector side cannot
be shown to satisfy one. Expect a keyword-heavy query to return fewer results
with this set than without it, beyond the ones actually filtered by distance.

### `RAG_MAX_FILES` (unset by default)

Keeps chunks only from the first *N* distinct source files encountered in
rank order, so results don't get dominated by one large, highly-relevant
document.

## Configuration

All of these are environment variables, each overridable per-command by the
CLI's `--base-dir`/`--db-path`/`--cache-dir`/`--model-name` flags. Root
resolution order is: CLI `--base-dir` (repeatable) > `BASE_DIRS` > `BASE_DIR`
> current directory — each level fully replaces the ones below it, never
merges with them.

| Env var | Default | Description |
|---|---|---|
| `BASE_DIR` | current directory | One document root; also the security boundary for file access. |
| `BASE_DIRS` | unset | JSON array of document roots, e.g. `["/docs/a", "/docs/b"]`. Takes precedence over `BASE_DIR`. An invalid value is a hard configuration error — `status` still answers and reports it, every other tool fails until it's fixed. |
| `DB_PATH` | `<first root>/.minirag/lancedb` | LanceDB directory. Lives next to the documents by default so each corpus gets its own index; set explicitly to share one index root elsewhere. |
| `CACHE_DIR` | platformdirs user cache dir, e.g. `~/Library/Caches/minirag-mcp/models` on macOS | Embedding model cache. Global by default so the ~220 MB model is downloaded once and shared across every corpus, not duplicated per project. |
| `MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | fastembed model id. **Changing this makes existing vectors incompatible with new queries** (different model, different embedding space — even a same-dimension model isn't comparable) — pair a `MODEL_NAME` change with a new `DB_PATH` or a full re-ingest. |
| `MAX_FILE_SIZE` | `104857600` (100 MB) | Per-file size limit, enforced before parsing. |
| `CHUNK_MIN_LENGTH` | `50` | Minimum chunk length in characters; shorter chunks merge into a neighbor instead of being dropped. |
| `RAG_HYBRID_WEIGHT` | `0.6` | See [Search Tuning](#search-tuning). |
| `RAG_GROUPING` | unset | See [Search Tuning](#search-tuning). |
| `RAG_MAX_DISTANCE` | unset | See [Search Tuning](#search-tuning). |
| `RAG_MAX_FILES` | unset | See [Search Tuning](#search-tuning). |
| `ALLOW_PRIVATE_URLS` | unset (off) | Let `ingest_url` fetch hosts that resolve to loopback, link-local, private, reserved, or unspecified addresses. Off by default — see [Security and Operation](#security-and-operation). Accepts `1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off`; anything else is a configuration error. |

## Security and Operation

- Every file operation resolves the real path — symlinks followed — and
  requires containment inside a configured document root; a symlink or path
  that escapes the root(s) is rejected with a clear error, not silently
  followed.
- The same containment rule applies to scanning, so `sync`/`sync_start`,
  `ingest <dir>`, and `list` cannot pull in a file the roots don't contain. A
  symlink inside a root whose target escapes every root is skipped silently —
  it isn't an error, it simply isn't part of the corpus. (This matters because
  the extension whitelist matches the link's *name* while the parser reads the
  *target*: without the check, a `notes.md` pointing at `~/.ssh/id_rsa` would
  be indexed and returned by search.) Symlinks pointing to files that stay
  inside a root are followed and indexed as normal, under the link's path.
- MCP tool file paths must be absolute. The CLI accepts relative paths and
  resolves them against the current directory.
- `scope` (on `query_documents` and `list_files`, and `--scope` on the CLI)
  narrows results to a path **and everything under it**. Matching stops at a
  path separator, so `/docs/proj` covers `/docs/proj/notes.md` but never
  `/docs/project-secret/notes.md`. The same rule covers data and url source
  ids, with `/` as the separator: a scope of `https://example.com/docs` matches
  `https://example.com/docs/page` and not `https://example.com/docs-private`.
- `MAX_FILE_SIZE` is enforced before a file is parsed.
- `ingest_url` accepts only `http`/`https` URLs. `file:` and `data:` schemes
  are rejected — `markitdown`'s `convert_uri` would otherwise read arbitrary
  local files, bypassing the document-root boundary entirely.
- `ingest_url` also checks the **host**, not just the scheme: a host that is,
  or resolves to, a loopback, link-local, private, reserved, or unspecified
  address is refused. That covers cloud instance metadata
  (`http://169.254.169.254/latest/meta-data/`), services bound to localhost
  (`http://localhost:8080/admin`), and anything on the LAN. The URL is usually
  chosen by an LLM which may be acting on text from an already-indexed
  document, so without this an attacker-authored document is a
  prompt-injection path into your network. A name is rejected if **any** of
  its addresses is blocked, and the error names the host and the reason. A
  host that simply fails to resolve is reported as a fetch error, not a
  security refusal.
- The host check runs again on **every redirect hop**, not just on the URL you
  supplied. Checking only the given URL leaves the fetch itself open: a
  permitted public host answering `302 -> http://169.254.169.254/` would have
  had its redirect followed and the metadata response indexed. The check sits
  in the HTTP transport, which sees each hop, and the chain is capped at 5
  redirects (`requests` would follow 30). A refusal names the blocked host and
  says the fetch was redirected there.
- Set `ALLOW_PRIVATE_URLS=1` to turn the host check off — for a server you
  point at an internal wiki on purpose. It applies to redirect hops as well as
  to the URL you supply, and changes nothing else: `file:` and `data:` are
  still rejected.
- No other network I/O happens: only an explicit `ingest_url` call and the
  one-time embedding-model download ever leave the machine.
- Single local user, no authentication. One writer per `DB_PATH` at a time
  (LanceDB doesn't support concurrent writers); reading while a sync is in
  progress is fine.
- Re-indexing a source replaces its chunks by deleting the old ones and
  writing the new ones, so a sync interrupted mid-file (Ctrl-C, a crash, a
  server restart) can leave that one source temporarily absent from the
  index while its file is still on disk. This is self-healing: the next
  `sync`/`sync_start` sees the file as not indexed and re-ingests it. Nothing
  on disk is ever modified, and no other source is affected.
- Backup: copy the `DB_PATH` directory while no writer (an ingest or sync)
  is active.

## Troubleshooting

**"No results found" / empty `results`.**
Nothing has been indexed yet, or your query's `scope` excludes everything
that matches. Run `sync_start` (or `minirag-mcp sync`) first, then confirm
with `status` or `list_files` that `chunkCount`/`sourceCount` are non-zero.

**Model download fails on first use.**
The first ingestion downloads ~220 MB from Hugging Face via fastembed; a
flaky connection or a corporate proxy can interrupt it. Check connectivity,
then retry — if a partial download left the cache in a bad state, delete
`CACHE_DIR` (see [Configuration](#configuration) for its default location)
and retry.

**"... exceeds MAX_FILE_SIZE" / "file too large".**
The file is bigger than the 100 MB default limit. Raise it:
`export MAX_FILE_SIZE=209715200` (200 MB), or exclude the file.

**"Refusing to fetch from host ...".**
`ingest_url` was pointed at a host that is, or resolves to, a private or local
address. If that is deliberate — an internal wiki, a service on this machine —
set `ALLOW_PRIVATE_URLS=1`. If it is not, treat the URL as untrusted: it may
have come from a document in the index rather than from you.

**"Path outside configured document roots".**
The path (or what a symlink resolves to) isn't inside any configured root.
Check `status` for the active `roots`, and remember MCP tool paths must be
absolute.

**"BASE_DIRS must be a JSON array of ... path strings".**
`BASE_DIRS` needs valid JSON — an array of one or more non-empty path
strings: `export BASE_DIRS='["/docs/a", "/docs/b"]'`. `status` keeps working
even with a broken `BASE_DIRS`; every other tool fails until it's fixed.

**MCP client doesn't show the tools.**
- Run the same command the client runs
  (`uvx --from git+https://github.com/sfrangulov/minirag-mcp minirag-mcp`)
  directly in a terminal — it should hang silently, waiting on stdio (Ctrl-C
  to exit). If that fails, the client will fail the same way.
- Restart the client after adding or editing the server config.
- Confirm `uv`/`uvx` is on the `PATH` the client's process sees — GUI apps
  sometimes launch with a different `PATH` than your shell.
- Run `minirag-mcp status --base-dir <root>` from a terminal to confirm the
  configuration resolves the way you expect.

## License

MIT — see [LICENSE](LICENSE).
