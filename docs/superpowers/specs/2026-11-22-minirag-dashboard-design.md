# minirag dashboard: a read-only overview page served by `minirag-mcp serve`

Date: 2026-11-22. Status: design, for the owner's approval.

## Intent

One `minirag-mcp serve` now hosts several indexes, each over a local folder and cloud-synced folders, with
exclusion patterns, skipped large tables and files that fail to convert. Today the only way to see what an
index holds is `status` (totals and label counts), `list` (every file, as text or JSON) and the output of the
last sync, which is gone once the terminal scrolls. The owner asked for a simple interface to see, at a
glance, **what is where and how much**: for every index, how many documents and chunks it holds, from which
root and folder, of which kinds, and what did not get in and why.

The page is served by the server itself, on this machine only (owner, 2026-11-22: «локально прям сам сервер
пусть отдаёт»). It is part of the product, not a private script: every user of `serve` gets it.

## Goals

1. **Overview page** (`/dashboard`; `/` redirects there): one block per index of `indexes.toml`, in file order:
   - name, description, state (`ready`, `rebuild pending`, `syncing`, `unavailable` with its reason, a
     `configError` of the file);
   - documents and chunks;
   - each root with its document and chunk counts (local folder or cloud folder — shown as the path, nothing
     guessed);
   - label facets with their value counts, uncertain and unlabelled (what `status` reports today);
   - the last sync: when it finished, whether the server or a terminal ran it, its counts, the number of
     failures and of large tables skipped, and the number of exclude patterns configured.
2. **Index page** (`/dashboard/index/<name>`): folders of each root ranked by size (documents, chunks), expandable a level
   at a time; the full failure list of the last sync (file, error) and its skipped large tables; the exclude
   patterns as configured, with their comments if the loader keeps them (else the patterns alone).
3. **Folder page**: the indexed files of one folder — title, chunk count, labels, when ingested.
4. **The last sync is remembered.** Every sync of an index writes a report next to the index (see Design), so
   the page — and `status` — can say what the last sync did even when it ran in a terminal or in a process
   that has since exited.
5. **`minirag-mcp dashboard`**: prints the page's address and opens it in the browser, signed in (see Access).

## Non-goals

- No writes of any kind: no sync button, no config edits, no delete, no relabel. The page only reads.
- No search box. Search stays with the MCP tools and `query`.
- No live comparison with the disk (which files exist but are not indexed). The server cannot read online-only
  cloud files, and walking a large root on every page load is slow. What did not get in is what the last
  sync reported.
- No access from another machine, no TLS, no user accounts.
- Not in the stdio mode (`minirag-mcp --index NAME`): it has no HTTP listener.

## Anti-goals (a failure even if it works)

- Anything that weakens the guard of the MCP endpoints: `/mcp/...` keeps requiring the bearer header exactly as
  today; a dashboard cookie never authorizes an MCP request.
- The page fetching anything from the internet (fonts, scripts, styles, icons): file names are client
  material; the page is self-contained.
- The token appearing in a URL, a browser history entry, a log line or the page.
- MCP clients slowed down: the page's statistics are computed off the event loop and cached, never on every
  request of an MCP client.
- A file name that runs as markup or script in the page (file names come from other people's folders).

## Constraints

- Same process, same port, same `LoopbackGuard` Host and Origin rules as the MCP endpoints.
- No new runtime dependencies: Starlette (already there through fastmcp) serves the routes; HTML is built with
  the standard library and escaped with `html.escape`; CSS is inline; no JavaScript is required (a little is
  allowed for expanding folders, inline).
- English, like every user-visible text of minirag (CLI, tools, errors). Numbers formatted plainly.
- CLAUDE.md, «Architecture»: behaviour lives in the core. The last-sync report and its summary are core
  features visible through `status` in both shells (MCP and CLI, identical); the page is a third view of the
  same data, not a place with logic of its own.
- Works with the existing index layout: a v2 index folder with `index.json`; rebuild into `<db>.next` and swap.

## Approaches considered

1. **Pages served by `minirag-mcp serve`** — chosen. Always current, no extra step, the same guard as MCP;
   costs a login mechanism for the browser and a cache.
2. **A static HTML report written by a CLI command** (`minirag-mcp report > report.html`). No auth and no new
   routes, but a snapshot the user must regenerate, and a second copy of client file names on disk.
3. **A page published outside the machine** (a hosted artifact). Rejected: it would send client file names off
   this Mac.

The owner picked (1) and left the rest of the how to the builder («давай как считаешь нужным», 2026-11-22),
including showing the last sync's report rather than only what the index holds.

## Design

### Access

- The browser cannot send `Authorization: Bearer`, so the dashboard routes (everything under `/dashboard`) accept
  a **session cookie** instead; every other path keeps today's rule. `/mcp/...` accepts the bearer header only.
- Signing in: `minirag-mcp dashboard` reads the token file, asks the running server for a **one-time code**
  (`POST /dashboard/code` with the bearer header), and opens `http://127.0.0.1:<port>/dashboard/login?code=<code>`.
  The server checks the code (single use, expires within a minute), sets the cookie and redirects to
  `/dashboard` with a 303, so the address bar and the history hold only a spent code, never the token.
- The cookie holds a random session id, not the token: `HttpOnly`, `SameSite=Strict`, `Path=/dashboard`. Sessions
  live in the server's memory, last at most 12 hours and end with the process; then `minirag-mcp dashboard` again.
- Residual risk, accepted (builder's question, decided 2026-11-22): browsers scope cookies by host, not port, so
  a request the browser makes to another server on 127.0.0.1 under `/dashboard` carries the cookie. A server the
  user runs as themselves could read the token file anyway; the cases it opens are a server of another user or
  inside a VM/container. `Path=/dashboard` and the 12-hour cap keep it narrow; `/mcp` is unaffected.
- Without a valid cookie, a dashboard route answers 401 with one line telling the user to run
  `minirag-mcp dashboard`.
- `minirag-mcp dashboard` without a running server fails with one line naming the address it tried and how to
  start the server; `--no-open` prints the address instead of opening it.

### The last-sync report

- At the end of every sync of an index — CLI or background job, scoped or not, rebuild or not, succeeded or
  failed — the core updates `last-sync.json` in the index folder (`dbPath`), atomically (temp file and
  rename), after a rebuild's swap so it lands in the live folder. It holds two parts (builder's question,
  decided 2026-11-22), so a narrow or crashed sync never hides what an earlier full one found:
  - `latest` — the most recent sync of any kind: `startedAt`, `finishedAt`, `ranBy` (`cli` or `server`), `scope`
    (or null), minirag version, `outcome` (`succeeded` / `failed` with the error), its `counts`, and the rebuild
    info when there was one. Every sync replaces it.
  - `coverage` — what did not get in, from the most recent **unscoped sync that finished**: the same header,
    `counts`, `errors` (source and error, capped at 1000 with `errorsTotal`) and the large tables skipped. Only
    such a sync replaces it; a scoped or crashed sync leaves it as it was.
- The page shows `coverage` with its own date as «what did not get in», and `latest` beside it when it is newer
  (for example «a sync of one folder ran later, at …»; «the last sync failed: …»).
- A failure to write the report never fails the sync: it is reported the way the sync reports other warnings,
  and the next sync tries again.
- `status` gains `lastSync` (from `latest`: `finishedAt`, `ranBy`, `scope`, `outcome`, `failed`, `largeTables`),
  or nothing when no report exists yet (an index synced before this feature).
- `list` gains, on each file row that `coverage` names, its `lastSyncError` — so the failure list the page shows
  is also reachable through the MCP tool and the CLI (builder's question on parity, decided 2026-11-22). The
  page's per-root and per-folder totals are sums over `list` rows and the exclude patterns are configuration;
  neither needs a new tool. No new MCP tool or CLI command besides `dashboard`; same output in both shells.

### Data for the pages

- Overview: `status_payload` per index (what `status` returns), plus per-root counts and the `lastSync` block.
- Folder statistics: one pass over the index's source metadata (`store.list_sources`), aggregated by root and
  by folder path; no file on disk is read.
- Computed in a worker thread, cached per index, and recomputed only when the index changed (its table
  version, or a new `last-sync.json`). An index that is `unavailable` shows its reason and no statistics.
- The folder parameter is matched against indexed source paths only; it is never opened on disk, so a crafted
  value cannot reach outside the index.

### Pages

```
minirag 2.0.0aN — 5 indexes                                  generated 21.11 18:40
┌ projects-strategy   ready   2 400 documents · 84 000 chunks
│   ~/Documents/projects/client/strategy                 310 docs     9 100 chunks
│   ~/Library/CloudStorage/…/Client strategy           2 090 docs    74 900 chunks
│   type: report 640 · data 510 · transcript 97 · …  (uncertain 30)
│   last sync 21.11 18:40 by cli — 2 ingested, 0 failed, 3 large tables · 28 exclude patterns
├ …
```

The index page lists folders as rows (path, documents, chunks, share of the index) with the largest first,
then «Last sync: failed» (file, error) and «Skipped large tables», then «Exclude patterns».

### Visual direction (owner, 2026-11-22: a restrained corporate look; a mockup on real data was reviewed locally)

- **Type:** Arial / Helvetica / system sans, one family. Page title 26px bold caps, navy; section titles 20px bold
  caps; section labels 12px bold caps tracked, navy, after an 8px bright-blue square; body 14–15px; meta 11–12px
  caps tracked. No italics, no underlined text except a link on hover. Numbers right-aligned with tabular figures.
- **Colour:** white page; navy `#002B49` for the top band, table headers, headings, links and text; body
  `#3B3939`; captions `#767171`; rules `#969696`, hairlines `#E5E9EE`, row banding `#F2F2F2`, panel wash `#F2F6F9`.
  Bright blue `#0085CA` only for accents that are not text (markers, focus ring, hover underline, the
  emphasised bar): on white it is ~4:1, under the 4.5:1 needed for body text. Bars and the kinds-of-documents
  composition use the blue ramp navy → steel `#5E8AB4` → bright → `#8CC8E7` → `#B2DAEF`, grey for «other».
  Red `#9E2A2B` only for a failure count above zero, amber `#CF7F00` only for a state that needs attention.
- **Structure:** tables with a navy header row (white 11px bold caps), banded rows, horizontal hairlines only,
  never vertical borders; a total row banded and bold. Figures as plain big numbers with a caps caption, no
  cards. Panels are square, flat, unbordered washes, optionally under a navy bar label.
- **Never:** shadows, gradients, rounded corners, glass, icons from any library, emoji, logos or marks of any
  company, web fonts or any other request off the machine.
- Folders expand in place (`<details>`), which needs no JavaScript.

## Tests the builder must include (each red on the current head first)

- `/mcp/...` with a valid dashboard cookie and no bearer header is refused (401), exactly as today.
- A dashboard route without a cookie is refused; with a cookie from `/login?code=` it answers.
- A login code works once and not after it expires; the token never appears in a response, a redirect
  `Location`, or a log line.
- A file name containing `<script>` and quotes renders as text.
- A folder parameter outside every root, or with `..`, shows nothing and reads no file.
- A sync through the CLI and one through the background job each write `last-sync.json`; a failing sync writes
  `latest` with `outcome: failed` and keeps `coverage`; a scoped sync replaces `latest` and keeps `coverage`; a
  rebuild's report lands in the swapped-in folder; a report that cannot be written leaves the sync's result
  unchanged.
- `list` shows `lastSyncError` on exactly the files `coverage` names, identically in MCP and CLI.
- The cookie is refused after 12 hours and is not sent to paths outside `/dashboard` (its `Path`).
- `status` shows `lastSync` in both shells identically, and nothing for an index with no report.
- The statistics are not recomputed between two page loads with no change to the index, and are after a sync.

## Left to the builder

Pixel details within the visual direction above; the cache mechanism; the code lifetime within a minute; the session cap if any; JSON variants of the pages for tests;
README and server instructions wording; whether `status` text output shows `lastSync` on one line or several.

## Release

Ships in the next v2 pre-release (2.0.0a2) with a README section «Dashboard». The owner's LaunchAgent picks it
up when its pinned version is raised.
