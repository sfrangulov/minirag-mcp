# Document labels (type, project, date) classified locally — design

Date: 2026-10-22. Status: draft, pending owner review. Issue: `minirag-mcp-3hm`.
Related: `minirag-mcp-lwm` (one model stack for `[classify]` and `[ocr-max]`, later).

## Why

The owner asks the index questions like these:

- "show me the calls on project X from last week";
- "what did we discuss on the call about X" (the project name only approximately);
- "find the specs about Y".

Today the index knows a document's text, title and path, and nothing else. An agent
answering "calls from last week" has to guess the kind of a document from its file
name and its date from nowhere: file times are useless in practice (in the owner's
corpus all 105 meeting transcripts carry the same file date, the day the folder was
copied).

Small local "typed decision" models answer closed questions about a text well. One of
them, Plumb-4B (Apache-2.0, a Qwen3.5-4B fine-tune that reads out the probability of
each option letter), labelled the kind of 97% of the owner's 746 documents (numbers
and caveats under Evidence). That makes a document type cheap enough to compute at
ingest, locally, for every file.

## Goals

- Every indexed document can carry labels from **facets** defined per corpus (the
  owner's corpus needs three: project, initiative, type) and a **document date**.
- `query_documents` and `list_files` filter on facet values (one value or several per
  facet) and on a date range; `list_files` can sort by date. Results and listing rows
  show the labels with their confidence and the date with its source.
- The agent can read the taxonomy (facets, values, descriptions, counts), so it maps
  an approximate name ("the currency one") to a value itself.
- Off by default. Without the configuration file, behaviour and required dependencies
  do not change.
- Local only: the model runs in-process; nothing leaves the machine.
- Identical through MCP and CLI, implemented once in the core.

## Non-goals

- Moving or filing files. The inbox (`minirag-mcp-ab4`) was dropped for good reasons;
  labels are metadata, never a file operation.
- Generating the taxonomy from the folder tree. A draft taken from folders reproduced
  the agent's own wrong guess twice during the brainstorm (business areas instead of
  initiatives); the structure lives in the owner's head. The owner writes the file,
  helped by the agent's questions.
- Open-ended extraction (counterparty, amount, number). The model chooses between
  given options; it does not write values.
- Several values of one facet on one document. A status report covering every
  initiative gets one value; the owner's taxonomy has a "several" value for that.
- Bundling a model server or talking to one. A local endpoint as an escape hatch is a
  question for `minirag-mcp-lwm`, not for this work.

## Anti-goals (failure even if it technically works)

- **A document vanishing silently from a filtered answer.** An uncertain label or a
  missing date must never make a document disappear without the answer saying so.
- **A slower server start.** The model is never loaded or downloaded at start; the MCP
  connect wait is already tight on a fresh venv.
- **A classification failure that aborts or blackens a sync.** Per-file failures never
  abort a sync (existing rule); a missing model or extra degrades to "unlabelled" with
  a stated reason, it does not fail ingestion.
- **A taxonomy edit that re-embeds the corpus.** Labels are recomputed from stored
  text; vectors are untouched.

## Constraints

- Python `>=3.11`: the configuration is TOML, read with `tomllib`, no new dependency.
- The base install (`uvx minirag-mcp`) gains no dependency and no compile step.
  `[classify]` is an optional extra; `llama-cpp-python` ships only an sdist, so
  installing the extra compiles llama.cpp (41 s on the owner's M5 Max with Xcode
  command-line tools). The owner accepted that for this extra.
- Weights: Apache-2.0 (Plumb-4B), within the licence rule the OCR design set for
  extras (an MIT-distributable chain, weights included; no OpenRAIL, CC-BY-NC or
  terminable custom licences).
- Public repository: no content of the owner's corpus in tests, fixtures or docs.
- Thin shells: MCP tools and CLI subcommands add arguments and formatting only.
- Vectors stay unit length and untouched; new columns are additive, through the
  existing `_ADDITIVE_COLUMNS` path, so an older index opens without migration.

## Decided by the owner

- **Engine:** in-process `llama-cpp-python` over a GGUF file, extra `[classify]`. No
  server. The spike showed identical answers to `llama-server` (below).
- **Later, not now:** `[ocr-max]` and `[classify]` on one stack (`minirag-mcp-lwm`).
- **The taxonomy is a per-corpus file inside the document root**, not environment
  variables: it describes the corpus, and CLI and MCP must read the same thing.
- **Project and initiative come from paths.** The owner re-filed the corpus by hand
  (one folder per initiative, everything outside the initiatives folder is the other
  project), so these facets need no model calls at all.
- **Document type comes from the model.** The owner named the kinds they search by:
  call transcripts, minutes (MoM), mail, reports, working data, other. The agent added
  normative acts, specs and instructions after reading the corpus.
- **Client data is not project correspondence.** The material a client hands over as
  test input for an initiative (contracts, invoices, acts, bank forms, cover letters
  to the bank, exports from the client's ERP) is one type, "client data", even when a
  piece of it is a letter. "Mail" is the project's own correspondence; "working data"
  is the team's own tables. The measured list is under Evidence.
- **Letters must have dates**, as must calls and minutes.
- **A file date counts.** A document whose only date is its file's creation time is
  included by date filters and placed by date sorting, marked with source `file`. For a
  project draft written in place it is the only date there is; after a bulk copy it is
  wrong, and the mark shows that.

## The configuration file

`<root>/.minirag/classify.toml`, next to the default index location (`DB_PATH`
overrides the index location; the file stays with the corpus root). The file's
presence turns the feature on. If it is present and the extra is missing, `status`,
`sync` and every filtered answer say so; nothing is silently skipped.

```toml
[model]
# Either a local GGUF file, or a Hugging Face file pinned by revision and sha256.
path = "~/llm/plumb-4b-v5-Q8_0.gguf"
# repo = "crh225/plumb-4b-GGUF"; file = "plumb-4b-v5-Q8_0.gguf"; revision = "..."; sha256 = "..."
temperature = 2.07          # the model's calibration temperature, from its card
uncertain_below = 0.7       # top probability under this marks a label uncertain

[facets.project]
default = "rollout"         # no rule matched: this value, no model call
values.rollout = { description = "The ERP rollout project." }
values.initiatives = { description = "AI initiatives.", paths = ["initiatives"] }

[facets.initiative]
within = ["initiatives"]    # outside these paths the facet is not set at all
values.payments = { description = "Payment-request checks.", paths = ["initiatives/payments"] }
values.fx = { description = "Currency control.", paths = ["initiatives/fx"] }
values.several = { description = "Several initiatives at once: status reports, overviews.", paths = ["initiatives/common"] }

[facets.type]
question = "What kind of document is this? Judge by its form, not by the topic it discusses."
override_at = 0.9           # a path rule is the default; the model replaces it only at p >= 0.9
values.transcript = { description = "Transcript. Verbatim record of a call or meeting: spoken turns with timestamps, whatever the topic.", paths = ["transcripts"] }
values.minutes = { description = "Minutes. Date, participants, questions discussed, decisions and actions." }
# ... one entry per value; descriptions in the corpus language
values.other = { description = "Other. Notes, READMEs, anything that fits no other kind." }
```

Resolution per document and facet, in order:

1. `within` is set and the document is outside it: the facet is absent.
2. The longest matching `paths` prefix gives the value (confidence 1, source `path`).
   If the facet sets `override_at`, the model is still asked, and its answer replaces
   the path value only at that confidence or above.
3. `default` is set: that value (source `default`), no model call.
4. Otherwise the model answers (source `model`).

Paths are relative to the root that holds the document, compared with the same
containment rule as `scope.py`, after Unicode NFC normalisation of both sides (macOS
stores the owner's Cyrillic file names decomposed; a rule typed in an editor is
composed). The question text defaults to a neutral "What kind of
document is this?" in the description language; `question` overrides it per facet.

A hash of the effective taxonomy (values, descriptions, question, model file,
temperature) is stored with each document's labels. A sync after the file changes
re-labels every document whose hash differs, from the stored chunk text, without
re-parsing or re-embedding, and reports how many it re-labelled.

## Computing a label

- **Input:** `<file-name label> + ": " + basename + "\n" + the first 6000 characters of
  the stored chunk text` (chunks in order, whitespace collapsed). The label defaults to
  "File name" and is set in `[model]` to the corpus language ("Имя файла" for the
  owner's corpus). That is the measured setting (p50 2481 prompt tokens, max 4020;
  `n_ctx` 8192 leaves room). Stored chunk text, not freshly parsed text, so a first
  labelling and a re-labelling after a taxonomy edit see the same input. *(Agent's
  call: the spike measured exactly this.)*
- **Which documents:** every ingest path labels: sync, CLI `ingest`, MCP `ingest_file`,
  text and URL documents. The model is loaded per job (one sync, one CLI run, one
  `ingest_file` call) and released at its end; a warm load of the 4.5 GB GGUF took
  0.3 s (memory-mapped). Text and URL documents have no root, so path rules and
  `default` do not apply to them; only facets without `within` get a model answer.
  *(Agent's call.)*
- **Readout:** the prompt format of the model's reference client (`jevk5`: a fixed
  system message, the evidence and options as JSON, the Qwen3.5 template with
  thinking off), tokenised with special tokens, one forward pass, logits at the last
  position (`llama_get_logits_ith(ctx, -1)`; `Llama.scores` is not filled without
  `logits_all`), log-softmax over the vocabulary, the option letters renormalised
  under the calibration temperature. The full distribution over values is kept.
- **Lifecycle:** loaded lazily by the first document in a sync job that needs a model
  call, released when the job ends. Never loaded by `status`, `list_files` or a query.
  A missing local file or a failed download leaves model-sourced facets unset with the
  reason recorded; path and default facets are still applied.
- **Speed:** 0.77–0.95 s per document (p50, three runs) on the owner's M5 Max; the
  owner's corpus labels in about 10 minutes once, then only new and changed files.

## Document date

No model. Taken in this order, first hit wins, and its source is stored:

1. **Header of the document** (`document`): an e-mail header date (`Date:`, `Sent:`,
   a printed "Friday, 21 August 2026" line), a "Date: …" field at the top of minutes,
   the recording header of a Teams transcript ("August 5, 2026, 10:32AM"), a date in a
   transcript's title line, a Russian letter's outgoing-number line at the top ("Исх.
   № … от 12.08.2026"). Only header positions in the first part of the text.
2. **The file name** (`name`): `YYYY-MM-DD`, `YYYYMMDD`, `DDMMYYYY`, `DD.MM.YY(YY)`,
   `DD-MM-YYYY`, `YY-MM-DD`, a leading `DDMMYY` with or without a following space.
3. **The file's creation time** (`file`; modification time where the platform has no
   creation time).

Never a date from the body: cover letters cite the dates of the contracts they refer
to, and the first date in such a letter is a 2025 contract date, not the letter's.

Every file therefore has a date and a source; a text or URL document has one only if
its header carries it. The date does not depend on the type label, so re-labelling
never recomputes dates. Results and listings show the source, so a `file` date
(meaningless after a bulk copy) is visible as such.

## Using labels: tools and CLI

- **`query_documents`** gains `labels: {facet: value | [values]}`, `dateFrom`,
  `dateTo` (ISO dates, inclusive). Each result carries the document's labels (value,
  confidence, source) and date (value, source).
- **`list_files`** gains the same filters and `sort: "date"`. This is what "show me the
  calls from last week" is: a listing of documents, not a search over chunks.
- **The taxonomy listing**: facets, values, descriptions, document counts per value,
  uncertain and unlabelled counts, model state. Whether this is part of `status` or a
  new tool is the builder's call; the server instructions must tell the agent it exists
  and that approximate project names are matched by reading it.
- **CLI:** the same filters as flags on `query` and `list`; `classify try <file>` prints
  every facet's resolution and the model's full distribution without writing, so the
  owner can tune descriptions on a handful of documents before a full run.

Filter semantics:

- **Uncertain labels filter softly.** A document matches a requested value if that is
  its label, or if its label is uncertain (top probability under `uncertain_below`) and
  the requested value is its second choice. Such matches are marked uncertain.
- **Counts are loud where a document could vanish unfairly.** A document with a
  confident, different label is excluded legitimately and is not counted. What is
  counted, by one cheap count over the scope and never by a second search: documents in
  scope with no label for a filtered facet (model unavailable, not yet labelled), and
  documents in scope with no date when a date filter is set. `list_files` and
  `query_documents` report the same two numbers. *(Agent's call.)*
- Both the vector search and the full-text search must apply the filter before their
  fetch window, or a selective filter (eight minutes documents among 746) starves.
  LanceDB 0.36 prefilters by default on both sides (checked: a filter matching 3 rows of
  200 returns all 3 under `limit(5)`, for FTS and for vectors), but nothing in the
  project pins that; the builder adds a test that goes red under post-filtering.

## Evidence

All on the owner's corpus (746 indexed documents), Plumb-4B v5 Q8_0 in-process,
labels assigned by the agent by reading every document (33 ambiguous documents accept
either of two labels). Raw output stays outside the repository.

- **Engine parity.** On a 45-document benchmark of another corpus, in-process
  `llama-cpp-python` 0.3.36 gave 45/45 identical answers to `llama-server`, maximum
  probability difference 0.003.
- **Type, first wording:** 681/746 (91%). Meetings about requirements were read as
  specs and status decks about payment requests as primary documents: the model judged
  by topic. Product manuals were filed under "normative" by the agent and read as
  instructions by the model, which was right.
- **Type, revised wording** (question says "by form, not topic"; manuals moved to
  instructions): 728/746 (98%).
- **Type, final list** (the owner's correction: client test material is "client data",
  cover letters to the bank included; "mail" is project correspondence only):
  727/746 (97%). Initiatives folder 192/201 (client data 117/119, instructions 37/38,
  minutes 8/8, mail 5/5, normative 5/5, reports 8/10, transcripts 6/7, working data
  5/7), specs 343/344, instructions 95/96, transcripts 97/105. At p >= 0.9: 612
  documents, 3 wrong. Under p 0.7: 29 documents (4%). The remaining confident errors are
  topic-over-form again (a pilot deck and a cost deck about payment requests read as
  client data) and a note listing received contracts.
- **Path versus model** on the three uniform folders (545 documents), final list: path
  rules 543, model 535, path as default with a model override at p >= 0.9: 544. The
  model found two instructions misfiled among the specs; the override also flipped one
  ambiguous development note from spec to instruction.
- **Dates**, quick pattern set: transcripts 107/112 from header or name (the misses are
  formats the set did not cover), minutes 7/8, reports 6/10, project mail 2/5 (the
  other three are drafts written in place, dated by their files: June and July, one
  file each). Cover letters to the bank carry no date of their own; the dates in their
  bodies belong to the contracts they cite.

**Caveats.** The descriptions and the 0.9 override threshold were tuned on the same
documents they were measured on, part of the gain is redefining a type to match how the
model reads it, and one person labelled the gold. These are tuning-set numbers. The
"other" value was checked on two documents only.

## Acceptance

- Without `classify.toml`: no behaviour change, no new required dependency, the suite
  green after `uv sync --exact` (the extra removed).
- Tests run against a fake model that also returns uncertain and wrong distributions; a
  fake that is always confident would hide the soft-filter and loud-count paths.
- Every new filter and every fallback has a test that fails on the code without it.
- Re-labelling after a taxonomy edit re-embeds nothing (asserted on vectors).
- On the owner's corpus, outside the repository: type accuracy and the uncertain share
  reproduce the final-list numbers within a few documents; every letter, call and
  minutes document has a date with its source.
- README: enabling, the file format, the extra and its compile step, the model download,
  what "uncertain" means.

## Left to the builder

- The column layout for labels, probabilities, the taxonomy hash and the date, within
  "additive columns" and "filters apply before both fetch windows".
- Status versus a new tool for the taxonomy listing.
- How a path rule names its root when several roots are configured.
- The exact date patterns, starting from the measured set.
- Download mechanics for a pinned Hugging Face file (the embedder's cache directory is
  the natural home).
