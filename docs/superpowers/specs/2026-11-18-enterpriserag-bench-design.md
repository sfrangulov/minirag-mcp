# EnterpriseRAG-Bench for minirag-mcp — design

Date: 2026-11-18. Status: approved by the owner. Issue: `minirag-mcp-f95`.
Related: `minirag-mcp-4we.11` (user-facing retrieval regression check, separate work),
`minirag-mcp-4we.8` (reranker, shipped only if it measurably helps — this stand is how
it would be measured).

## Why

The owner wants two things:

1. **Results, and a comparison with comparable systems, in the README.**
2. **A regression check** that catches retrieval getting worse after a change.

The academic document benchmarks first considered (DocBench, MMLongBench-Doc) publish
numbers for *models* (GPT-4o, Qwen-VL), not for retrieval systems, and MMLongBench-Doc
is built for vision models: much of its evidence sits in charts and layout that a
text pipeline never sees. No published head-to-head of local RAG / MCP retrieval tools
exists.

EnterpriseRAG-Bench (Onyx, arXiv 2605.05253, May 2026, MIT) fills exactly that gap: a
public leaderboard of **whole systems** answering the same 500 questions over the same
corpus. Snapshot of 2026-11-18, overall score (43 systems):

| System | Overall |
|---|---|
| PipesHub (top) | 93.24 |
| Mixedbread (+ Opus 5) | 86.58 |
| haiku.rag (qwen3.8-27b) — local | 73.11 |
| OpenAI File Search | 61.03 |
| Bash Agent (GPT-5.4) + GPT-5.4 | 52.63 |
| BM25 + GPT-5.4 | 50.60 |
| RAGFlow | 50.24 |
| Azure AI Search | 48.42 |
| AnythingLLM | 35.58 |
| LlamaIndex (default configs) | 27.20 |
| LangChain (default configs) | 24.98 |
| Open WebUI + Chroma | 24.89 |

Source: `data/final_display_data/leaderboard.csv` in the Hugging Face Space
`onyx-dot-app/EnterpriseRAG-Bench-Leaderboard`.

## The benchmark: facts the plan depends on

All verified against the repository, dataset card and leaderboard Space on 2026-11-18.

- **Corpus.** ~500,000 synthetic documents of a fictional company ("Redwood
  Inference"): Slack 275k, Gmail 120k, Linear 35k, Google Drive 25k, HubSpot 15k,
  Fireflies 10k, GitHub 8k, Jira 6k, Confluence 5k. English.
- **Format.** Plain `.txt`. File names start with the document id (`dsid_<32 hex>`)
  followed by a semantic name; content starts with the title. `all_documents.zip`
  keeps the directory tree (top level = source type, nested subdirectories carry
  meaning); `<source>_slice_NNNN.zip` are flat zips of up to 5,000 documents each.
  The dataset parquet on Hugging Face is 1.41 GB.
- **Questions.** `questions.jsonl`, 500 rows: `question_id`, `question_type` (10
  categories: basic 175, semantic 125, intra-document 40, project 40, constrained 30,
  conflicting 20, completeness 20, misc 20, high-level 10, info-not-found 20),
  `source_types`, `question`, `expected_doc_ids`, `gold_answer`, `answer_facts`.
  High-level and info-not-found questions have no gold documents.
- **Answer file.** JSONL, one row per question:
  `{"question_id", "answer", "document_ids"}`.
- **Scorer.** `src.scripts.answer_evaluation.metrics_based_eval` in the benchmark repo:
  - correctness: binary, LLM-judged against the gold answer;
  - completeness: fraction of `answer_facts` the answer supports;
  - document recall: share of gold documents in `document_ids`;
  - invalid extra documents: count of submitted documents that are neither gold nor
    judged "valid".
  Citations are stripped before judging. Submitted documents outside the gold set go
  through a three-judge consensus that can *update the gold set*, so a system that
  finds a better document is not penalised.
- **Leaderboard overall** = mean over questions of `correct × completeness`
  (`transform_raw_data.py` in the Space).
- **Judge LLM.** `LLM_PROVIDER=openai` (default `gpt-5.4`, cheap model `gpt-5-mini`) or
  `anthropic` (default `claude-sonnet-4-6`, cheap `claude-haiku-4-5`). The judge used
  for the leaderboard is not recorded in its result files. The evaluators call the LLM
  only through `src.llm.get_llm(tools=None)` and parse JSON out of the returned text;
  `LLMInterface` is one method, `generate(messages)`.
- **Published per-question results.** The Space holds every system's answers
  (`answers_<key>.jsonl`) and per-question scores (`results_<key>.json`:
  `answer_correct`, `completeness_pct`, `document_recall_pct`, `invalid_extra_docs`).
- **Their agent baseline** gets 10 minutes of wall-clock time per question, then a
  forced final answer.
- **Submission.** By email to the maintainers; open-source systems provide a guide to
  reproduce the result.

## Goals

1. **End-to-end score** for "Claude Code + minirag over MCP" on all 500 questions,
   scored by the benchmark's own scorer, comparable with the leaderboard.
2. **Control arm**: the same Claude Code, same model, same prompt, minirag replaced by
   Claude Code's built-in read-only file tools on the same corpus. The difference
   between the arms is what minirag adds; without it a high score could be the model's.
3. **Retrieval-only recall**, no LLM: for every question with gold documents, the
   documents `query_documents` would return (same code path, `results.run_query`) and
   document recall@k against `expected_doc_ids`. Deterministic, cheap, reported by
   question type.
4. **Regression check**: the retrieval-only measurement on a reduced corpus that indexes
   in minutes, compared against a committed baseline, loud on a drop.
5. **README section** with our rows among a dated selection of leaderboard rows.
6. **A submission package** (answers, reproduction guide) the owner can send to the
   leaderboard maintainers.

## Non-goals

- DocBench, MMLongBench-Doc, ViDoRe v3, and a head-to-head with local tools (qmd,
  mcp-local-rag) on page-labelled PDFs. That is approach B, a later project.
- GPU support in minirag. Measured on the owner's M5 Max: the default model (int8 ONNX)
  embeds ~498 chunks/s on CPU and the same through `CoreMLExecutionProvider` — CoreML
  falls back to CPU for those ops. Indexing is CPU-bound; a GPU path is its own issue.
- Committing benchmark data to this repository: it is large, and the dataset carries a
  canary asking that it stay out of training corpora. Code, question-id lists, the
  subset's document-id list and baseline numbers only.
- A user-facing feature. This is a development stand under `evals/`, like
  `evals/behaviour/`; `4we.11` stays the user-facing regression command.

## Anti-goals (failure even if it technically works)

- **A flattering number through benchmark-specific tuning.** No Redwood-specific
  instructions in the minirag arm that the control arm does not also get; no special
  handling of `dsid_` or of this corpus anywhere in `src/`. Defects found at this scale
  are fixed as ordinary product work through issues, and a fix lands before a number
  is published only if it is a fix any user would want.
- **Silent success.** A score computed over a partly indexed corpus, or over questions
  that crashed and were dropped. Every report states documents indexed / failed, and
  a question that errored or timed out is scored as answered empty, never removed.
- **Comparing numbers judged by different judges as if they were the same.** The judge
  provider and model are recorded with every score and shown in the README table.

## Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **A. EnterpriseRAG-Bench first** | Ready leaderboard of 43 systems incl. OpenAI, Azure, LlamaIndex, haiku.rag; plain `.txt`; scorer accepts Anthropic or OpenAI judges. Costs: synthetic English Slack/email-heavy corpus rather than PDFs; ~500k documents (hours of indexing); two arms × 500 agent runs of model usage. | **Chosen by the owner.** |
| B. Local analogs on page-labelled PDFs (MMLongBench-Doc text subset, ViDoRe v3 subset; minirag vs qmd vs mcp-local-rag vs BM25), retrieval only | Closer to how minirag is used (PDFs), cheap and deterministic. No ready leaderboard; an adapter per competitor. | Later. |
| C. DocBench / MMLongBench end-to-end, paper protocol | Published numbers are for models; MMLongBench is visual; a GPT judge per paper. | Rejected for now. |

## Design

### Layout

`evals/enterpriserag/`, mirroring `evals/behaviour/`:

| File | Purpose |
|---|---|
| `README.md` | What this measures, how to run each step, how long each takes, caveats. |
| `fetch.py` | Download the corpus and the benchmark repo at a pinned commit into a data directory **outside the repo**; verify document count and checksums; idempotent. |
| `index.py` | Build the index with the server binary's own `sync` (as `evals/behaviour/run.py` does); `--slice` for the 5,000-document timing run; writes an index report. |
| `recall.py` | Retrieval-only recall@k through `results.run_query`, per question type; full corpus or subset. |
| `run.py` | The two agent arms through `claude -p`; resumable; per-question transcripts. |
| `answers.py` | Turn transcripts into the benchmark's answer JSONL. |
| `score.py` | Call the benchmark's `metrics_based_eval` and compute the leaderboard overall exactly as the Space does. |
| `subset.py` + `subset_ids.txt` | Build the reduced corpus for the regression check. |
| `baseline.json` | Committed retrieval-only numbers for the regression check. |

Run output goes to `evals/enterpriserag/runs/<UTC stamp>/`, git-ignored.

### Data and index

- Use `all_documents.zip` (directory tree preserved; the tree carries meaning, and
  minirag's `scope` and the control arm's file tools both benefit from it equally).
- Default data directory under the user cache (`platformdirs`), overridable. The index
  lives outside the document root (`run.py` in `evals/behaviour` already refuses an
  index inside the root; same rule here).
- minirag runs **with defaults**: default embedding model, `RAG_FTS_LANGUAGE` default
  (English), default chunk budget and hybrid weight. The number describes minirag as a
  user gets it.
- **Step 0, before the full index:** index one 5,000-document slice, time it, and
  extrapolate. Stop and report to the owner if the extrapolated full index exceeds
  24 hours or if any document fails.
- The index report records documents seen / indexed / failed, chunks, wall time,
  index size, minirag version and commit.

### Retrieval-only recall

- For each question with `expected_doc_ids`, run the question text through
  `results.run_query` with `topK=100` (`MAX_TOP_K`), map each chunk's source to its
  `dsid_` by the file-name prefix, and keep the first k **distinct documents** in rank
  order, k ∈ {5, 10, 20}. *(Builder-check call.)* `topK` counts chunks, and the
  leaderboard's baselines count documents: their vector baseline fetches 100 chunks and
  keeps the first 10 distinct documents, BM25 returns 10 documents. Counting chunks
  would understate minirag against them.
- recall@k = |gold ∩ first k documents| / |gold|, averaged per question type and overall;
  also MRR of the first gold document.
- **Headline: document recall@10.** Secondary, reported but not compared with the
  leaderboard: the distinct documents in one default `query_documents` call (`topK=8`),
  which is what the agent actually sees per search.
- This is the "one-shot retrieval" analogue of the leaderboard's BM25 and vector rows,
  reported next to them with that caveat.

### Agent arms

Both arms run through `claude -p` with the isolation `evals/behaviour/run.py` already
uses: a fresh temporary working directory outside any repository,
`--setting-sources project`, `--disable-slash-commands`, `--strict-mcp-config`,
stream-json transcripts.

| | minirag arm | files arm (control) |
|---|---|---|
| MCP | only minirag, read tools only | none |
| Built-in tools | none | `Read`, `Grep`, `Glob`, and `Bash` limited to read-only commands (`grep`, `find`, `ls`, `head`, `tail`, `wc`, `cat`) |
| Working directory | temp dir | the corpus root (read-only copy or read-only permissions) |
| Model | Opus 5.5, owner's subscription | same |
| Prompt | identical | identical |

- **Prompt** (identical for both arms): the question; the corpus description (the
  "Knowledge Sources" section of the benchmark's own `agents_example.md`, so neither arm
  gets more context than the benchmark's reference agent); and the output contract:
  answer the question, then end with a list of the documents that support the answer,
  most important first, each identified so that its `dsid_` id is visible (the file
  name carries it). *(Builder-check call.)* The prompt does not mandate a heading:
  minirag's own server instructions already ask for a `Sources` list of `displayPath`s,
  and a `displayPath` contains the file name and so the `dsid_`. A mandated
  `Documents:` heading would put two citation contracts in front of the minirag arm
  only.
- **Pilot first** *(builder-check call)*: 20 questions, two per question type, through
  both arms. It measures time per question and usage, and checks that transcripts,
  answers and the scorer work end to end. The full run starts only after the owner has
  seen the pilot's time and usage figures.
- **Time limit** 10 minutes per question, matching the benchmark's agent baseline. A
  timeout records the question as answered empty.
- **Resumable**: a question with a complete transcript is skipped on re-run.
- **Parallelism** is a flag; the default is the builder's call based on rate limits.

### Judge

The owner has Codex (ChatGPT account), no OpenAI API key. Verified 2026-11-18 with
Codex CLI 0.169.1: `gpt-5.4` is refused on a ChatGPT account ("not supported when using
Codex with a ChatGPT account"); the account lists GPT-5.6 and GPT-6.x models.

- **The judge is a GPT model through `codex exec`.** A provider shim implements the
  benchmark's `LLMInterface.generate` by flattening the messages into one prompt and
  running `codex exec --ephemeral --skip-git-repo-check -s read-only -m <model>
  -c model_reasoning_effort="medium" -o <file>` in an empty temporary directory,
  returning the last message. `score.py` injects it at runtime in place of `get_llm`;
  the benchmark checkout stays untouched at its pinned commit.
- A GPT judge rather than Claude, because the answers come from Claude: a judge from
  the answering model's own family risks self-preference.
- **Calibration, because the judge differs from the leaderboard's.** On the pilot's 20
  questions, the same judge re-scores three published systems from their published
  answers: BM25 + GPT-5.4, OpenAI File Search, haiku.rag. Per-question agreement with
  their published `answer_correct` and the mean completeness difference are reported.
  The Codex model is chosen by this calibration: the available model with the best
  agreement.
  - Agreement high (the plan's working threshold: ≥ 90% on correctness, completeness
    within 3 points): our numbers go into the README next to the published rows as
    they are, with the judge named.
  - Agreement low: the README shows our rows next to those three systems **re-judged
    by our judge on the full 500**, not next to the published numbers, and says why.
- Judging runs through the owner's ChatGPT plan limits; the pilot measures calls and
  time per question, and the full run is sized from that.

### Answers and scoring

- `answers.py` takes the final assistant text as `answer`. `document_ids` are the
  `dsid_[0-9a-f]{32}` ids in the answer's trailing citation block (the last block of
  list lines at the end of the answer), deduplicated, order kept; when there is no such
  block, every `dsid_` in the answer in order of appearance. The same rule for both arms.
  Per arm, the report counts answers with a citation block, answers that fell back to
  the whole text, and answers with no ids at all.
- `score.py` runs two evaluations *(builder-check call)*:
  - **Each arm alone** with `metrics_based_eval` (pinned commit, default correction
    flow), then the leaderboard overall. These are the README's leaderboard-comparable
    rows. Note that the correction flow can rewrite a question's gold answer and facts
    when an arm's documents change the gold set, so two arms scored alone can be graded
    against different answer keys.
  - **The arms against each other** with `comparative_eval`, which corrects both against
    the union of their documents and so grades them against one key. This, not the
    difference of the two standalone scores, is the minirag-versus-control result.
    Required, not optional.
- Every score file records: arm, answering model, judge provider and model, minirag
  version and commit, benchmark commit, date, counts of errored / timed-out / no-ids
  questions.

### Regression check

- **Reduced corpus**: the union of every question's `expected_doc_ids`, plus a seeded,
  per-source-proportional sample of other documents up to ~20,000 total. The id list is
  committed (`subset_ids.txt`) so every run indexes the same documents.
- `recall.py --subset` indexes it into a temporary index (minutes on CPU) and compares
  with `baseline.json`: per-type and overall document recall@10 and MRR, printed with
  deltas, plus the list of questions whose first gold document moved out of the top 10
  or into it.
- **The gate is overall only** *(builder-check call)*: exit non-zero when overall
  document recall@10 drops by more than 1 point (about five of the ~470 questions with
  gold documents) or MRR by more than 0.02. Per-type deltas are printed but do not
  gate: in the small categories one question is 2.5–5 points, so a per-type threshold
  would fire on noise and get ignored.
- `baseline.json` records the environment fingerprint: machine and OS, Python,
  onnxruntime and fastembed versions, model name, minirag commit, subset file hash.
  A run whose fingerprint differs prints a warning above the deltas, since CPU numerics
  and library versions can reorder near-ties.
- `--update-baseline` rewrites the baseline deliberately.
- Not part of `uv run pytest` and not in CI: it needs the downloaded corpus. Run by hand
  before a release and on any change to chunking, embedding, FTS or fusion. Documented in
  `evals/enterpriserag/README.md` and listed in CLAUDE.md's Build & Test.

### README

A "Benchmark" section with:

- our two arms' overall, correctness, completeness, recall and invalid extras;
- the retrieval-only recall@10;
- a dated selection of leaderboard rows (the table under Why), linked to the Space;
- the answering model and the judge, named;
- the caveats: synthetic, English, Slack/email-heavy corpus; the leaderboard's own
  systems use different answering models.

### Costs and time (estimates, to be replaced by Step 0 measurements)

- Indexing: CPU-bound at ~500 chunks/s; the full corpus is millions of chunks, so
  hours — an overnight run.
- Agent runs: 2 arms × 500 questions, each ≤ 10 minutes.
- Judging: per question one correctness call, one call per answer fact, and three judge
  calls per submitted non-gold document; for both arms.

## Decided with the owner

- Approach A, EnterpriseRAG-Bench first; B later.
- Judge through the owner's Codex (see Judge).

## Calls made during design (owner to check)

- Answering model Opus 5.5 on the owner's Claude subscription, both arms (proposed, no
  objection; the closest leaderboard neighbour, Mixedbread, used Opus 5).
- Pilot of 20 questions before any full run; the full run waits for the owner after the
  pilot's time and usage figures.
- Retrieval recall counts distinct documents, headline recall@10.
- The minirag-versus-control result comes from `comparative_eval`.
- Regression gate on overall recall@10 and MRR only.
- Citation contract without a mandated heading; one parsing rule for both arms.

## Deferred

- Whether to send the result to the leaderboard maintainers: decided after the owner
  has seen the numbers; it does not change what gets built.

## Left to the builder

- Exact file split inside `evals/enterpriserag/`, CLI flags, default parallelism.
- How the files arm is made read-only (permission rules or a read-only copy).
- The seed and exact size of the reduced corpus, within ~20,000 documents.
