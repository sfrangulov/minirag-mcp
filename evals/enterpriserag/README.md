# EnterpriseRAG-Bench

A stand that measures minirag on [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)
(Onyx, arXiv 2605.05253, MIT): ~512,000 synthetic documents of a fictional company and
500 questions, with a public [leaderboard](https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard)
of whole RAG systems. Design: `docs/superpowers/specs/2026-11-18-enterpriserag-bench-design.md`.

It measures three things:

- **End to end**: Claude Code answering each question with only minirag's read tools
  (the *minirag* arm), against the same Claude Code with only its built-in read-only file
  tools on the same corpus (the *files* arm, the control). Scored by the benchmark's own
  evaluators.
- **Retrieval only**: document recall@k of `query_documents`' code path, no LLM.
- **Regression**: retrieval-only recall on a 20,000-document subset, against
  `baseline.json`.

No benchmark data enters this repository. Everything lives under `ERB_DATA_DIR`
(default: the user cache, `…/minirag-mcp/enterpriserag`).

## Files

| File | What it is |
|---|---|
| `common.py` | Paths, questions, document ids, the pilot selection. |
| `subset.py` | The regression subset: every gold document plus a seeded per-source sample. |
| `bulk_index.py` | Builds the full index in parallel; `--verify N` proves it equals stock `sync`. |
| `index.py` | Builds or reports an index through the stock `minirag-mcp sync`. |
| `recall.py` | Retrieval-only recall (`--full`), or the regression check (`--subset`). |
| `doc_bm25.py` | Whole-document BM25 table, fused with a saved `--full` run's document ranking. |
| `rerank_full.py` | A `/v1/rerank` cross-encoder over the top 50 chunks of a full run, alone and with `doc_bm25`. |
| `run.py` | The two `claude -p` arms. Resumable; stops on usage limits. |
| `answers.py` | Transcripts → the benchmark's answer file, with counts of what went wrong. |
| `judge_codex.py`, `judged_eval.py` | The benchmark's evaluators with a Codex-served judge. |
| `score.py` | `metrics`, `comparative` and `calibrate` against published scores. |
| `report.py` | Our rows inside the leaderboard table. |

## Setup (once)

```bash
D=~/Library/Caches/minirag-mcp/enterpriserag          # or $ERB_DATA_DIR
curl -L -o $D/downloads/all_documents.zip \
  https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/download/v1.0.0/all_documents.zip
unzip -q $D/downloads/all_documents.zip -x questions.jsonl -d $D/corpus
git clone --filter=blob:none https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git $D/bench
git -C $D/bench checkout d36685e273713975ee20299bbf1ab64165575b3c   # evaluator reads generated_data/
uv venv -p 3.12 $D/bench/.venv && uv pip install -p $D/bench/.venv/bin/python -r $D/bench/requirements.txt
```

The judge runs through `codex exec` on a ChatGPT account (`gpt-5.4`, the benchmark's
default judge, is refused there; `ERB_JUDGE_MODEL` picks another). `score.py` gives it a
`CODEX_HOME` holding only a link to `~/.codex/auth.json`, so no plugin, MCP server, hook
or memory of a personal Codex setup reaches a judge prompt.

## Running

```bash
uv run python evals/enterpriserag/bulk_index.py --verify 300     # equivalence, minutes
uv run python evals/enterpriserag/bulk_index.py                  # full index, ~10 h on 18 cores
uv run python evals/enterpriserag/recall.py --full                # retrieval only, and the
                                                                  # reranker ceiling (deep_recall@k)
R=evals/enterpriserag/runs/<name>
uv run python evals/enterpriserag/run.py --arm minirag --ids sample --max-utilization 0.56 --out $R
uv run python evals/enterpriserag/run.py --arm files   --ids sample --max-utilization 0.56 --out $R
uv run python evals/enterpriserag/answers.py $R minirag; uv run python evals/enterpriserag/answers.py $R files
uv run python evals/enterpriserag/score.py metrics $R minirag
uv run python evals/enterpriserag/score.py metrics $R files
uv run python evals/enterpriserag/score.py comparative $R
uv run python evals/enterpriserag/report.py $R
```

`--ids sample` is 100 questions in the full set's type proportions; `--ids all` is 500
and `--ids pilot` two of every type. `--max-utilization` is required: every run reports
the account's seven-day usage, and the runner stops once it reaches the cap — the agent
runs share the weekly limit of the account doing other work. Run the arms one after the
other, on an otherwise idle machine and never during indexing: the files arm's searches
scan 3.2 GB and time out under contention, which measures the machine, not the agent.

On a sample, `report.py` also rescored every published system on the same questions from
its published per-question results, so the comparison is on identical questions without
re-judging anything. Over all 500 those rescorings reproduce the published leaderboard to
the hundredth.

The judge runs on the ChatGPT plan's limits too: one hour of calibration (about 1,500
calls) used up a Plus plan's five-hour window. `score.py calibrate $R` re-judges
three published systems' answers on the run's questions and reports agreement with their
published scores, which says how far our judge's numbers can sit next to the leaderboard's.

## Regression check

```bash
uv run python evals/enterpriserag/recall.py --subset                   # compare with baseline.json
uv run python evals/enterpriserag/recall.py --subset --update-baseline # deliberately move it
```

Run it before a release and after any change to parsing, chunking, embedding, FTS or
fusion. It fails when overall document recall@10 drops by more than 1 point or MRR by
more than 0.02; per-type deltas are printed but do not gate (one question in a small
category is 2.5–5 points). A different environment fingerprint is printed as a warning.

## Experiments

The stand runs the production retrieval path, so an experiment is configuration, not a
patch: the product's own variables pass through (`MODEL_NAME`, `CHUNK_TOKEN_BUDGET`,
`RAG_HYBRID_WEIGHT`, `RAG_DOCUMENT_WEIGHT`, `RAG_RERANK_TOP`, `RAG_FTS_LANGUAGE`,
`EMBED_URL`, `EMBED_ENDPOINT_MODEL`, `RERANK_URL`, `RERANK_MODEL`,
`ENDPOINT_API_KEY_FILE`). A `--subset` run with any of them set, or with `--budget`, is
written to `runs/subset_<model>-b<budget>-dw<weight>[-rr][-endpoint].json` and never
touches `baseline.json`.

```bash
EMBED_URL=http://127.0.0.1:8080/v1/embeddings EMBED_ENDPOINT_MODEL=multilingual-e5-large \
RERANK_URL=http://127.0.0.1:8091/v1/rerank RERANK_MODEL=bge-reranker-v2-m3 \
  uv run python evals/enterpriserag/recall.py --subset
uv run python evals/enterpriserag/recall.py --subset --budget 300
```

`--budget N` (subset only) sets `CHUNK_TOKEN_BUDGET` for both the index build and the
measurement, since a different budget is a different index. `rerank_full.py` reads
`RERANK_URL` and `RERANK_MODEL` the same way. `baseline.json` predates these defaults and
is re-taken after the final pipeline is checked as a whole.

## Why a parallel indexer

Stock `sync` ingests one file at a time on one core: 3.3 files/s on this corpus, about
40 hours. `bulk_index.py` runs the real `Pipeline.ingest_file` in worker processes and
writes in batches. `--verify` builds the same subset both ways and requires identical
rows, signatures and query results, and a stock `sync` over the bulk-built index that
finds nothing to do.

## Caveats

- The corpus is synthetic, English, and mostly Slack and email, not PDFs.
- The leaderboard's systems use their own answering models.
- Our judge is not the leaderboard's; see the calibration numbers next to any score.
