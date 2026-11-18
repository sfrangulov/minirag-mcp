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
uv run python evals/enterpriserag/recall.py --full                # retrieval only
R=evals/enterpriserag/runs/<name>
uv run python evals/enterpriserag/run.py --arm minirag --ids all --out $R
uv run python evals/enterpriserag/run.py --arm files   --ids all --out $R
uv run python evals/enterpriserag/answers.py $R minirag; uv run python evals/enterpriserag/answers.py $R files
uv run python evals/enterpriserag/score.py metrics $R minirag
uv run python evals/enterpriserag/score.py metrics $R files
uv run python evals/enterpriserag/score.py comparative $R
uv run python evals/enterpriserag/report.py $R
```

`--ids pilot` runs two questions of every type (20); `score.py calibrate $R` re-judges
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
