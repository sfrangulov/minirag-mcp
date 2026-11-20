# EnterpriseRAG-Bench stand — Implementation Plan

> **Dropped 2026-11-20 (owner):** no further benchmarking on EnterpriseRAG-Bench. The stand's code
> (`evals/enterpriserag/`) is removed from the repository; the numbers recorded in the v2 spec stay as the record of
> what was measured. Retrieval decisions are measured on the owner's private Russian ruler only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure minirag on EnterpriseRAG-Bench end to end (two `claude -p` arms, Codex-backed
judge, calibration against published scores), retrieval-only, and as a regression check.

**Architecture:** A development stand under `evals/enterpriserag/`, mirroring `evals/behaviour/`:
small single-purpose scripts around the core (`results.run_query`, the `minirag-mcp sync` CLI)
and around the benchmark's own scorer, which runs unmodified in its own venv with an LLM shim
injected at runtime.

**Tech Stack:** Python 3.11+, uv, minirag-mcp core, `claude -p` (Claude Code), `codex exec`
(Codex CLI, ChatGPT account), EnterpriseRAG-Bench at commit `d36685e273713975ee20299bbf1ab64165575b3c`.

**Spec:** `docs/superpowers/specs/2026-11-18-enterpriserag-bench-design.md`

**Execution:** the owner delegated the whole run ("do it without me, just show results"): no
review pause between pilot and full run; the pilot still runs first as a smoke test.

## Global Constraints

- English in code, comments, docs. Comments state constraints only.
- No benchmark data in the repository. Data lives in `ERB_DATA_DIR`, default
  `platformdirs.user_cache_path("minirag-mcp") / "enterpriserag"`.
- No change under `src/` for the benchmark's sake.
- Corpus: release `v1.0.0` `all_documents.zip` (txt export). Questions: `questions.jsonl` at the
  pinned commit (byte-identical to the leaderboard Space's `data/raw_data/questions.jsonl`).
- Answering model `claude-opus-5-5`, both arms, 600 s per question.
- Judge: Codex, isolated `CODEX_HOME` holding only a symlink to `~/.codex/auth.json` and a
  minimal `config.toml` (the owner's Codex config has plugins, MCP servers, hooks and memories
  that must not reach judge prompts). `gpt-5.4` is refused on a ChatGPT account.
- Tests: `uv run pytest evals/enterpriserag -q`; lint `uv run ruff check evals/enterpriserag`.

## Deviations from the spec, decided while planning

- The evaluator reads raw JSON documents from `generated_data/sources` (512k files, in git at
  the pinned commit), not the txt export: the benchmark checkout includes them.
- The reduced corpus is not committed as an id list: it is a deterministic function of the
  sorted corpus listing, the questions, a seed and a size; `baseline.json` records its hash.
- Step 0 (5,000-document timing) is replaced by watching the full sync's rate; stop only if
  it extrapolates past 24 h.

## Review Focus

1. Sync skips files without failing them (oversized, unreadable): the index report must count
   `scanned − ingested − skipped − moved` as unaccounted and exit non-zero. → Task 2.
2. A transcript with no `result` event (killed at 600 s, crashed): answered empty and counted
   as timeout/error, never dropped. → Task 4.
3. A `dsid_` in the body plus a citation block: the block wins; `Sources` lines are
   `displayPath`s with directories — the id is found inside a path. → Task 4.
4. Judge calls failing (rate limit, expired auth): the shim raises, counts failures, and the
   score report shows the count. → Task 6.
5. Resume after interruption: a partial transcript (no `result`) is re-run; a complete one is
   skipped. → Task 5.

---

### Task 1: common.py and subset.py

**Files:** Create `evals/enterpriserag/common.py`, `evals/enterpriserag/subset.py`,
`evals/enterpriserag/test_common.py`, `evals/enterpriserag/.gitignore` (`runs/`).

**Interfaces — Produces:**
- `DATA_DIR: Path`, `CORPUS = DATA_DIR/"corpus"`, `BENCH = DATA_DIR/"bench"`,
  `QUESTIONS = BENCH/"questions.jsonl"`, `RUNS = HERE/"runs"`
- `DSID_RE = re.compile(r"dsid_[0-9a-f]{32}")`
- `dsid_of(path: str) -> str | None` — from the base name's leading `dsid_…` only
- `@dataclass(frozen=True) Question(question_id, question_type, question, expected_doc_ids: tuple[str, ...])`
- `load_questions(path: Path = QUESTIONS) -> list[Question]`
- `pilot_ids(questions, per_type: int = 2) -> list[str]` — per type in file order, positions
  `int((i + 0.5) * n / per_type)`
- `outside_root(db: Path, root: Path) -> Path` — SystemExit when `db` is inside `root`
- `corpus_files(corpus: Path) -> list[str]` — sorted relative paths of `*.txt`
- `subset(files, questions, total: int, seed: int) -> list[str]` — every gold file plus a
  `random.Random(seed)` sample, proportional per top-level directory, up to `total`
- `materialize(corpus: Path, rel: list[str], dest: Path) -> None` — hard links, same tree
- `subset_hash(rel: list[str]) -> str` — sha256 of the newline-joined sorted list

- [ ] Tests: `dsid_of("slack/x/dsid_<32hex>_foo.txt") == "dsid_<32hex>"`, `None` for a name
  without the prefix and for a dsid only in a directory name; `pilot_ids` returns 2 per type,
  deterministic, all types; `subset` contains every gold file, has `total` items, is identical
  for the same seed and differs for another; `outside_root` refuses `root/x`.
- [ ] Run → FAIL, implement, run → PASS, commit.

### Task 2: index.py

**Files:** Create `evals/enterpriserag/index.py`, `evals/enterpriserag/test_index.py`.

**Interfaces — Produces:** `sync(server_bin: Path, base_dir: Path, db: Path, log: Path) -> dict`
(runs `minirag-mcp sync --base-dir --db-path --json`, stderr to `log`, returns
`{"counts", "errors", "seconds"}`); `unaccounted(counts: dict) -> int`
(`scanned − ingested − skipped − moved`); `write_report(result, db, out: Path) -> dict`
(adds chunk and source counts from `Store`, minirag version, git commit); CLI
`--corpus --db [--subset N --seed S]`.

- [ ] Tests: `unaccounted({"scanned": 10, "ingested": 7, "skipped": 1, "moved": 1, ...}) == 1`;
  CLI exits 1 when `failed > 0` or `unaccounted > 0` (fake server bin script printing JSON).
- [ ] Implement, PASS, commit.

### Task 3: recall.py

**Files:** Create `evals/enterpriserag/recall.py`, `evals/enterpriserag/test_recall.py`.

**Interfaces — Produces:** `ranked_docs(payload: dict) -> list[str]` (dsids from
`payload["sources"]` in order, `None`s dropped, deduplicated); `recall_at(ranked, gold, k) -> float`;
`reciprocal_rank(ranked, gold) -> float`; `measure(store, emb, cfg, questions) -> list[dict]`
(per question with gold: `run_query(top_k=100)` → `ranked[:20]`, and `run_query(top_k=8)` →
`default`); `summarize(rows) -> dict` (`overall` and `by_type`: `n`, `recall@5/10/20`,
`mrr`, `recall_default`); `fingerprint() -> dict`; `gate(cur, base) -> list[str]`
(fails on overall `recall@10` drop > 1.0 point or `mrr` drop > 0.02; per-type deltas never
gate). Recall values are percentages (0–100), MRR 0–1. CLI: `--db --corpus` (full, writes
`runs/<stamp>/recall.json`), or `--subset` (temp index of `subset(total=20000, seed=20261118)`,
compare `baseline.json`, `--update-baseline`).

- [ ] Tests: ranked dedup; `recall_at(["a","b"], ("b","c"), 1) == 0.0`, `(…, 2) == 50.0`;
  `reciprocal_rank(["a","b"], ("b",)) == 0.5`; gate fires at −1.1 pt, not at −0.9 pt, and not on
  a −10 pt per-type drop alone.
- [ ] Implement, PASS, commit.

### Task 4: answers.py

**Files:** Create `evals/enterpriserag/answers.py`, `evals/enterpriserag/test_answers.py`.

**Interfaces — Produces:** `final_result(transcript: Path) -> dict | None` (last
`{"type":"result"}` event); `tool_calls(transcript: Path) -> Counter`; `cited_ids(answer: str)
-> tuple[list[str], str]` — kind `"block"` | `"whole"` | `"none"`. Block = trailing lines, after
stripping blanks, that are list items (`-`, `*`, `•`, `1.`) or contain a dsid, plus one heading
line directly above them; ids from the block if any, else from the whole text;
`build(run_dir: Path, arm: str, ids: list[str]) -> dict` writes `run_dir/arm/answers.jsonl`
(every id; missing/timeout/error → `answer ""`, `document_ids []`) and returns stats
(`block`, `whole`, `none`, `timeout`, `error`, `missing`, `cost_usd`, `mean_seconds`, `tools`).

- [ ] Tests: block wins over a body dsid; a `Sources:` list of `displayPath`s with directories
  yields their dsids in order, deduplicated; no list → `"whole"`; nothing → `"none"`;
  transcript with no result event → `answer ""` and `timeout` counted when a `.timeout`
  marker exists, `error` otherwise.
- [ ] Implement, PASS, commit.

### Task 5: run.py

**Files:** Create `evals/enterpriserag/run.py`, `evals/enterpriserag/test_run.py`.

**Interfaces — Consumes:** Task 1 (`load_questions`, `pilot_ids`, `outside_root`), Task 4
(`final_result`). **Produces:** `knowledge_sources(bench: Path) -> str` (the
`## Knowledge Sources` section of `src/scripts/answer_generation/agents_example.md`);
`build_prompt(q, sources_note: str, arm: str, corpus: Path) -> str` (identical except one line
saying where the documents are: minirag tools vs the absolute corpus path);
`claude_command(arm, prompt, mcp_config, model, corpus) -> list[str]` — both arms:
`--output-format stream-json --verbose --setting-sources project --disable-slash-commands
--no-session-persistence --strict-mcp-config --model`; minirag arm: `--mcp-config`, `--tools ""`,
`--allowedTools mcp__minirag__<read tools>`; files arm: `--tools Read,Grep,Glob,Bash`,
`--add-dir <corpus>`, `--allowedTools Read,Grep,Glob,Bash(grep:*),Bash(find:*),Bash(ls:*),
Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(cat:*),Bash(sort:*),Bash(uniq:*)`;
`done(transcript) -> bool`; CLI `--arm --ids pilot|all|<comma list> --parallel --model --db
--corpus --server-bin --out`. Each run: fresh temp cwd, `MCP_TIMEOUT=120000`, 600 s limit,
`.timeout` marker on kill; prewarm the server once.

- [ ] Tests: minirag command has no built-in tools and only `mcp__minirag__` allowed; files
  command has no `--mcp-config` and no write tool allowed; prompts for the two arms differ in
  exactly one line; `done` false for a partial transcript, true with a result event.
- [ ] Implement, PASS, commit.

### Task 6: judge_codex.py and judged_eval.py

**Files:** Create `evals/enterpriserag/judge_codex.py`, `evals/enterpriserag/judged_eval.py`,
`evals/enterpriserag/test_judge.py`.

**Interfaces — Produces:** `flatten(messages) -> str` (`ROLE:\ncontent` blocks);
`codex_command(model, effort, out_file) -> list[str]` (`codex exec --ephemeral
--skip-git-repo-check -s read-only -m <model> -c model_reasoning_effort="<effort>" -o <out> -`);
`class CodexLLM(model, effort)` with `generate(messages) -> Iterator[str]` (prompt on stdin,
fresh temp cwd, `CODEX_HOME` from env, 300 s timeout, 3 attempts, then `RuntimeError`);
module counters `CALLS`, `FAILURES`; `install(model, cheap_model, effort)` replaces `get_llm`
and `get_cheap_llm` on every loaded `src.*` module that has them; `judged_eval.py
{metrics|comparative} --model M --cheap-model C -- <evaluator args>` runs inside the bench venv
with cwd = bench, installs the shim, runs the evaluator's `main()`, writes
`<results>.judge.json` with model, calls, failures.

- [ ] Tests (fake `codex` script on `PATH` that writes a fixed reply to the `-o` file, or exits
  1): reply returned; three failures → `RuntimeError` and `FAILURES == 1`; `flatten` keeps
  order and roles.
- [ ] Implement, PASS, commit.

### Task 7: score.py and report.py

**Files:** Create `evals/enterpriserag/score.py`, `evals/enterpriserag/report.py`,
`evals/enterpriserag/test_score.py`.

**Interfaces — Produces:** `overall(results: dict) -> float` = mean over `questions` of
`(1 if answer_correct else 0) * completeness_pct`; `agreement(ours: dict, published: dict,
ids) -> dict` (`correct_agree_pct`, `completeness_mean_diff`, `n`); `score.py metrics|comparative|
calibrate` driving `judged_eval.py` with `ERB_JUDGE_MODEL` (default `gpt-5.6-sol`) and the bench
venv; `calibrate` fetches `answers_<key>.jsonl` and `results_<key>.json` for `bm25`,
`openai_filesearch`, `haiku_rag` from the leaderboard Space, filters to the ids, scores them;
`report.py <run_dir>` writes `REPORT.md` + `summary.json`: our rows inserted into the
leaderboard snapshot by overall, comparative result, recall, index report, judge, costs.

- [ ] Tests: `overall` on two questions (correct 80 %, wrong 100 %) == 40.0; `agreement` counts.
- [ ] Implement, PASS, commit.

### Task 8: docs

**Files:** Create `evals/enterpriserag/README.md`; modify `CLAUDE.md` Build & Test (one line for
the regression check); README "Benchmark" section after results exist.

---

## Execution (operational, after Tasks 1–8)

- E1 unzip corpus to `DATA_DIR/corpus`; full sync in background; index report.
- E2 retrieval-only recall on the full index; subset baseline → `baseline.json`.
- E3 pilot (20 ids) both arms; answers; calibration on the pilot ids; pick the judge model.
- E4 full 500 both arms (parallel 6).
- E5 metrics per arm, comparative, overall; report; README section; results page.
