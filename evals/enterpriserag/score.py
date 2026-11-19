"""Score answer files with the benchmark's own evaluators and the Codex judge.

    uv run python evals/enterpriserag/score.py metrics <run_dir> <arm>
    uv run python evals/enterpriserag/score.py comparative <run_dir>
    uv run python evals/enterpriserag/score.py calibrate <run_dir>

The evaluators run unmodified, in the benchmark's venv. They skip a row with neither an
answer nor documents and average over the rest, so a timed-out question would raise the
average; `aggregate` counts every question of the run, an unscored one as zero.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from common import BENCH, DATA_DIR, HERE, QUESTIONS, SPACE_RAW, load_questions

JUDGE_MODEL = os.environ.get("ERB_JUDGE_MODEL", "gpt-5.6-sol")
JUDGE_CHEAP = os.environ.get("ERB_JUDGE_CHEAP", "gpt-5.6-luna")
CODEX_HOME = DATA_DIR / "codex-home"
PARALLEL = int(os.environ.get("ERB_JUDGE_PARALLEL", "8"))
CALIBRATION_SYSTEMS = ("bm25", "openai_filesearch", "haiku_rag")


def overall(results: dict, ids: list[str]) -> float:
    by_id = {r["question_id"]: r for r in results.get("questions", [])}
    total = 0.0
    for qid in ids:
        r = by_id.get(qid)
        if r:
            total += (1 if r.get("answer_correct") else 0) * (r.get("completeness_pct") or 0)
    return round(total / len(ids), 2)


def aggregate(results: dict, ids: list[str], gold: dict[str, tuple]) -> dict:
    by_id = {r["question_id"]: r for r in results.get("questions", [])}
    n = len(ids)
    with_gold = [q for q in ids if gold.get(q)]
    recall = [(by_id.get(q) or {}).get("document_recall_pct") or 0 for q in with_gold]
    extras = [(by_id.get(q) or {}).get("invalid_extra_docs") or 0 for q in with_gold]
    return {
        "n": n,
        "scored": sum(1 for q in ids if q in by_id),
        "overall": overall(results, ids),
        "correctness": round(
            100 * sum(1 for q in ids if (by_id.get(q) or {}).get("answer_correct")) / n, 2
        ),
        "completeness": round(
            sum((by_id.get(q) or {}).get("completeness_pct") or 0 for q in ids) / n, 2
        ),
        "recall": round(sum(recall) / len(recall), 2) if recall else None,
        "invalid_extra_docs": round(sum(extras) / len(extras), 2) if extras else None,
        "corrected": sum(1 for q in ids if (by_id.get(q) or {}).get("corrected")),
    }


def agreement(ours: dict, published: dict, ids: list[str]) -> dict:
    a = {r["question_id"]: r for r in ours.get("questions", [])}
    b = {r["question_id"]: r for r in published.get("questions", [])}
    both = [q for q in ids if q in a and q in b]
    if not both:
        return {"n": 0}
    agree = sum(
        1 for q in both if bool(a[q].get("answer_correct")) == bool(b[q].get("answer_correct"))
    )
    diffs = [
        (a[q].get("completeness_pct") or 0) - (b[q].get("completeness_pct") or 0) for q in both
    ]
    return {
        "n": len(both),
        "correct_agree_pct": round(100 * agree / len(both), 1),
        "completeness_mean_diff": round(sum(diffs) / len(both), 2),
        "ours_overall": overall(ours, both),
        "published_overall": overall(published, both),
    }


def _codex_home() -> Path:
    CODEX_HOME.mkdir(parents=True, exist_ok=True)
    auth = CODEX_HOME / "auth.json"
    if not auth.exists():
        auth.symlink_to(Path.home() / ".codex" / "auth.json")
    (CODEX_HOME / "config.toml").write_text('model_reasoning_effort = "medium"\n')
    return CODEX_HOME


def judged(which: str, stats: Path, evaluator_args: list[str]) -> int:
    cmd = [
        str(BENCH / ".venv" / "bin" / "python"),
        str(HERE / "judged_eval.py"),
        which,
        "--model",
        JUDGE_MODEL,
        "--cheap-model",
        JUDGE_CHEAP,
        "--stats",
        str(stats),
        "--",
        *evaluator_args,
        "--questions-file",
        str(QUESTIONS),
        "--uuid-index-cache-file",
        str(BENCH / "generated_data" / "uuid_index.json"),
        "--parallelism",
        str(PARALLEL),
        "--resume",
    ]
    # Any call that slipped past the swapped factories fails instead of reaching OpenAI.
    env = dict(os.environ, CODEX_HOME=str(_codex_home()), LLM_PROVIDER="codex-only")
    return subprocess.run(cmd, cwd=BENCH, env=env).returncode


def metrics(run_dir: Path, arm: str) -> int:
    d = run_dir / arm
    return judged(
        "metrics",
        d / "judge_stats.json",
        [
            "--answers-file",
            str(d / "answers.jsonl"),
            "--results-file",
            str(d / "results.json"),
            "--updated-questions-file",
            str(d / "questions_updated.jsonl"),
        ],
    )


def comparative(run_dir: Path) -> int:
    return judged(
        "comparative",
        run_dir / "comparative_judge_stats.json",
        [
            "--answer-file-1",
            str(run_dir / "minirag" / "answers.jsonl"),
            "--answer-file-2",
            str(run_dir / "files" / "answers.jsonl"),
            "--results-file",
            str(run_dir / "comparative.json"),
            "--updated-questions-file",
            str(run_dir / "comparative_questions_updated.jsonl"),
        ],
    )


def _space(name: str) -> Path:
    path = DATA_DIR / "space" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{SPACE_RAW}/raw_data/{name}", path)
    return path


def calibrate(run_dir: Path) -> int:
    ids = json.loads((run_dir / "ids.json").read_text(encoding="utf-8"))
    wanted = set(ids)
    report = {}
    for key in CALIBRATION_SYSTEMS:
        d = run_dir / "calibration" / key
        d.mkdir(parents=True, exist_ok=True)
        rows = [
            line
            for line in _space(f"answers_{key}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("question_id") in wanted
        ]
        (d / "answers.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        code = metrics(run_dir / "calibration", key)
        published = json.loads(_space(f"results_{key}.json").read_text(encoding="utf-8"))
        ours = json.loads((d / "results.json").read_text(encoding="utf-8"))
        report[key] = {"exit": code, **agreement(ours, published, ids)}
        print(key, report[key], flush=True)
    (run_dir / "calibration" / "agreement.json").write_text(json.dumps(report, indent=2))
    return 0


def summarize(run_dir: Path, arm: str) -> dict:
    ids = json.loads((run_dir / "ids.json").read_text(encoding="utf-8"))
    gold = {q.question_id: q.expected_doc_ids for q in load_questions()}
    results = json.loads((run_dir / arm / "results.json").read_text(encoding="utf-8"))
    out = aggregate(results, ids, gold)
    (run_dir / arm / "score.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    which, run_dir = sys.argv[1], Path(sys.argv[2]).resolve()
    if which == "metrics":
        arm = sys.argv[3]
        code = metrics(run_dir, arm)
        print(json.dumps(summarize(run_dir, arm), indent=2))
        return code
    if which == "comparative":
        return comparative(run_dir)
    if which == "calibrate":
        return calibrate(run_dir)
    raise SystemExit(f"unknown command {which}")


if __name__ == "__main__":
    sys.exit(main())
