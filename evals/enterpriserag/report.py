"""Put a run's numbers next to the leaderboard's.

    uv run python evals/enterpriserag/report.py <run_dir> [--recall <recall.json>]

Writes <run_dir>/summary.json and <run_dir>/REPORT.md. The leaderboard rows come from the
Space's leaderboard.csv as fetched (its date is in the report); our rows are inserted by
overall score and marked.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from common import DATA_DIR, SPACE_RAW

ARM_NAMES = {
    "minirag": "Claude Code (Opus 5.5) + minirag-mcp",
    "files": "Claude Code (Opus 5.5) + built-in file tools (control)",
}


def leaderboard() -> tuple[list[dict], str]:
    path = DATA_DIR / "space" / "leaderboard.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{SPACE_RAW}/final_display_data/leaderboard.csv", path)
    fetched = datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%d")
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    return rows, fetched


def on_questions(ids: list[str]) -> list[dict]:
    """Every published system scored on these questions, from its per-question results."""
    import re

    from common import load_questions
    from score import aggregate

    space = DATA_DIR / "space"
    names = dict(
        re.findall(
            r'^([a-z0-9_]+):\s*\n\s+display_name:\s*"([^"]+)"',
            (space / "systems.yaml").read_text(encoding="utf-8"),
            re.M,
        )
    )
    gold = {q.question_id: q.expected_doc_ids for q in load_questions()}
    rows = []
    for key, name in names.items():
        results = json.loads((space / f"results_{key}.json").read_text(encoding="utf-8"))
        a = aggregate(results, ids, gold)
        rows.append(
            {
                "model": name,
                "overall_score": a["overall"],
                "correctness": a["correctness"],
                "completeness": a["completeness"],
                "recall": a["recall"],
                "invalid_extra_docs": a["invalid_extra_docs"],
            }
        )
    return rows


def _f(x) -> str:
    return "—" if x is None or x == "" else f"{float(x):.2f}"


def table(ours: dict[str, dict], board: list[dict]) -> str:
    rows = [
        {
            "name": r["model"],
            "overall": float(r["overall_score"]),
            "correctness": r["correctness"],
            "completeness": r["completeness"],
            "recall": r["recall"],
            "extra": r["invalid_extra_docs"],
            "ours": False,
        }
        for r in board
    ]
    for arm, s in ours.items():
        rows.append(
            {
                "name": f"**{ARM_NAMES[arm]}**",
                "overall": s["overall"],
                "correctness": s["correctness"],
                "completeness": s["completeness"],
                "recall": s["recall"],
                "extra": s["invalid_extra_docs"],
                "ours": True,
            }
        )
    rows.sort(key=lambda r: -r["overall"])
    out = [
        "| # | System | Overall | Correct | Complete | Doc recall | Invalid extras |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        out.append(
            f"| {i} | {r['name']} | {_f(r['overall'])} | {_f(r['correctness'])} | "
            f"{_f(r['completeness'])} | {_f(r['recall'])} | {_f(r['extra'])} |"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--recall", type=Path)
    args = ap.parse_args()
    run = args.run_dir
    ours = {
        arm: json.loads((run / arm / "score.json").read_text())
        for arm in ARM_NAMES
        if (run / arm / "score.json").exists()
    }
    stats = {
        arm: json.loads((run / arm / "answers_stats.json").read_text())
        for arm in ARM_NAMES
        if (run / arm / "answers_stats.json").exists()
    }
    judge = {
        arm: json.loads((run / arm / "judge_stats.json").read_text())
        for arm in ARM_NAMES
        if (run / arm / "judge_stats.json").exists()
    }
    board, fetched = leaderboard()
    ids = json.loads((run / "ids.json").read_text(encoding="utf-8"))
    summary = {"ours": ours, "answers": stats, "judge": judge, "leaderboard_fetched": fetched}
    parts = [f"# EnterpriseRAG-Bench run `{run.name}`", ""]
    if len(ids) < 500:
        parts += [
            f"On this run's {len(ids)} questions, every published system rescored from its "
            "published per-question results:",
            "",
            table(ours, on_questions(ids)),
            "",
        ]
    parts += ["Full leaderboard (all 500 questions):", "", table(ours, board), ""]
    parts.append(f"Leaderboard rows as fetched on {fetched}; ours in bold.")
    for name in ("comparative.json", "calibration/agreement.json"):
        if (run / name).exists():
            summary[name] = json.loads((run / name).read_text())
    if args.recall:
        summary["recall"] = json.loads(args.recall.read_text())["summary"]
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    parts += ["", "```json", json.dumps({k: v for k, v in summary.items()}, indent=1), "```"]
    (run / "REPORT.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("\n".join(parts[:4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
