"""Turn `claude -p` transcripts into the benchmark's answer file.

    uv run python evals/enterpriserag/answers.py <run_dir> <arm>

`document_ids` come from the answer's trailing citation block, whatever its heading:
minirag's own instructions ask for a Sources list of display paths, which carry the
id in the file name. The same rule applies to both arms. A question whose run left no
answer is written with an empty one and counted, never dropped.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from common import DSID_RE

_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_BARE = re.compile(r"^\s*`?\S*dsid_[0-9a-f]{32}\S*`?\s*$")


def cited_ids(answer: str) -> tuple[list[str], str]:
    lines = answer.rstrip().splitlines()
    block: list[str] = []
    i = len(lines) - 1
    while i >= 0 and (_ITEM.match(lines[i]) or _BARE.match(lines[i])):
        block.append(lines[i])
        i -= 1
    ids = list(dict.fromkeys(DSID_RE.findall("\n".join(reversed(block)))))
    if ids:
        return ids, "block"
    ids = list(dict.fromkeys(DSID_RE.findall(answer)))
    return (ids, "whole") if ids else ([], "none")


def _events(transcript: Path):
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def final_result(transcript: Path) -> dict | None:
    last = None
    for e in _events(transcript):
        if e.get("type") == "result":
            last = e
    return last


def tool_calls(transcript: Path) -> Counter:
    calls: Counter = Counter()
    for e in _events(transcript):
        if e.get("type") != "assistant":
            continue
        for block in e.get("message", {}).get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls[block.get("name", "?")] += 1
    return calls


def build(run_dir: Path, arm: str, ids: list[str]) -> dict:
    raw = run_dir / arm / "raw"
    stats: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    seconds: list[float] = []
    cost = 0.0
    rows = []
    for qid in ids:
        transcript = raw / f"{qid}.jsonl"
        answer, doc_ids = "", []
        if not transcript.exists():
            stats["missing"] += 1
        else:
            tools.update(tool_calls(transcript))
            result = final_result(transcript)
            if result is None:
                stats["timeout" if (raw / f"{qid}.timeout").exists() else "error"] += 1
            elif result.get("is_error") or not (result.get("result") or "").strip():
                stats["error"] += 1
            else:
                answer = result["result"]
                doc_ids, kind = cited_ids(answer)
                stats[kind] += 1
            if result is not None:
                cost += result.get("total_cost_usd") or 0.0
                seconds.append((result.get("duration_ms") or 0) / 1000)
        rows.append({"question_id": qid, "answer": answer, "document_ids": doc_ids})
    out = run_dir / arm / "answers.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary: dict = {
        k: stats.get(k, 0) for k in ("block", "whole", "none", "timeout", "error", "missing")
    }
    summary["answered"] = len(rows) - summary["timeout"] - summary["error"] - summary["missing"]
    summary["cost_usd"] = round(cost, 2)
    summary["mean_seconds"] = round(sum(seconds) / len(seconds), 1) if seconds else None
    summary["tools"] = dict(tools.most_common())
    (run_dir / arm / "answers_stats.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    run_dir, arm = Path(sys.argv[1]), sys.argv[2]
    ids = json.loads((run_dir / "ids.json").read_text(encoding="utf-8"))
    print(json.dumps(build(run_dir, arm, ids), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
