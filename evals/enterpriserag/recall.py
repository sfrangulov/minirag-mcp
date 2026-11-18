"""Retrieval-only document recall, through the code path `query_documents` uses.

    uv run python evals/enterpriserag/recall.py --full            # the full index
    uv run python evals/enterpriserag/recall.py --subset          # regression check
    uv run python evals/enterpriserag/recall.py --subset --update-baseline

`topK` counts chunks; the leaderboard's one-shot baselines count documents (their vector
baseline fetches 100 chunks and keeps the first 10 distinct documents). Recall@k here
is the first k distinct documents of a 100-chunk query, so the two compare.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from common import CORPUS, HERE, INDEX, RUNS, corpus_files, dsid_of, load_questions, outside_root
from subset import SEED, TOTAL, materialize, subset, subset_hash

BASELINE = HERE / "baseline.json"
KS = (5, 10, 20)
RECALL_DROP = 1.0
MRR_DROP = 0.02


def ranked_docs(payload: dict) -> list[str]:
    seen: dict[str, None] = {}
    for s in payload["sources"]:
        d = dsid_of(s["source"])
        if d:
            seen.setdefault(d, None)
    return list(seen)


def recall_at(ranked: list[str], gold, k: int) -> float:
    gold = set(gold)
    return 100.0 * len(gold & set(ranked[:k])) / len(gold)


def reciprocal_rank(ranked: list[str], gold) -> float:
    gold = set(gold)
    return next((1.0 / (i + 1) for i, d in enumerate(ranked) if d in gold), 0.0)


def measure(store, emb, cfg, questions) -> list[dict]:
    from minirag_mcp.results import run_query

    rows = []
    for q in questions:
        if not q.expected_doc_ids:
            continue
        wide = ranked_docs(run_query(store, emb, cfg, q.question, top_k=100))
        default = ranked_docs(run_query(store, emb, cfg, q.question, top_k=8))
        rows.append(
            {
                "question_id": q.question_id,
                "question_type": q.question_type,
                "gold": list(q.expected_doc_ids),
                "ranked": wide[: max(KS)],
                "default": default,
            }
        )
    return rows


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    out: dict = {"n": n}
    for k in KS:
        out[f"recall@{k}"] = round(sum(recall_at(r["ranked"], r["gold"], k) for r in rows) / n, 2)
    out["mrr"] = round(sum(reciprocal_rank(r["ranked"], r["gold"]) for r in rows) / n, 4)
    out["recall_default_call"] = round(
        sum(recall_at(r["default"], r["gold"], len(r["default"]) or 1) for r in rows) / n, 2
    )
    return out


def summarize(rows: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    return {
        "overall": _metrics(rows),
        "by_type": {t: _metrics(rs) for t, rs in sorted(by_type.items())},
    }


def gate(current: dict, baseline: dict) -> list[str]:
    cur, base = current["overall"], baseline["overall"]
    failures = []
    if base["recall@10"] - cur["recall@10"] > RECALL_DROP:
        failures.append(f"recall@10 {base['recall@10']} -> {cur['recall@10']}")
    if base["mrr"] - cur["mrr"] > MRR_DROP:
        failures.append(f"mrr {base['mrr']} -> {cur['mrr']}")
    return failures


def fingerprint(cfg) -> dict:
    return {
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "onnxruntime": version("onnxruntime"),
        "fastembed": version("fastembed"),
        "lancedb": version("lancedb"),
        "model": cfg.model_name,
        "minirag": version("minirag-mcp"),
    }


def _open(db: Path, corpus: Path):
    from minirag_mcp.config import load_config
    from minirag_mcp.embedder import Embedder
    from minirag_mcp.store import Store

    cfg = load_config({"BASE_DIR": str(corpus), "DB_PATH": str(db)})
    emb = Embedder(cfg.model_name, cfg.cache_dir)
    store = Store(cfg.db_path, dim=emb.dim, fts_language=cfg.fts_language)
    return cfg, emb, store


def _print(summary: dict, baseline: dict | None = None) -> None:
    def line(name: str, m: dict, b: dict | None) -> str:
        cells = []
        for key in ("recall@5", "recall@10", "recall@20", "mrr", "recall_default_call"):
            delta = f" ({m[key] - b[key]:+.2f})" if b and key in b else ""
            cells.append(f"{key} {m[key]}{delta}")
        return f"{name:<28} n={m['n']:<4} " + "  ".join(cells)

    print(line("overall", summary["overall"], baseline and baseline["overall"]))
    for t, m in summary["by_type"].items():
        print(line(t, m, baseline and baseline["by_type"].get(t)))


def run_full(db: Path, corpus: Path) -> int:
    cfg, emb, store = _open(outside_root(db, corpus), corpus)
    rows = measure(store, emb, cfg, load_questions())
    summary = summarize(rows)
    out = RUNS / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "recall.json").write_text(
        json.dumps({"summary": summary, "fingerprint": fingerprint(cfg), "rows": rows}, indent=1),
        encoding="utf-8",
    )
    _print(summary)
    print(f"\nwritten to {out / 'recall.json'}")
    return 0


def run_subset(corpus: Path, update: bool) -> int:
    from minirag_mcp.ingest.pipeline import Pipeline
    from minirag_mcp.sync import run_sync

    questions = load_questions()
    rel = subset(corpus_files(corpus), questions, total=TOTAL, seed=SEED)
    work = Path(tempfile.mkdtemp(prefix="minirag-erb-subset-"))
    root, db = work / "corpus", work / "lancedb"
    materialize(corpus, rel, root)
    cfg, emb, store = _open(db, root)
    counts, errors = run_sync(Pipeline(store, emb, cfg), store, cfg.roots, cfg.max_file_size)
    if errors or counts["failed"] or counts["ingested"] != len(rel):
        print(f"subset index incomplete: {counts}, {errors[:3]}", file=sys.stderr)
        return 1
    summary = summarize(measure(store, emb, cfg, questions))
    current = {"summary": summary, "fingerprint": fingerprint(cfg), "subset": subset_hash(rel)}
    if update or not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        _print(summary)
        print(f"\nbaseline written to {BASELINE}")
        return 0
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    if base["subset"] != current["subset"]:
        print("warning: the subset differs from the baseline's; compare with care")
    if base["fingerprint"] != current["fingerprint"]:
        print(f"warning: environment differs from the baseline's: {base['fingerprint']}")
    _print(summary, base["summary"])
    failures = gate(summary, base["summary"])
    for f in failures:
        print(f"REGRESSION: {f}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--subset", action="store_true")
    ap.add_argument("--db", type=Path, default=INDEX)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()
    if args.full:
        return run_full(args.db, args.corpus)
    return run_subset(args.corpus, args.update_baseline)


if __name__ == "__main__":
    sys.exit(main())
