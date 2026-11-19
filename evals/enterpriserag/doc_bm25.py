"""Whole-document BM25 as an extra fusion signal, on the full corpus (issue g7q.4).

Builds one row per document (title from the file name + full text) with an English FTS
index, then fuses its ranking with the document rankings saved by recall.py full runs
(rows[].deep: top 100 documents from 1000 fused chunks) by weighted RRF (k=60):

  uv run python evals/enterpriserag/doc_bm25.py build
  uv run python evals/enterpriserag/doc_bm25.py measure runs/<hybrid stamp>/recall.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import lancedb
from lancedb.index import FTS

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA_DIR, dsid_of, load_questions  # noqa: E402

K = 60
KS = (5, 10, 20, 50)


def docs_db() -> Path:
    return DATA_DIR / "doc_bm25"


def build() -> None:
    corpus = DATA_DIR / "corpus"
    db = lancedb.connect(str(docs_db()))
    batch, table = [], None
    for n, path in enumerate(sorted(corpus.rglob("*.txt")), 1):
        dsid = dsid_of(str(path))
        if not dsid:
            continue
        slug = path.stem.split("__", 1)[-1].replace("-", " ")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        batch.append({"dsid": dsid, "title": slug, "text": text})
        if len(batch) == 20_000:
            table = _flush(db, table, batch)
            batch = []
            print(n, flush=True)
    table = _flush(db, table, batch)
    for col in ("text", "title"):
        table.create_index(col, config=FTS(language="English"), replace=True)
    print("rows", table.count_rows())


def _flush(db, table, batch):
    if not batch:
        return table
    if table is None:
        return db.create_table("docs", batch, mode="overwrite")
    table.add(batch)
    return table


def rrf(lists, weights):
    sc: dict[str, float] = defaultdict(float)
    for lst, w in zip(lists, weights, strict=True):
        for i, d in enumerate(lst):
            sc[d] += w / (K + i + 1)
    return [d for d, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))]


def recall_at(ranked, gold, k):
    return 100.0 * len(set(gold) & set(ranked[:k])) / len(gold)


def rr(ranked, gold):
    g = set(gold)
    return next((1 / (i + 1) for i, d in enumerate(ranked) if d in g), 0.0)


def measure(run: Path) -> None:
    rows = json.load(open(run))["rows"]
    questions = {q.question_id: q for q in load_questions()}
    table = lancedb.connect(str(docs_db())).open_table("docs")
    df = {}
    for r in rows:
        text = questions[r["question_id"]].question
        try:
            res = (
                table.search(text, query_type="fts", fts_columns=["text", "title"])
                .limit(100)
                .to_list()
            )
        except Exception:
            res = []
        res.sort(key=lambda x: (-x["_score"], x["dsid"]))
        df[r["question_id"]] = [x["dsid"] for x in res]
    variants = {
        "hybrid (saved)": lambda r: r["deep"],
        "doc BM25": lambda r: df[r["question_id"]],
        "hybrid + doc BM25 .25": lambda r: rrf([r["deep"], df[r["question_id"]]], [0.75, 0.25]),
        "hybrid + doc BM25 .4": lambda r: rrf([r["deep"], df[r["question_id"]]], [0.6, 0.4]),
        "hybrid + doc BM25 .5": lambda r: rrf([r["deep"], df[r["question_id"]]], [0.5, 0.5]),
    }
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    for name, fn in variants.items():
        n = len(rows)
        line = " ".join(
            f"@{k} {sum(recall_at(fn(r), r['gold'], k) for r in rows) / n:5.1f}" for k in KS
        )
        mrr = sum(rr(fn(r), r["gold"]) for r in rows) / n
        types = " ".join(
            f"{t} {sum(recall_at(fn(r), r['gold'], 10) for r in rs) / len(rs):4.1f}"
            for t, rs in sorted(by_type.items())
        )
        print(f"{name:24} {line}  MRR {mrr:.3f} | @10 by type: {types}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        measure(Path(sys.argv[2]))
