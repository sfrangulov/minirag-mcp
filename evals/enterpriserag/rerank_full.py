"""A cross-encoder over the first-stage chunks, on the full corpus (issue g7q.7).

For each question: store.search(top_k=50) with production fusion, then the 50 chunk texts
are rescored by a reranker behind an OpenAI-style /v1/rerank endpoint (ERB_RERANK_URL,
e.g. llama.cpp serving bge-reranker-v2-m3). Documents are ranked by first occurrence in
the reranked chunks, then the rest of the saved deep ranking. Optionally fused with the
whole-document BM25 table from doc_bm25.py.

  ERB_RERANK_URL=http://127.0.0.1:8091/v1/rerank uv run python evals/enterpriserag/rerank_full.py \
      evals/enterpriserag/runs/<hybrid stamp>/recall.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from common import CORPUS, dsid_of, load_questions  # noqa: E402
from doc_bm25 import KS, docs_db, recall_at, rr, rrf  # noqa: E402
from recall import _open  # noqa: E402


def main() -> None:
    run = json.load(open(sys.argv[1]))
    rows = run["rows"]
    db = Path(os.environ.get("ERB_INDEX", str(CORPUS.parent / "index" / "lancedb")))
    cfg, emb, store = _open(db, CORPUS)
    questions = {q.question_id: q for q in load_questions()}
    url = os.environ["ERB_RERANK_URL"]
    s = requests.Session()
    import lancedb

    dtab = lancedb.connect(str(docs_db())).open_table("docs")
    out = []
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        text = questions[r["question_id"]].question
        res = store.search(text, emb.embed_query(text), top_k=50, hybrid_weight=cfg.hybrid_weight)
        resp = s.post(url, json={"query": text, "documents": [x.text for x in res]}, timeout=300)
        resp.raise_for_status()
        order = sorted(resp.json()["results"], key=lambda x: -x["relevance_score"])
        reranked = list(
            dict.fromkeys(d for d in (dsid_of(res[x["index"]].source) for x in order) if d)
        )
        first = list(dict.fromkeys(d for d in (dsid_of(x.source) for x in res) if d))
        try:
            hits = (
                dtab.search(text, query_type="fts", fts_columns=["text", "title"])
                .limit(100)
                .to_list()
            )
        except Exception:
            hits = []
        hits.sort(key=lambda x: (-x["_score"], x["dsid"]))
        df = [h["dsid"] for h in hits]
        out.append(
            {
                "gold": r["gold"],
                "type": r["question_type"],
                "deep": r["deep"],
                "first50": first,
                "rr": reranked + [d for d in r["deep"] if d not in reranked],
                "df": df,
            }
        )
        if n % 50 == 0:
            print(n, f"{(time.time() - t0) / n:.2f} s/question", flush=True)
    variants = {
        "hybrid (saved deep)": lambda o: o["deep"],
        "hybrid, rerank top 50 chunks": lambda o: o["rr"],
        "doc BM25": lambda o: o["df"],
        "hybrid + doc BM25 .5": lambda o: rrf([o["deep"], o["df"]], [0.5, 0.5]),
        "rerank + doc BM25 .5": lambda o: rrf([o["rr"], o["df"]], [0.5, 0.5]),
        "rerank + doc BM25 .3": lambda o: rrf([o["rr"], o["df"]], [0.7, 0.3]),
    }
    n = len(out)
    for name, fn in variants.items():
        line = " ".join(
            f"@{k} {sum(recall_at(fn(o), o['gold'], k) for o in out) / n:5.1f}" for k in KS
        )
        mrr = sum(rr(fn(o), o["gold"]) for o in out) / n
        sem = [o for o in out if o["type"] == "semantic"]
        print(
            f"{name:30} {line}  MRR {mrr:.3f}  semantic@10 "
            f"{sum(recall_at(fn(o), o['gold'], 10) for o in sem) / len(sem):.1f}"
        )


if __name__ == "__main__":
    main()
