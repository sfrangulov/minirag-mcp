"""Confirm each item is built the way it claims, against the real retriever, before any
model sees it. Needs the embedding model (cached after the first run); no Claude call.

    uv run python evals/behaviour/check_items.py [--top-k 8] [--items R1,R2]

What is checked, per group:

- reading: for every probe, the `fact` passage is absent from every hit's `text` and
  present in `parents`. Otherwise a run cannot tell answering from the section apart
  from answering from the snippet, and the item measures nothing.
- thin (W2): the literal probes do not rank the allowed document first; the good probes
  do. Otherwise the item does not need a reworded query.
- thin (W1): printed only — the top hits, to read that nothing in them is the answer.
- search, injection: an allowed document ranks first for every probe.

Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

from items import CORPUS, ITEMS_FILE, existing_index, load_items, select

from minirag_mcp.config import load_config
from minirag_mcp.embedder import Embedder
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.results import run_query
from minirag_mcp.store import Store
from minirag_mcp.sync import run_sync


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--items")
    ap.add_argument("--items-file", type=Path, default=ITEMS_FILE)
    ap.add_argument("--base-dir", type=Path, default=CORPUS)
    ap.add_argument("--db", type=Path, help="an existing index (not inside --base-dir); no sync")
    args = ap.parse_args()

    if args.db:
        db = existing_index(args.db, args.base_dir)
    else:
        db = Path(tempfile.mkdtemp(prefix="minirag-check-items-")) / "db"
    cfg = load_config({"BASE_DIR": str(args.base_dir), "DB_PATH": str(db)})
    emb = Embedder(cfg.model_name, cfg.cache_dir)
    store = Store(cfg.db_path, dim=emb.dim)
    pipeline = Pipeline(store, emb, cfg)
    if not args.db:
        counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
        print(f"indexed: {counts}  chunks: {store.chunk_count()}")
        if errors:
            print(f"sync errors: {errors}")
            return 1

    def search(q: str):
        payload = run_query(store, emb, cfg, q, top_k=args.top_k)
        return payload, [s["displayPath"] for s in payload["sources"]]

    failed = []
    for item in select(load_items(args.items_file), args.items):
        probes = item.probes
        if not probes:
            continue
        print(f"\n{item.id} [{item.group}]")
        for q in probes:
            payload, docs = search(q)
            rank = next((i + 1 for i, d in enumerate(docs) if d in item.allowed), None)
            line = f"  {q!r}: allowed doc rank {rank} of {len(docs)} docs"
            if item.group == "reading":
                fact = re.compile(item.fact, re.I)
                in_text = any(fact.search(r["text"]) for r in payload["results"])
                in_parent = any(fact.search(t) for t in payload["parents"].values())
                ok = not in_text and in_parent
                line += f"; fact in snippet: {in_text}, in parents: {in_parent}"
            elif item.id == "W1":
                ok = True
                top = payload["results"][:3]
                line += "\n" + "\n".join(f"      {r['text'][:110]!r}" for r in top)
            elif item.group == "thin":
                ok = rank != 1
            else:
                ok = rank == 1
            print(("  ok  " if ok else "  FAIL") + line[1:])
            if not ok:
                failed.append((item.id, q))
        for q in item.good_probes:
            _, docs = search(q)
            rank = next((i + 1 for i, d in enumerate(docs) if d in item.allowed), None)
            ok = rank == 1
            print(("  ok  " if ok else "  FAIL") + f" good {q!r}: allowed doc rank {rank}")
            if not ok:
                failed.append((item.id, q))

    print(f"\n{len(failed)} failing check(s)" + (f": {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
