"""Build the full benchmark index in parallel, with rows identical to what `sync` writes.

    uv run python evals/enterpriserag/bulk_index.py --verify 300   # prove the equivalence
    uv run python evals/enterpriserag/bulk_index.py                # the full corpus

Stock `minirag-mcp sync` ingests one file at a time on one core: measured on this corpus
at 3.3 files/s, about 40 hours for 512k files, with a third of the time spent in two
LanceDB commits per file. Here worker processes run the real `Pipeline.ingest_file`
(parser, chunker, embedder untouched) against a store that only collects the rows, and
the parent writes them in large batches, signatures included.

`--verify N` builds the same N-file subset both ways and compares every chunk row, the
signatures, the documents rows, the recorded identity and probe, and the results of 40
queries under the production fusion (document weight included), then runs stock `sync`
over the bulk-built index and requires it to find nothing to do.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

from common import CORPUS, INDEX, core_env, corpus_files, load_questions, open_index, outside_root
from subset import materialize, subset

_pipeline = None


class _Collect:
    """The one store method `ingest_file` calls; the parent does the writing."""

    def __init__(self):
        self.records = None

    def replace_source(self, source, records):
        self.records = list(records)


def _init(corpus: str, db: str, threads: int, scheme, token_budget: int) -> None:
    """A worker embeds and chunks the way the index records, never the way the
    environment would now."""
    global _pipeline
    from dataclasses import replace

    from minirag_mcp.config import load_config
    from minirag_mcp.handle import make_embedder
    from minirag_mcp.ingest.pipeline import Pipeline

    cfg = replace(load_config(core_env(BASE_DIR=corpus, DB_PATH=db)), token_budget=token_budget)
    # Several processes share the cores; each onnxruntime session gets a slice of them.
    _pipeline = Pipeline(_Collect(), make_embedder(cfg, scheme, threads=threads), cfg)


def _ingest(path: str):
    try:
        _pipeline.ingest_file(Path(path), rules_checked=True)
    except Exception as e:
        return path, None, f"{type(e).__name__}: {e}"
    return path, _pipeline.store.records, None


def build(
    corpus: Path,
    db: Path,
    workers: int,
    threads: int,
    batch_files: int,
    log,
    fixed: dict[str, str] | None = None,
) -> dict:
    """`fixed` sets product variables over the environment's (CHUNK_TOKEN_BUDGET)."""
    from minirag_mcp.ingest.scanner import scan_roots

    corpus = corpus.resolve()
    handle = open_index(db, corpus, **(fixed or {}))
    # RebuildPendingError while the folder holds an index of another configuration.
    store = handle.ingest_pipeline().store
    index = handle.get()
    cfg = index.config
    done = {s.source for s in store.list_sources()} if store.chunk_count() else set()
    # The scanner's own walk, so the file set and the row order are the ones sync writes:
    # BM25 ties between near-duplicates break by row order.
    entries = scan_roots(list(cfg.roots))
    oversized = [str(e.path) for e in entries if e.size > cfg.max_file_size]
    todo = [str(e.path) for e in entries if str(e.path) not in done and e.size <= cfg.max_file_size]
    counts = {
        "already": len(done),
        "todo": len(todo),
        "oversized": len(oversized),
        "ingested": 0,
        "failed": 0,
        "chunks": 0,
    }
    errors = [{"source": p, "error": "larger than MAX_FILE_SIZE"} for p in oversized]
    started = time.monotonic()
    batch: list = []
    if todo:
        # Fixes the model a fresh index is built with before the first row is written.
        handle.ensure_probe()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers,
        initializer=_init,
        initargs=(
            str(corpus),
            str(db),
            threads,
            index.identity.embedding,
            index.identity.chunking.token_budget,
        ),
    ) as pool:
        for path, records, err in pool.imap(_ingest, todo, chunksize=8):
            if err:
                counts["failed"] += 1
                errors.append({"source": path, "error": err})
                continue
            batch.append((path, records))
            counts["ingested"] += 1
            counts["chunks"] += len(records)
            if len(batch) >= batch_files:
                store.add_sources(batch)
                batch = []
                rate = counts["ingested"] / (time.monotonic() - started)
                left = (len(todo) - counts["ingested"] - counts["failed"]) / rate / 3600
                print(
                    f"{counts['ingested']}/{len(todo)} files, {counts['chunks']} chunks, "
                    f"{rate:.1f} files/s, ~{left:.1f} h left, failed {counts['failed']}",
                    file=log,
                    flush=True,
                )
    if batch:
        store.add_sources(batch)
    counts["load_seconds"] = round(time.monotonic() - started, 1)
    # The documents table is written from the stored chunk text, as `sync` writes it.
    _, problems = store.backfill_documents()
    for problem in problems:
        errors.append(problem)
        counts["failed"] += 1
    if problem := store.update_indices():
        errors.append({"source": str(db), "error": problem})
        counts["failed"] += 1
    counts["seconds"] = round(time.monotonic() - started, 1)
    return {"counts": counts, "errors": errors}


def _rows(db: Path) -> dict:
    import lancedb

    table = lancedb.connect(str(db)).open_table("chunks")
    out = {}
    for r in table.to_arrow().to_pylist():
        out[r["id"]] = r
    return out


def _signatures(db: Path) -> dict:
    import lancedb

    conn = lancedb.connect(str(db))
    if "signatures" not in conn.table_names():
        return {}
    rows = conn.open_table("signatures").to_arrow().to_pylist()
    return {r["source"]: list(r["signature"]) for r in rows}


def _documents(db: Path) -> dict:
    import lancedb

    conn = lancedb.connect(str(db))
    if "documents" not in conn.table_names():
        return {}
    return {r["source"]: r for r in conn.open_table("documents").to_arrow().to_pylist()}


def compare_indexes(a_db: Path, b_db: Path) -> list[str]:
    """What differs between two indexes of the same corpus: chunk rows, look-alike
    signatures, the whole-document rows the document BM25 bonus ranks, the identity
    recorded in index.json, and the probe both must carry."""
    import numpy as np

    from minirag_mcp.identity import read_identity

    problems = []
    a, b = _rows(a_db), _rows(b_db)
    if a.keys() != b.keys():
        problems.append(f"row ids differ: {len(a.keys() - b.keys())} / {len(b.keys() - a.keys())}")
    skip = {"vector", "ingested_at"}
    for key in a.keys() & b.keys():
        ra, rb = a[key], b[key]
        diff = [c for c in ra if c not in skip and ra[c] != rb.get(c)]
        if diff:
            problems.append(f"{key}: columns differ {diff}")
        if not np.allclose(ra["vector"], rb["vector"], atol=1e-5):
            problems.append(f"{key}: vectors differ")
    sa, sb = _signatures(a_db), _signatures(b_db)
    if sa != sb:
        problems.append(f"signatures differ: {len(sa)} vs {len(sb)} sources")
    da, db_ = _documents(a_db), _documents(b_db)
    if not da:
        problems.append("the stock index has no documents rows")
    if da.keys() != db_.keys():
        problems.append(
            f"document rows differ: {len(da.keys() - db_.keys())} / {len(db_.keys() - da.keys())}"
        )
    for key in da.keys() & db_.keys():
        diff = [c for c in da[key] if c != "ingested_at" and da[key][c] != db_[key].get(c)]
        if diff:
            problems.append(f"{key}: document columns differ {diff}")
    ia, ib = read_identity(a_db), read_identity(b_db)
    if ia is None or ib is None:
        problems.append("an index has no index.json")
    else:
        if not ia.probe or not ib.probe:
            problems.append("an index records no probe")
        elif not np.allclose(ia.probe, ib.probe, atol=1e-5):
            problems.append("probes differ")
        for field in ("format", "embedding", "chunking", "fts_language", "document_table"):
            if getattr(ia, field) != getattr(ib, field):
                problems.append(
                    f"identity {field} differs: {getattr(ia, field)} / {getattr(ib, field)}"
                )
    return problems


def verify(corpus: Path, n: int, workers: int, threads: int) -> int:
    from minirag_mcp.results import run_query

    questions = load_questions()
    rel = subset(corpus_files(corpus), questions[:5], total=n, seed=7)
    work = Path(tempfile.mkdtemp(prefix="minirag-erb-verify-"))
    root = work / "corpus"
    materialize(corpus, rel, root)
    root = root.resolve()

    handle_a = open_index(work / "a", root)
    counts, errs, _ = handle_a.run_sync(None)
    index_a = handle_a.get()
    cfg, emb, store_a = index_a.config, index_a.embedder, index_a.store
    print(f"stock sync: {counts} errors {len(errs)}")
    result = build(root, work / "b", workers, threads, batch_files=50, log=sys.stdout)
    print(f"bulk build: {result['counts']} errors {len(result['errors'])}")

    problems = compare_indexes(work / "a", work / "b")
    a = _rows(work / "a")
    handle_b = open_index(work / "b", root)
    index_b = handle_b.get()
    emb_b, store_b = index_b.embedder, index_b.store
    for q in questions[:40]:
        ra = [s["source"] for s in run_query(store_a, emb, cfg, q.question, top_k=50)["sources"]]
        rb = [s["source"] for s in run_query(store_b, emb_b, cfg, q.question, top_k=50)["sources"]]
        if ra != rb:
            problems.append(f"{q.question_id}: ranked sources differ")

    resync, errs, _ = handle_b.run_sync(None)
    if resync["ingested"] or resync["deleted"] or resync["moved"] or resync["skipped"] != len(rel):
        problems.append(f"stock sync over the bulk index was not a no-op: {resync}")

    for p in problems[:30]:
        print("MISMATCH", p)
    print(f"{len(problems)} mismatch(es) over {len(a)} rows, 40 queries, and a re-sync")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--db", type=Path, default=INDEX)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch-files", type=int, default=1000)
    ap.add_argument("--verify", type=int, metavar="N")
    args = ap.parse_args()
    if args.verify:
        return verify(args.corpus, args.verify, args.workers, args.threads)
    db = outside_root(args.db, args.corpus)
    result = build(args.corpus, db, args.workers, args.threads, args.batch_files, sys.stderr)
    (db.parent / "bulk_errors.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["counts"]))
    return 1 if result["counts"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
