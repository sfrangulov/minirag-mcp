"""The reduced corpus the regression check indexes: every gold document plus a seeded,
per-source proportional sample of the rest.

It is a function of the sorted corpus listing, the questions, the seed and the size, so
it is reproducible without committing the list; `subset_hash` names it in a baseline.
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from common import Question, dsid_of

SEED = 20261118
TOTAL = 20_000


def subset(files: list[str], questions: list[Question], total: int, seed: int) -> list[str]:
    gold = {d for q in questions for d in q.expected_doc_ids}
    chosen = {f for f in files if dsid_of(f) in gold}
    if len(chosen) > total:
        raise SystemExit(f"{len(chosen)} gold documents do not fit in a subset of {total}")
    rest: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if f not in chosen:
            rest[f.split("/", 1)[0]].append(f)
    pool = sum(len(v) for v in rest.values())
    room = total - len(chosen)
    rng = random.Random(seed)
    sources = sorted(rest)
    quotas = {s: room * len(rest[s]) // pool for s in sources} if pool else {}
    short = room - sum(quotas.values())
    for s in sorted(sources, key=lambda s: -(room * len(rest[s]) % pool))[:short]:
        quotas[s] += 1
    for s in sources:
        chosen.update(rng.sample(rest[s], min(quotas[s], len(rest[s]))))
    return sorted(chosen)


def subset_hash(rel: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(rel)).encode()).hexdigest()


def materialize(corpus: Path, rel: list[str], dest: Path) -> None:
    """Hard links keep a 20,000-file tree free; a copy is the fallback across volumes."""
    for r in rel:
        target = dest / r
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(corpus / r, target)
        except OSError:
            shutil.copy2(corpus / r, target)
