"""Paths, questions and document ids shared by every EnterpriseRAG-Bench script.

The benchmark's data never enters this repository: it is large, and the dataset asks to
stay out of training corpora. Everything lives under ERB_DATA_DIR.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import platformdirs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUNS = HERE / "runs"
DATA_DIR = Path(
    os.environ.get("ERB_DATA_DIR") or platformdirs.user_cache_path("minirag-mcp") / "enterpriserag"
)
CORPUS = DATA_DIR / "corpus"
BENCH = DATA_DIR / "bench"
QUESTIONS = BENCH / "questions.jsonl"
INDEX = DATA_DIR / "index" / "lancedb"

BENCH_COMMIT = "d36685e273713975ee20299bbf1ab64165575b3c"
SPACE_RAW = (
    "https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard/raw/main/data"
)

DSID_RE = re.compile(r"dsid_[0-9a-f]{32}")
_NAME_RE = re.compile(r"^(dsid_[0-9a-f]{32})__")


@dataclass(frozen=True)
class Question:
    question_id: str
    question_type: str
    question: str
    expected_doc_ids: tuple[str, ...]


def dsid_of(path: str) -> str | None:
    """The document id an exported file carries at the start of its own name."""
    m = _NAME_RE.match(os.path.basename(path))
    return m.group(1) if m else None


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            Question(
                row["question_id"],
                row["question_type"],
                row["question"],
                tuple(row.get("expected_doc_ids") or ()),
            )
        )
    return out


def pilot_ids(questions: list[Question], per_type: int = 2) -> list[str]:
    """Evenly spaced questions of every type, in file order; no randomness to record."""
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)
    ids = []
    for group in by_type.values():
        n = len(group)
        picks = sorted({int((i + 0.5) * n / per_type) for i in range(min(per_type, n))})
        ids += [group[p].question_id for p in picks]
    return ids


def outside_root(db: Path, root: Path) -> Path:
    """Opening an index can write to it, so an index inside the document root is refused."""
    db, root = db.resolve(), root.resolve()
    if db == root or root in db.parents:
        raise SystemExit(f"refusing {db}: it is inside the document root {root}")
    return db


def corpus_files(corpus: Path = CORPUS) -> list[str]:
    return sorted(p.relative_to(corpus).as_posix() for p in corpus.rglob("*.txt"))


def sample_ids(questions: list[Question], fraction: float = 0.2) -> list[str]:
    """A type-proportional sample, evenly spaced within each type, at least one per type.

    The leaderboard's overall mixes types in the full set's proportions; a sample that
    keeps them estimates that mix, and its questions are scored for every published
    system too, so the comparison is on the same questions.
    """
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)
    ids = []
    for group in by_type.values():
        n = len(group)
        k = max(1, round(n * fraction))
        ids += [group[int((i + 0.5) * n / k)].question_id for i in range(k)]
    return sorted(ids)


# What the product reads: the stand passes these through to its own config, so an
# experiment is a set of environment variables and nothing is patched in.
CORE_ENV = (
    "MODEL_NAME",
    "CHUNK_TOKEN_BUDGET",
    "RAG_HYBRID_WEIGHT",
    "RAG_DOCUMENT_WEIGHT",
    "RAG_RERANK_TOP",
    "RAG_FTS_LANGUAGE",
    "EMBED_URL",
    "EMBED_ENDPOINT_MODEL",
    "RERANK_URL",
    "RERANK_MODEL",
    "ENDPOINT_API_KEY_FILE",
)


def core_env(**fixed: str) -> dict[str, str]:
    """The product's variables that are set, with `fixed` over them."""
    env = {name: os.environ[name] for name in CORE_ENV if name in os.environ}
    env.update(fixed)
    return env


def open_index(db: Path, corpus: Path, **fixed: str):
    """The index at `db` opened the way the server opens it: through `IndexHandle`, which
    stamps a new folder with its identity and carries the write guards."""
    from minirag_mcp.config import load_config
    from minirag_mcp.handle import IndexHandle

    return IndexHandle(load_config(core_env(BASE_DIR=str(corpus), DB_PATH=str(db), **fixed)))
