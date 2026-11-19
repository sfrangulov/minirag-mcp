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


def apply_e5_prefixes() -> None:
    """Prefix texts the way multilingual-e5 was trained: "query: " and "passage: ".

    An experiment, applied per process: neither fastembed nor minirag adds them, and the
    model card warns that without them quality drops.
    """
    from minirag_mcp.embedder import Embedder

    if getattr(Embedder, "_e5_prefixed", False):
        return
    docs, query = Embedder.embed_documents, Embedder.embed_query
    Embedder.embed_documents = lambda self, texts: docs(self, [f"passage: {t}" for t in texts])
    Embedder.embed_query = lambda self, text: query(self, f"query: {text}")
    Embedder._e5_prefixed = True


def apply_embedding_endpoint() -> bool:
    """Embed through an OpenAI-compatible endpoint named by ERB_EMBED_URL, when set.

    An experiment, applied per process before any prefixing: vectors come from the
    endpoint (llama.cpp's /v1/embeddings measured within cosine 0.99999 of fastembed's
    for multilingual-e5-large), while chunk sizes are still counted locally with the
    model's own tokenizer.json, untruncated and unpadded as minirag counts them.
    """
    url = os.environ.get("ERB_EMBED_URL")
    if not url:
        return False
    import glob

    import requests
    from tokenizers import Tokenizer

    from minirag_mcp.embedder import Embedder, _unit

    if getattr(Embedder, "_endpoint", None) == url:
        return True
    # A llama.cpp router routes by model name and may require a key.
    model_name = os.environ.get("ERB_EMBED_MODEL")
    key = os.environ.get("ERB_EMBED_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    counters: dict[str, object] = {}

    def count_tokens(self, text: str) -> int:
        counter = counters.get(self.model_name)
        if counter is None:
            repo = "models--qdrant--" + self.model_name.split("/")[-1] + "-onnx"
            found = glob.glob(
                str(self.cache_dir / f"{repo}*" / "snapshots" / "*" / "tokenizer.json")
            )
            found += glob.glob(
                str(
                    self.cache_dir
                    / f"{repo.replace('qdrant', 'Qdrant')}*"
                    / "snapshots"
                    / "*"
                    / "tokenizer.json"
                )
            )
            if not found:
                raise SystemExit(f"no tokenizer.json for {self.model_name} under {self.cache_dir}")
            counter = Tokenizer.from_file(found[0])
            counter.no_truncation()
            counter.no_padding()
            counters[self.model_name] = counter
        return len(counter.encode(text).ids)

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            body: dict = {"input": texts[i : i + 32]}
            if model_name:
                body["model"] = model_name
            r = requests.post(url, json=body, headers=headers, timeout=600)
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out += [_unit([float(x) for x in d["embedding"]]) for d in data]
        return out

    Embedder.count_tokens = count_tokens
    Embedder.embed_documents = lambda self, texts: embed(list(texts)) if texts else []
    Embedder.embed_query = lambda self, text: embed([text])[0]
    Embedder._endpoint = url
    return True
