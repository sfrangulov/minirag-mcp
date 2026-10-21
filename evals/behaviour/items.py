"""The eval's question set, loaded from items.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
ITEMS_FILE = HERE / "items.toml"

GROUPS = ("search", "no_search", "thin", "reading", "injection")


@dataclass(frozen=True)
class Item:
    id: str
    group: str
    block: str
    prompt: str
    search: bool
    answer: tuple[str, ...]
    allowed: tuple[str, ...]
    index: str = "corpus"
    max_queries: int | None = None
    fact: str | None = None
    forbidden_tools: tuple[str, ...] = ()
    obey_prefix: str | None = None
    planted: str | None = None
    mention: str | None = None
    cite_optional: bool = False
    probes: tuple[str, ...] = field(default=(), compare=False)
    good_probes: tuple[str, ...] = field(default=(), compare=False)


def load_items(path: Path = ITEMS_FILE) -> list[Item]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))["item"]
    items = []
    for row in raw:
        row = {k: tuple(v) if isinstance(v, list) else v for k, v in row.items()}
        item = Item(**row)
        if item.group not in GROUPS:
            raise ValueError(f"{item.id}: unknown group {item.group!r}")
        if item.index not in ("corpus", "empty"):
            raise ValueError(f"{item.id}: unknown index {item.index!r}")
        items.append(item)
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item ids")
    return items


def select(items: list[Item], ids: str | None) -> list[Item]:
    if not ids:
        return items
    wanted = [s.strip() for s in ids.split(",") if s.strip()]
    unknown = set(wanted) - {i.id for i in items}
    if unknown:
        raise SystemExit(f"unknown item id(s): {', '.join(sorted(unknown))}")
    return [i for i in items if i.id in wanted]


def existing_index(db: Path, base_dir: Path) -> Path:
    """An index to run against as it is, refused when it lies inside the document root.

    Pointed at a real corpus, the eval must leave both the documents and their own
    index untouched: opening an index can write to it (schema and FTS upgrades), so the
    eval runs on a copy, and an index inside the root is the original by construction.
    """
    db, base_dir = db.resolve(), base_dir.resolve()
    if db == base_dir or base_dir in db.parents:
        raise SystemExit(f"refusing {db}: it is inside the document root; run on a copy")
    if not db.is_dir():
        raise SystemExit(f"no index at {db}")
    return db
