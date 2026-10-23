"""`lookAlikes` on the rows `query_documents` and `find_mentions` return."""

from pathlib import Path

import pytest

from lookalike_texts import (
    MINUTES,
    REGULATION_V1,
    REGULATION_V2,
    SPEC_RECRUITING,
    SPEC_WAREHOUSE,
)
from minirag_mcp.chunker import join_document
from minirag_mcp.config import Config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.lookalike import overlap, signature
from minirag_mcp.results import run_find, run_query, signature_status
from minirag_mcp.store import SIGNATURE_TABLE, Store

FILES = {
    "regs/v1.md": REGULATION_V1,
    "archive/v2.md": REGULATION_V2,
    "specs/warehouse.md": SPEC_WAREHOUSE,
    "specs/recruiting.md": SPEC_RECRUITING,
    "minutes.md": MINUTES,
}


def make_config(root: Path, tmp_path: Path) -> Config:
    return Config(
        roots=(root,),
        db_path=tmp_path / "db",
        cache_dir=tmp_path / "cache",
        model_name="fake",
        max_file_size=1_000_000,
        token_budget=200,
        hybrid_weight=0.6,
        grouping=None,
        max_distance=None,
        max_files=None,
        allow_private_urls=False,
    )


@pytest.fixture
def corpus(tmp_path, fake_embedder):
    root = (tmp_path / "docs").resolve()
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    config = make_config(root, tmp_path)
    store = Store(config.db_path, dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, config)
    for rel in FILES:
        pipeline.ingest_file(root / rel)
    return root, config, store, fake_embedder


def src(root, rel):
    return str(root / rel)


def expected_overlap(store, a, b):
    return round(
        overlap(
            signature(join_document(store.all_chunks(a))),
            signature(join_document(store.all_chunks(b))),
        ),
        2,
    )


def test_a_query_names_the_other_version_of_a_document_it_returns(corpus):
    root, config, store, embedder = corpus
    v1, v2 = src(root, "regs/v1.md"), src(root, "archive/v2.md")
    store.set_document_labels(
        v2,
        labels_json="",
        label_keys="",
        label_hash="",
        doc_date="2026-03-01",
        doc_date_source="text",
    )
    payload = run_query(store, embedder, config, "согласование заявки на закупку", top_k=20)
    rows = {row["source"]: row for row in payload["sources"]}
    assert v1 in rows and v2 in rows

    assert rows[v1]["lookAlikes"] == [
        {
            "source": v2,
            "displayPath": "archive/v2.md",
            "overlap": expected_overlap(store, v1, v2),
            "date": {"value": "2026-03-01", "source": "text"},
        }
    ]
    assert rows[v2]["lookAlikes"] == [
        {"source": v1, "displayPath": "regs/v1.md", "overlap": expected_overlap(store, v1, v2)}
    ]
    assert 0.5 <= rows[v1]["lookAlikes"][0]["overlap"] <= 1.0


def test_a_document_without_look_alikes_keeps_exactly_its_old_keys(corpus):
    """Two specifications from one template share their boilerplate, not most of
    their text, and neither is a look-alike of the other."""
    root, config, store, embedder = corpus
    payload = run_query(store, embedder, config, "техническое задание система", top_k=50)
    rows = {row["source"]: row for row in payload["sources"]}
    for rel in ("specs/warehouse.md", "specs/recruiting.md", "minutes.md"):
        assert set(rows[src(root, rel)]) == {"source", "title", "hits", "displayPath"}


def test_look_alikes_come_from_the_whole_index_whatever_the_scope(corpus):
    """A sibling outside the scope is exactly the one the caller has to hear about."""
    root, config, store, embedder = corpus
    payload = run_query(
        store, embedder, config, "заявки на закупку", top_k=20, scopes=(str(root / "regs"),)
    )
    [row] = payload["sources"]
    assert [a["source"] for a in row["lookAlikes"]] == [src(root, "archive/v2.md")]


def test_find_mentions_names_the_look_alikes_of_the_documents_it_lists(corpus):
    root, config, store, _ = corpus
    payload = run_find(store, config.roots, "закупочной")
    rows = {row["source"]: row for row in payload["documents"]}
    assert set(rows) == {src(root, "regs/v1.md"), src(root, "archive/v2.md")}
    assert [a["displayPath"] for a in rows[src(root, "regs/v1.md")]["lookAlikes"]] == [
        "archive/v2.md"
    ]
    lone = run_find(store, config.roots, "кандидатов")
    [row] = lone["documents"]
    assert "lookAlikes" not in row


def test_find_mentions_names_look_alikes_outside_its_scope(corpus):
    root, config, store, _ = corpus
    payload = run_find(store, config.roots, "закупочной", scopes=(str(root / "archive"),))
    [row] = payload["documents"]
    assert [a["source"] for a in row["lookAlikes"]] == [src(root, "regs/v1.md")]


def test_status_counts_documents_without_a_signature_only_when_there_are_any(corpus):
    root, config, store, embedder = corpus
    assert signature_status(store) == {}
    store._db.drop_table(SIGNATURE_TABLE)
    store = Store(config.db_path, dim=embedder.dim)
    status = signature_status(store)
    assert status["unsignedSourceCount"] == len(FILES)
    assert "sync" in status["lookAlikesWarning"]
    assert run_query(store, embedder, config, "заявки на закупку")["sources"]
    assert all(
        "lookAlikes" not in r for r in run_query(store, embedder, config, "заявки")["sources"]
    )
