"""Labelling at ingest, and re-labelling stale documents during a sync."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from conftest import FakeChoiceModel
from minirag_mcp.config import Config, load_config
from minirag_mcp.ingest.pipeline import Pipeline, file_sha256
from minirag_mcp.labels import Labeler, LabelerSource, model, taxonomy_path
from minirag_mcp.labels.dates import DocDate
from minirag_mcp.labels.filters import LabelFilterError
from minirag_mcp.labels.model import ModelError, default_model_factory
from minirag_mcp.labels.resolve import Label, decode_labels
from minirag_mcp.labels.taxonomy import load_taxonomy
from minirag_mcp.store import LABEL_COLUMNS, Store
from minirag_mcp.sync import run_sync
from store_helpers import _older_table, _raise_commit_conflict, _row

TAXONOMY = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.initiative]
default = "general"
values.general = { description = "Work outside any initiative." }
values.alpha = { description = "The Alpha initiative.", paths = ["alpha"] }
values.beta = { description = "The Beta initiative.", paths = ["beta"] }

[facets.type]
values.minutes = { description = "Minutes. Date, participants, decisions." }
values.report = { description = "Report. Findings and figures." }
"""

# The same file with one more value the model may answer: an edit that changes the hash.
WITH_MEMO = TAXONOMY + 'values.memo = { description = "Memo. A short internal note." }\n'

FIELDS = ("labels_json", "label_keys", "label_hash", "doc_date", "doc_date_source")


class SaltedEmbedder:
    """Counts `embed_documents` calls and salts each call's vectors with its number.

    A deterministic embedder gives unchanged text the same vector again, so comparing
    stored vectors could not tell a re-embedding from none; with the salt it can.
    """

    def __init__(self, inner):
        self._inner = inner
        self.dim = inner.dim
        self.model_name = inner.model_name
        self.calls = 0

    def count_tokens(self, text: str) -> int:
        return self._inner.count_tokens(text)

    def embed_documents(self, texts):
        self.calls += 1
        return self._inner.embed_documents([f"{t}\x00{self.calls}" for t in texts])

    def embed_query(self, text: str):
        return self._inner.embed_query(text)


@dataclass
class Corpus:
    root: Path
    cfg: Config
    store: Store
    embedder: SaltedEmbedder
    pipeline: Pipeline

    @property
    def taxonomy_file(self) -> Path:
        return taxonomy_path(self.cfg.roots)

    def taxonomy(self, text: str = TAXONOMY) -> Path:
        path = self.taxonomy_file
        path.parent.mkdir(parents=True, exist_ok=True)
        before = path.stat().st_mtime if path.exists() else None
        path.write_text(text, encoding="utf-8")
        if before is not None:
            # Reload detection keys on mtime and size; do not rely on clock resolution.
            os.utime(path, (before + 10, before + 10))
        return path

    def doc(self, rel: str, head: str, parts: int = 1) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n\n".join(
            f"## Part {i}\n\n" + f"Body paragraph {i} of {path.stem}. " * 60 for i in range(parts)
        )
        path.write_text(f"# {path.stem}\n\n{head}\n\n{body}", encoding="utf-8")
        return path

    def sync(self, scope: Path | None = None):
        return run_sync(
            self.pipeline, self.store, self.cfg.roots, self.cfg.max_file_size, scope=scope
        )

    def rows(self, source: str | None = None) -> list[dict]:
        q = self.store._table.search()
        if source is not None:
            q = q.where(f"source = '{source}'")
        return q.limit(100_000).to_list()

    def vectors(self) -> dict[str, list[float]]:
        return {r["id"]: list(r["vector"]) for r in self.rows()}

    def labels(self, source: Path | str) -> dict[str, Label]:
        return decode_labels(self.store.get_source(str(source)).labels_json)

    def hash(self) -> str:
        return load_taxonomy(self.taxonomy_file).hash


def make_corpus(tmp_path: Path, embedder, factory) -> Corpus:
    root = tmp_path / "docs"
    root.mkdir()
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
    store = Store(tmp_path / "index", dim=embedder.dim)
    salted = SaltedEmbedder(embedder)
    source = LabelerSource(cfg.roots, tmp_path / "cache", factory)
    return Corpus(cfg.roots[0], cfg, store, salted, Pipeline(store, salted, cfg, labels=source))


@pytest.fixture
def corpus(tmp_path, fake_embedder, model_factory) -> Corpus:
    return make_corpus(tmp_path, fake_embedder, model_factory)


# --- ingest ------------------------------------------------------------------------


def test_ingest_writes_labels_and_date_on_every_chunk(corpus, model_factory):
    corpus.taxonomy()
    model_factory.answers = {"Minutes of": {"minutes": 0.9, "report": 0.1}}
    doc = corpus.doc("alpha/weekly.md", "Minutes of the weekly sync. Date: 14 March 2026.", 4)

    corpus.pipeline.ingest_file(doc)

    rows = corpus.rows(str(doc))
    assert len(rows) >= 2, "the document must span several chunks for this test to mean much"
    for row in rows:
        assert json.loads(row["labels_json"]) == {
            "initiative": {
                "value": "alpha",
                "confidence": 1.0,
                "source": "path",
                "second": None,
                "uncertain": False,
            },
            "type": {
                "value": "minutes",
                "confidence": 0.9,
                "source": "model",
                "second": "report",
                "uncertain": False,
            },
        }
        assert row["label_keys"] == "|initiative=alpha|type=minutes|"
        assert row["label_hash"] == corpus.hash()
        assert (row["doc_date"], row["doc_date_source"]) == ("2026-03-14", "document")
    assert len(model_factory.created) == 1
    assert model_factory.created[0].closed


def test_a_file_without_a_date_of_its_own_is_dated_by_its_creation_time(corpus):
    corpus.taxonomy()
    doc = corpus.doc("beta/notes.md", "Quarterly report without any date in it.")
    # Forward, not back: macOS pulls the creation time back to an earlier mtime. Where a
    # creation time is recorded, the two now differ by years.
    later = datetime(2031, 1, 15, 12).timestamp()
    os.utime(doc, (later, later))
    info = doc.stat()

    corpus.pipeline.ingest_file(doc)

    expected = datetime.fromtimestamp(getattr(info, "st_birthtime", info.st_mtime)).date()
    stored = corpus.store.get_source(str(doc))
    assert (stored.doc_date, stored.doc_date_source) == (expected.isoformat(), "file")


def test_text_documents_get_model_labels_but_no_path_or_default_ones(corpus, model_factory):
    corpus.taxonomy()
    model_factory.answers = {"Minutes of": {"minutes": 0.8, "report": 0.2}}

    corpus.pipeline.ingest_data("Minutes of the board. Date: 2 May 2026.", source="note-1")

    assert corpus.labels("note-1") == {
        "type": Label("minutes", 0.8, "model", second="report", uncertain=False)
    }
    stored = corpus.store.get_source("note-1")
    assert (stored.doc_date, stored.doc_date_source) == ("2026-05-02", "document")
    assert stored.label_hash == corpus.hash()


def test_without_classify_toml_nothing_is_labelled_and_no_model_is_made(corpus, model_factory):
    corpus.pipeline.ingest_file(corpus.doc("alpha/weekly.md", "Minutes of the weekly sync."))
    corpus.pipeline.ingest_data("Minutes of the board.", source="note-1")
    corpus.doc("beta/b.md", "Quarterly report.")

    counts, errors = corpus.sync()

    assert errors == []
    assert counts["ingested"] == 1 and counts["relabelled"] == 0
    assert model_factory.calls == []
    rows = corpus.rows()
    assert {r["source"] for r in rows} == {
        str(corpus.root / "alpha" / "weekly.md"),
        str(corpus.root / "beta" / "b.md"),
        "note-1",
    }
    for row in rows:
        assert all(row[name] == "" for name in FIELDS), row["source"]
    with corpus.pipeline.job() as report:
        assert report.problems == []


def test_a_labelling_error_leaves_the_document_ingested_and_unlabelled(corpus, monkeypatch):
    corpus.taxonomy()
    doc = corpus.doc("alpha/weekly.md", "Minutes of the weekly sync.")

    def broken(self, **kwargs):
        raise RuntimeError("cannot label this")

    monkeypatch.setattr(Labeler, "label", broken)
    with corpus.pipeline.job() as report:
        result = corpus.pipeline.ingest_file(doc)

    assert result.chunk_count >= 1
    for row in corpus.rows(str(doc)):
        assert all(row[name] == "" for name in FIELDS)
    assert report.problems == [f"{doc}: RuntimeError: cannot label this"]


def test_documents_ingested_inside_one_job_share_one_model_and_one_taxonomy(corpus, model_factory):
    """An edit to classify.toml in the middle of a sync takes effect with the next job:
    one sync labels with one taxonomy, so its re-label pass compares against the hash
    its own documents were written with."""
    corpus.taxonomy()
    first_hash = corpus.hash()
    a = corpus.doc("alpha/a.md", "Quarterly report A.")
    b = corpus.doc("alpha/b.md", "Quarterly report B.")

    with corpus.pipeline.job() as outer:
        corpus.pipeline.ingest_file(a)
        corpus.taxonomy(WITH_MEMO)
        with corpus.pipeline.job() as inner:
            assert inner is outer
        corpus.pipeline.ingest_file(b)
        assert not model_factory.created[0].closed

    assert len(model_factory.created) == 1 and model_factory.created[0].closed
    assert corpus.store.get_source(str(a)).label_hash == first_hash
    assert corpus.store.get_source(str(b)).label_hash == first_hash
    corpus.pipeline.ingest_file(b)
    assert corpus.store.get_source(str(b)).label_hash == corpus.hash() != first_hash


# --- sync --------------------------------------------------------------------------


def test_a_sync_loads_the_model_once_and_releases_it(corpus, model_factory):
    corpus.taxonomy()
    for i in range(5):
        corpus.doc(f"{'alpha' if i % 2 else 'beta'}/doc{i}.md", f"Quarterly report {i}.")

    counts, errors = corpus.sync()

    assert errors == []
    assert counts["ingested"] == 5 and counts["relabelled"] == 0
    assert len(model_factory.created) == 1
    assert model_factory.created[0].closed
    assert len(model_factory.created[0].calls) == 5


def test_editing_the_taxonomy_relabels_without_re_embedding(corpus, model_factory):
    corpus.taxonomy()
    # `memo` is not an option until the edit, so the same answers yield a new label.
    model_factory.default = {"memo": 0.8, "report": 0.2}
    sources = [str(corpus.doc(f"alpha/doc{i}.md", f"Quarterly report {i}.")) for i in range(5)]
    corpus.sync()
    assert {corpus.labels(s)["type"].value for s in sources} == {"report"}
    embeds = corpus.embedder.calls
    vectors = corpus.vectors()

    corpus.taxonomy(WITH_MEMO)
    counts, errors = corpus.sync()

    assert errors == []
    assert counts["relabelled"] == 5
    assert counts["ingested"] == 0 and counts["skipped"] == 5 and counts["failed"] == 0
    assert corpus.embedder.calls == embeds
    assert corpus.vectors() == vectors
    for s in sources:
        assert corpus.labels(s)["type"] == Label("memo", 0.8, "model", second="report")
        assert corpus.store.get_source(s).label_hash == corpus.hash()
    assert all(r["label_keys"] == "|initiative=alpha|type=memo|" for r in corpus.rows())
    counts, _ = corpus.sync()
    assert counts["relabelled"] == 0


def test_a_file_moved_into_another_folder_gets_its_new_path_label(corpus):
    corpus.taxonomy()
    (corpus.root / "beta").mkdir()
    doc = corpus.doc("alpha/plan.md", "Quarterly report on the plan.")
    corpus.sync()
    assert corpus.labels(doc)["initiative"].value == "alpha"
    embeds = corpus.embedder.calls

    moved = doc.rename(corpus.root / "beta" / "plan.md")
    counts, errors = corpus.sync()

    assert errors == []
    assert counts["moved"] == 1 and counts["relabelled"] == 1 and counts["ingested"] == 0
    assert corpus.embedder.calls == embeds
    assert corpus.labels(moved)["initiative"] == Label("beta", 1.0, "path")
    assert corpus.store.get_source(str(moved)).label_hash == corpus.hash()


def test_relabelling_keeps_a_stored_date_and_fills_a_missing_one(corpus):
    dated = corpus.doc("alpha/dated.md", "Minutes of the sync. Date: 14 March 2026.")
    kept = corpus.doc("alpha/kept.md", "Minutes of the review. Date: 1 June 2026.")
    corpus.sync()
    assert corpus.store.get_source(str(dated)).doc_date_source == ""

    corpus.taxonomy()
    corpus.store.set_document_labels(
        str(kept),
        labels_json="",
        label_keys="",
        label_hash="an older taxonomy",
        doc_date="2020-01-01",
        doc_date_source="name",
    )
    counts, errors = corpus.sync()

    assert errors == [] and counts["relabelled"] == 2
    first = corpus.store.get_source(str(dated))
    assert (first.doc_date, first.doc_date_source) == ("2026-03-14", "document")
    second = corpus.store.get_source(str(kept))
    assert (second.doc_date, second.doc_date_source) == ("2020-01-01", "name")
    assert second.label_hash == corpus.hash()
    assert corpus.labels(kept)["initiative"].value == "alpha"


def test_a_scoped_sync_relabels_only_inside_its_scope(corpus):
    corpus.taxonomy()
    corpus.doc("alpha/a1.md", "Quarterly report A1.")
    corpus.doc("alpha/a2.md", "Quarterly report A2.")
    outside = corpus.doc("beta/b1.md", "Quarterly report B1.")
    corpus.sync()
    old = corpus.hash()

    corpus.taxonomy(WITH_MEMO)
    counts, errors = corpus.sync(scope=corpus.root / "alpha")

    assert errors == [] and counts["relabelled"] == 2
    assert corpus.store.get_source(str(outside)).label_hash == old
    counts, _ = corpus.sync()
    assert counts["relabelled"] == 1


def test_a_relabel_that_raises_is_one_error_for_that_source(corpus, monkeypatch):
    corpus.taxonomy()
    docs = [corpus.doc(f"alpha/doc{i}.md", f"Quarterly report {i}.") for i in range(3)]
    corpus.sync()
    corpus.taxonomy(WITH_MEMO)
    real = corpus.store.set_document_labels

    def refuse_one(source, **kwargs):
        if source == str(docs[1]):
            raise RuntimeError("the index has no label columns")
        return real(source, **kwargs)

    monkeypatch.setattr(corpus.store, "set_document_labels", refuse_one)
    counts, errors = corpus.sync()

    assert counts["failed"] == 0 and counts["relabelled"] == 2
    assert errors == [
        {"source": str(docs[1]), "error": "labels: RuntimeError: the index has no label columns"}
    ]
    monkeypatch.setattr(corpus.store, "set_document_labels", real)
    counts, errors = corpus.sync()
    assert errors == [] and counts["relabelled"] == 1


def test_a_missing_model_is_one_error_and_not_a_failure(tmp_path, fake_embedder, monkeypatch):
    # Past the extra check, so the real factory fails on the file and not on the install.
    monkeypatch.setattr(model, "available", lambda: True)
    corpus = make_corpus(tmp_path, fake_embedder, default_model_factory)
    missing = tmp_path / "missing.gguf"
    path = corpus.taxonomy(TAXONOMY.replace("/models/fake.gguf", str(missing)))
    in_alpha = corpus.doc("alpha/a.md", "Quarterly report A.")
    in_beta = corpus.doc("beta/b.md", "Quarterly report B.")
    at_root = corpus.doc("c.md", "Quarterly report C.")

    counts, errors = corpus.sync()

    assert counts["failed"] == 0 and counts["ingested"] == 3
    assert errors == [
        {
            "source": str(path),
            "error": f"labels: the classify model file {missing} does not exist",
        }
    ]
    assert corpus.labels(in_alpha) == {"initiative": Label("alpha", 1.0, "path")}
    assert corpus.labels(in_beta) == {"initiative": Label("beta", 1.0, "path")}
    assert corpus.labels(at_root) == {"initiative": Label("general", 1.0, "default")}


def test_documents_a_missing_model_left_unlabelled_are_relabelled_once_it_answers(
    corpus, model_factory
):
    corpus.taxonomy()
    model_factory.error = ModelError("the classify model is not installed")
    docs = [
        corpus.doc("alpha/a.md", "Quarterly report A."),
        corpus.doc("beta/b.md", "Quarterly report B."),
        corpus.doc("c.md", "Quarterly report C."),
    ]

    counts, errors = corpus.sync()

    assert counts["failed"] == 0 and counts["ingested"] == 3 and counts["relabelled"] == 0
    assert errors == [
        {
            "source": str(corpus.taxonomy_file),
            "error": "labels: the classify model is not installed",
        }
    ]
    for doc in docs:
        assert corpus.store.get_source(str(doc)).label_hash == ""
        assert set(corpus.labels(doc)) == {"initiative"}

    # Still out: one attempt per sync, and the same partial labels are not written again.
    version = corpus.store._table.version
    counts, errors = corpus.sync()
    assert counts["relabelled"] == 0 and len(errors) == 1
    assert len(model_factory.calls) == 2
    assert corpus.store._table.version == version

    embeds = corpus.embedder.calls
    model_factory.error = None
    model_factory.default = {"report": 0.9, "minutes": 0.1}
    counts, errors = corpus.sync()

    assert errors == []
    assert counts["relabelled"] == 3 and counts["ingested"] == 0
    assert corpus.embedder.calls == embeds
    for doc in docs:
        assert corpus.store.get_source(str(doc)).label_hash == corpus.hash()
        assert corpus.labels(doc)["type"] == Label("report", 0.9, "model", second="minutes")


def test_a_document_labelled_without_the_model_gets_the_hash_while_the_model_is_missing(
    corpus, model_factory
):
    corpus.taxonomy(TAXONOMY.replace("[facets.type]\n", '[facets.type]\nwithin = ["alpha"]\n'))
    model_factory.error = ModelError("the classify model is not installed")
    asks = corpus.doc("alpha/a.md", "Quarterly report A.")
    needs_none = corpus.doc("beta/b.md", "Quarterly report B.")

    counts, errors = corpus.sync()

    assert counts["failed"] == 0 and len(errors) == 1
    assert corpus.labels(needs_none) == {"initiative": Label("beta", 1.0, "path")}
    assert corpus.store.get_source(str(needs_none)).label_hash == corpus.hash()
    assert corpus.store.get_source(str(asks)).label_hash == ""


def test_an_index_without_label_columns_is_one_error_and_not_a_failure(
    tmp_path, fake_embedder, model_factory, monkeypatch
):
    root = tmp_path / "docs"
    root.mkdir()
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
    rows = []
    for name in ("a.md", "b.md"):
        path = root / name
        path.write_text(f"# {name}\n\nQuarterly report.", encoding="utf-8")
        # The real hash, so the sync finds the files unchanged and keeps their rows.
        rows.append({**_row(str(path), 0, "Quarterly report."), "file_hash": file_sha256(path)})
    db_path = tmp_path / "old"
    table = _older_table(db_path, set(LABEL_COLUMNS), rows)
    monkeypatch.setattr(type(table), "add_columns", _raise_commit_conflict)
    with pytest.warns(RuntimeWarning, match="labels_json"):
        store = Store(db_path, dim=fake_embedder.dim)
    source = LabelerSource(cfg.roots, tmp_path / "cache", model_factory)
    pipeline = Pipeline(store, fake_embedder, cfg, labels=source)
    source.path.parent.mkdir()
    source.path.write_text(TAXONOMY, encoding="utf-8")

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert counts["failed"] == 0 and counts["skipped"] == 2 and counts["relabelled"] == 0
    assert len(errors) == 1
    assert errors[0]["source"] == str(db_path)
    assert all(column in errors[0]["error"] for column in LABEL_COLUMNS)
    assert "write access" in errors[0]["error"]
    assert model_factory.calls == []


def test_an_invalid_taxonomy_is_one_error_and_not_a_failure(corpus, model_factory):
    path = corpus.taxonomy("[facets.type]\nvalues = 3\n")
    corpus.doc("alpha/a.md", "Quarterly report A.")
    corpus.doc("beta/b.md", "Quarterly report B.")

    counts, errors = corpus.sync()

    assert counts["failed"] == 0 and counts["ingested"] == 2 and counts["relabelled"] == 0
    assert len(errors) == 1
    assert errors[0]["source"] == str(path)
    assert errors[0]["error"].startswith(f"labels: {path} is invalid: ")
    assert model_factory.calls == []
    for row in corpus.rows():
        assert all(row[name] == "" for name in FIELDS)


def test_a_model_that_cannot_be_released_is_reported_by_the_sync(corpus, model_factory):
    """The model is released as the job ends, after the sync's own work is done."""

    class StuckModel(FakeChoiceModel):
        def close(self) -> None:
            super().close()
            raise RuntimeError("still mapped")

    model_factory.model_class = StuckModel
    corpus.taxonomy()
    corpus.doc("alpha/a.md", "Quarterly report A.")

    counts, errors = corpus.sync()

    assert counts["failed"] == 0 and counts["ingested"] == 1
    assert errors == [
        {
            "source": str(corpus.taxonomy_file),
            "error": "labels: could not release the classify model: RuntimeError: still mapped",
        }
    ]


# --- preview -----------------------------------------------------------------------


def test_preview_labels_writes_nothing(corpus, model_factory):
    corpus.taxonomy()
    model_factory.answers = {"Minutes of": {"minutes": 0.6, "report": 0.4}}
    doc = corpus.doc("alpha/weekly.md", "Minutes of the weekly sync. Date: 14 March 2026.", 3)

    labels, trace = corpus.pipeline.preview_labels(doc)

    assert labels.labels == {
        "initiative": Label("alpha", 1.0, "path"),
        "type": Label("minutes", 0.6, "model", second="report", uncertain=True),
    }
    assert labels.date == DocDate("2026-03-14", "document")
    assert labels.taxonomy_hash == corpus.hash()
    assert trace == {"type": {"minutes": 0.6, "report": 0.4}}
    assert corpus.embedder.calls == 0
    assert corpus.store.chunk_count() == 0
    assert model_factory.created[0].closed

    # It saw what an ingest of the same file sees.
    corpus.pipeline.ingest_file(doc)
    previewed, ingested = (m.calls[0][1] for m in model_factory.created)
    assert previewed == ingested


def test_preview_labels_without_classify_toml_is_refused(corpus, model_factory):
    doc = corpus.doc("alpha/weekly.md", "Minutes of the weekly sync.")

    with pytest.raises(LabelFilterError, match="document labels are off: there is no"):
        corpus.pipeline.preview_labels(doc)

    corpus.taxonomy("[facets.type]\nvalues = 3\n")
    with pytest.raises(LabelFilterError, match="document labels are off: .* is invalid: "):
        corpus.pipeline.preview_labels(doc)
    assert model_factory.calls == []


def test_a_pipeline_without_a_label_source_labels_nothing(tmp_path, fake_embedder):
    root = tmp_path / "docs"
    root.mkdir()
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
    store = Store(tmp_path / "index", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)
    path = taxonomy_path(cfg.roots)
    path.parent.mkdir()
    path.write_text(TAXONOMY, encoding="utf-8")
    (root / "a.md").write_text("# A\n\nQuarterly report A.", encoding="utf-8")

    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)

    assert errors == [] and counts["ingested"] == 1 and counts["relabelled"] == 0
    assert store.get_source(str(root / "a.md")).label_hash == ""
    assert pipeline.relabel_stale() == (0, [])
