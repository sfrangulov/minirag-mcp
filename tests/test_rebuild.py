"""A rebuild into `<db>.next`, its swap, and every process picking up the new generation."""

from __future__ import annotations

import errno
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import minirag_mcp.identity as identity_mod
import minirag_mcp.ingest.pipeline as pipeline_mod
import minirag_mcp.ocr as ocr_mod
import minirag_mcp.rebuild as rebuild_mod
import minirag_mcp.store as store_mod
from conftest import FakeEmbedder
from minirag_mcp.chunker import join_document
from minirag_mcp.config import load_config
from minirag_mcp.embedder import E5_LARGE, MINILM, default_budget
from minirag_mcp.handle import IndexHandle, RebuildPendingError
from minirag_mcp.identity import (
    INDEX_FILE,
    STAGING_MARKER,
    IdentityError,
    IndexState,
    previous_dir,
    read_identity,
    staging_dir,
)
from minirag_mcp.ingest.parser import ParsedDoc, ParserError
from minirag_mcp.lock import SyncLockBusy, sync_lock
from minirag_mcp.rebuild import KEPT
from minirag_mcp.results import run_query
from minirag_mcp.store import DOCUMENT_TABLE, SIGNATURE_TABLE, ChunkRecord, IndexBusyError

LONG = " ".join(
    f"Sentence {i} " + " ".join(f"w{i}x{j}" for j in range(10)) + "." for i in range(40)
)
TAXONOMY = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.type]
values.minutes = { description = "Minutes. Decisions." }
values.report = { description = "Report. Findings." }
"""


class Hooked(FakeEmbedder):
    """Runs `during` at its first document embedding, then embeds as FakeEmbedder does."""

    def __init__(self, *args, during=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.during = during
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.during is not None and self.calls == 1:
            self.during()
        return super().embed_documents(texts)


class Killed(BaseException):
    """The process dying where it is raised: nothing after it runs but `finally` blocks,
    which a dying process skips too, and whose work (releasing flocks) the kernel does."""


@pytest.fixture
def corpus(tmp_path) -> Path:
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("# Alpha\n\nAlpha body about budgets and reviews.")
    (root / "b.md").write_text("# Beta\n\nBeta body about tokens and audits.")
    (root / "sub" / "c.md").write_text("# Gamma\n\n" + LONG)
    return root


@pytest.fixture
def models() -> dict[str, FakeEmbedder]:
    return {
        MINILM: FakeEmbedder(MINILM, salt="minilm"),
        E5_LARGE: FakeEmbedder(E5_LARGE, salt="e5"),
    }


class Parses:
    def __init__(self) -> None:
        self.count = 0


@pytest.fixture
def parses(monkeypatch) -> Parses:
    seen = Parses()
    real = pipeline_mod.parse_file

    def counting(path, config=None):
        seen.count += 1
        return real(path, config)

    monkeypatch.setattr(pipeline_mod, "parse_file", counting)
    return seen


# Both models chunk at MiniLM's budget unless a test sets one, so switching between them
# changes the embedding alone.
SAME_BUDGET = str(default_budget(MINILM))


def handle(root: Path, models: dict, model_factory=None, **env: str) -> IndexHandle:
    config = load_config(
        {"BASE_DIR": str(root), "CHUNK_TOKEN_BUDGET": SAME_BUDGET, **env}, cwd=root
    )
    return IndexHandle(
        config, embedder_factory=lambda scheme: models[scheme.model], model_factory=model_factory
    )


def built(root: Path, models: dict, model_factory=None, **env: str) -> IndexHandle:
    """An index of `root` built with MiniLM, synced."""
    h = handle(root, models, model_factory, MODEL_NAME=MINILM, **env)
    h.run_sync(None)
    return h


def chunk_rows(h: IndexHandle) -> list[dict]:
    rows = h.get().store._table.search().select(["id", "text", "vector"]).limit(10**6).to_list()
    return sorted(rows, key=lambda r: r["id"])


def table_rows(h: IndexHandle, name: str, columns: list[str]) -> list[dict]:
    table = h.get().store._db.open_table(name)
    return sorted(table.search().select(columns).limit(10**6).to_list(), key=lambda r: r["source"])


def same_vectors(rows: list[dict], embedder) -> bool:
    expected = embedder.embed_documents([r["text"] for r in rows])
    return np.allclose([r["vector"] for r in rows], expected, atol=1e-6)


def test_an_unscoped_sync_reembeds_on_a_model_change_without_reading_files(
    corpus, models, parses, model_factory
):
    a = built(corpus, models, model_factory)
    before = chunk_rows(a)
    parses.count = 0
    b = handle(corpus, models, model_factory)
    assert b.state()[0] is IndexState.REBUILD_PENDING

    counts, errors, info = b.run_sync(None)

    assert (info.kind, info.generation, info.reembedded, info.restored) == (
        "reembed",
        2,
        len(before),
        0,
    )
    assert info.reasons == (f"model {MINILM} -> {E5_LARGE}",)
    assert errors == [] and (counts["ingested"], counts["skipped"]) == (0, 3)
    assert parses.count == 0 and model_factory.calls == []
    after = chunk_rows(b)
    assert [(r["id"], r["text"]) for r in after] == [(r["id"], r["text"]) for r in before]
    assert same_vectors(after, models[E5_LARGE]) and not same_vectors(after, models[MINILM])
    db = b.config.db_path
    assert read_identity(db).embedding.model == E5_LARGE
    assert read_identity(db).probe == pytest.approx(models[E5_LARGE].probe_vector())
    assert read_identity(previous_dir(db)).embedding.model == MINILM
    assert b.get().identity.generation == 2 and b.get().embedder is models[E5_LARGE]
    assert b.state() == (IndexState.READY, "")
    assert not staging_dir(db).exists()


def test_reembedding_keeps_document_rows_signatures_and_labels(
    corpus, models, parses, model_factory
):
    (corpus / ".minirag").mkdir()
    (corpus / ".minirag" / "classify.toml").write_text(TAXONOMY)
    model_factory.answers = {"Alpha": {"minutes": 1.0}}
    model_factory.default = {"report": 1.0}
    a = built(corpus, models, model_factory)
    assert model_factory.calls
    labels = {
        s.source: (s.labels_json, s.label_keys, s.label_hash, s.doc_date)
        for s in a.get().store.list_sources()
    }
    assert all(v[0] and v[1] and v[2] for v in labels.values())
    documents = table_rows(a, DOCUMENT_TABLE, ["source", "title", "text", "ingested_at"])
    signatures = table_rows(a, SIGNATURE_TABLE, ["source", "ingested_at", "signature"])
    assert len(documents) == len(signatures) == 3
    model_factory.calls.clear()
    parses.count = 0

    b = handle(corpus, models, model_factory)
    events: list[str] = []
    counts, errors, info = b.run_sync(None, on_event=events.append)

    assert info.kind == "reembed" and errors == []
    assert model_factory.calls == [] and parses.count == 0
    # Carried over, not rebuilt by the sync's own backfills.
    assert counts["signed"] == 0
    assert [e for e in events if e.startswith(("documented ", "signed ", "relabelled "))] == []
    store = b.get().store
    assert {
        s.source: (s.labels_json, s.label_keys, s.label_hash, s.doc_date)
        for s in store.list_sources()
    } == labels
    assert table_rows(b, DOCUMENT_TABLE, ["source", "title", "text", "ingested_at"]) == documents
    assert table_rows(b, SIGNATURE_TABLE, ["source", "ingested_at", "signature"]) == signatures
    assert store.undocumented_sources() == [] and store.unsigned_sources() == []


def test_a_new_vector_width_is_rebuilt_at_that_width(corpus, models):
    models[E5_LARGE] = FakeEmbedder(E5_LARGE, dim=16, salt="e5")
    built(corpus, models)
    b = handle(corpus, models)
    _, _, info = b.run_sync(None)
    opened = b.get()
    assert info.kind == "reembed"
    assert opened.identity.embedding.dim == 16 and opened.store.dim == 16
    assert opened.store._table.schema.field("vector").type.list_size == 16
    assert same_vectors(chunk_rows(b), models[E5_LARGE])
    found = run_query(opened.store, opened.embedder, b.config, "Beta body about tokens")
    assert found["sources"][0]["source"] == str(corpus / "b.md")


def test_a_budget_change_reingests_from_disk(corpus, models, parses):
    small = 32
    long_doc = str(corpus / "sub" / "c.md")
    a = built(corpus, models, CHUNK_TOKEN_BUDGET=str(small))
    cut_small = len(a.get().store.all_chunks(long_doc))
    parses.count = 0
    b = handle(corpus, models, MODEL_NAME=MINILM)
    assert b.config.token_budget > small

    counts, errors, info = b.run_sync(None)

    assert (info.kind, info.generation, info.reembedded, info.restored) == ("reingest", 2, 0, 0)
    assert info.reasons == (f"chunk budget {small} -> {b.config.token_budget} tokens",)
    assert counts["ingested"] == 3 and parses.count == 3 and errors == []
    assert read_identity(b.config.db_path).chunking.token_budget == b.config.token_budget
    assert b.get().identity.chunking.token_budget == b.config.token_budget
    assert len(b.get().store.all_chunks(long_doc)) < cut_small
    assert b.state() == (IndexState.READY, "")


def test_a_reingest_reads_its_files_in_workers_and_writes_what_a_serial_one_writes(
    tmp_path, models, inline_workers, monkeypatch
):
    # With the [ocr] extra installed, a parallel sync fetches Tesseract's data first.
    monkeypatch.setattr(ocr_mod, "available", lambda: False)
    root = tmp_path / "docs"
    root.mkdir()
    for i in range(10):
        (root / f"{i}.md").write_text(f"# Doc {i}\n\n" + LONG.replace("Sentence", f"Doc {i}"))

    def rebuilt(name: str, workers: int) -> list[dict]:
        env = {"DB_PATH": str(tmp_path / name), "CACHE_DIR": str(tmp_path / "cache")}
        built(root, models, CHUNK_TOKEN_BUDGET="32", RAG_SYNC_WORKERS="1", **env)
        h = handle(root, models, MODEL_NAME=MINILM, RAG_SYNC_WORKERS=str(workers), **env)
        counts, errors, info = h.run_sync(None)
        assert (info.kind, counts["ingested"], errors) == ("reingest", 10, [])
        rows = h.get().store._table.to_arrow().sort_by("id").to_pylist()
        return [{k: v for k, v in r.items() if k != "ingested_at"} for r in rows]

    serial = rebuilt("serial", 1)
    assert inline_workers == [] and len(serial) > 10
    assert rebuilt("parallel", 2) == serial
    assert inline_workers == [2]


def _words(text: str) -> list[str]:
    return text.split()


def test_a_reingest_keeps_what_the_disk_cannot_give_back(
    corpus, models, monkeypatch, model_factory
):
    small = 32
    (corpus / ".minirag").mkdir()
    (corpus / ".minirag" / "classify.toml").write_text(TAXONOMY)
    model_factory.default = {"report": 1.0}
    a = built(corpus, models, model_factory)
    a.ingest_pipeline().ingest_data("# Field notes\n\n" + LONG, source="notes")
    notes_before = a.get().store.all_chunks("notes")
    labels_before = {
        s.source: (s.labels_json, s.label_keys, s.label_hash) for s in a.get().store.list_sources()
    }
    assert labels_before["notes"][0]
    broken, gone = corpus / "b.md", corpus / "a.md"
    broken_before = join_document(a.get().store.all_chunks(str(broken)))
    gone.unlink()
    real = pipeline_mod.parse_file

    def parse(path, config=None):
        if Path(path) == broken:
            raise ParserError(f"{path}: this installation cannot read it any more")
        return real(path, config)

    monkeypatch.setattr(pipeline_mod, "parse_file", parse)
    b = handle(corpus, models, model_factory, MODEL_NAME=MINILM, CHUNK_TOKEN_BUDGET=str(small))

    counts, errors, info = b.run_sync(None)

    assert (info.kind, info.restored) == ("reingest", 2)
    store = b.get().store
    for source in ("notes", str(broken)):
        info_after = store.get_source(source)
        assert (info_after.labels_json, info_after.label_keys, info_after.label_hash) == (
            labels_before[source]
        )
    assert sorted(s.source for s in store.list_sources()) == sorted(
        ["notes", str(broken), str(corpus / "sub" / "c.md")]
    )
    assert {"source": str(broken), "error": KEPT} in errors
    assert [e for e in errors if e["source"] == "notes"] == []
    assert counts["failed"] == 1 and counts["deleted"] == 0
    notes = store.get_source("notes")
    assert (notes.source_type, notes.title) == ("data", "Field notes")
    notes_after = store.all_chunks("notes")
    assert len(notes_after) > len(notes_before)
    assert max(models[MINILM].count_tokens(c.text) for c in notes_after) <= small
    assert _words(join_document(notes_after)) == _words(join_document(notes_before))
    assert _words(join_document(store.all_chunks(str(broken)))) == _words(broken_before)
    assert store.get_source(str(broken)).file_hash == pipeline_mod.file_sha256(broken)


@pytest.mark.parametrize(
    "change",
    [{}, {"MODEL_NAME": MINILM, "CHUNK_TOKEN_BUDGET": "32"}],
    ids=["reembed", "reingest"],
)
def test_a_rebuild_does_not_keep_a_table_that_grew_over_the_limit(corpus, models, change):
    """A sync takes such a table out; a rebuild restoring its old text would put it back."""
    table = corpus / "t.csv"
    table.write_text("".join(f"{i},v{i}\n" for i in range(3)))
    built(corpus, models, RAG_MAX_TABLE_ROWS="5")
    table.write_text("".join(f"{i},v{i}\n" for i in range(9)))
    h = handle(corpus, models, RAG_MAX_TABLE_ROWS="5", **change)

    counts, errors, info = h.run_sync(None)

    assert (info.restored, counts["largeTables"], counts["failed"]) == (0, 1, 0)
    assert h.get().store.get_source(str(table)) is None
    assert [e["source"] for e in errors] == [str(table)]


def test_a_reingest_leaves_out_an_unchanged_table_over_a_new_limit(corpus, models):
    """The rebuild reads it again and skips it; its stored text does not put it back."""
    table = corpus / "t.csv"
    table.write_text("".join(f"{i},v{i}\n" for i in range(9)))
    built(corpus, models)
    h = handle(corpus, models, MODEL_NAME=MINILM, CHUNK_TOKEN_BUDGET="32", RAG_MAX_TABLE_ROWS="5")

    counts, errors, info = h.run_sync(None)

    skipped = "skipped: more than 5 data rows exceeds RAG_MAX_TABLE_ROWS (5)"
    assert (info.kind, info.restored, counts["largeTables"]) == ("reingest", 0, 1)
    assert errors == [{"source": str(table), "error": skipped}]
    assert h.get().store.get_source(str(table)) is None


def test_a_reingest_keeps_the_files_of_a_root_it_cannot_see(tmp_path, models):
    docs, drive = tmp_path / "docs", tmp_path / "drive"
    docs.mkdir()
    drive.mkdir()
    (docs / "a.md").write_text("# Alpha\n\nAlpha body.")
    (drive / "x.md").write_text("# Remote\n\nRemote body about cranes.")
    roots = f'["{docs}", "{drive}"]'
    built(docs, models, BASE_DIRS=roots, CHUNK_TOKEN_BUDGET="32")
    hidden = drive.rename(tmp_path / "unmounted")

    b = handle(docs, models, MODEL_NAME=MINILM, BASE_DIRS=roots)
    _, errors, info = b.run_sync(None)

    assert info.restored == 1
    assert {"source": str(drive / "x.md"), "error": KEPT} in errors
    assert b.get().store.get_source(str(drive / "x.md")) is not None
    assert hidden.is_dir()


def test_a_scoped_sync_on_a_pending_index_is_refused_and_writes_nothing(corpus, models):
    a = built(corpus, models, CHUNK_TOKEN_BUDGET="32")
    (corpus / "a.md").write_text("# Alpha\n\nAlpha body, changed on disk.")
    before = chunk_rows(a)
    b = handle(corpus, models, MODEL_NAME=MINILM)
    db = b.config.db_path

    with pytest.raises(RebuildPendingError) as refused:
        b.run_sync(corpus / "a.md")
    with pytest.raises(RebuildPendingError) as at_start:
        b.start_sync(corpus / "a.md")
    job = b.sync.start(corpus / "a.md")  # past the check at start, the run refuses it too
    b.sync.wait()

    message = str(refused.value)
    assert "index needs a rebuild: run an unscoped sync" in message
    assert f"chunk budget 32 -> {b.config.token_budget} tokens" in message
    assert str(corpus / "a.md") in message and "nothing was synced" in message
    assert str(at_start.value) == message
    assert (b.sync.status(job).state, b.sync.status(job).error) == ("failed", message)
    assert chunk_rows(b) == before
    assert read_identity(db).generation == 1 and b.get().identity.generation == 1
    assert b.state()[0] is IndexState.REBUILD_PENDING


def test_the_old_index_answers_until_the_swap(corpus, models):
    built(corpus, models)
    serving = handle(corpus, models, MODEL_NAME=MINILM)
    seen: dict = {}

    def during() -> None:
        opened = serving.get()
        found = run_query(opened.store, opened.embedder, serving.config, "Alpha body")
        seen["serving"] = (opened.identity.generation, [s["source"] for s in found["sources"]])
        seen["rebuilding"] = b.get().identity.generation

    models[E5_LARGE] = Hooked(E5_LARGE, salt="e5", during=during)
    b = handle(corpus, models)
    b.run_sync(None)

    generation, sources = seen["serving"]
    assert generation == 1 and str(corpus / "a.md") in sources
    assert seen["rebuilding"] == 1
    assert serving.get().identity.generation == 2


def test_a_failed_rebuild_leaves_the_old_index_serving(corpus, models):
    a = built(corpus, models)

    def fail() -> None:
        raise RuntimeError("the embedding endpoint went away")

    models[E5_LARGE] = Hooked(E5_LARGE, salt="e5", during=fail)
    b = handle(corpus, models)
    with pytest.raises(RuntimeError, match="went away"):
        b.run_sync(None)

    db = b.config.db_path
    assert read_identity(db).generation == 1 and b.get().identity.generation == 1
    assert b.state()[0] is IndexState.REBUILD_PENDING
    assert not staging_dir(db).exists() and not previous_dir(db).exists()
    opened = b.get()
    found = run_query(opened.store, opened.embedder, b.config, "Beta body")
    assert str(corpus / "b.md") in [s["source"] for s in found["sources"]]
    a.ingest_pipeline().ingest_data("Written after the failure.", source="after")
    _, _, info = b.run_sync(None)
    assert info.generation == 2 and b.get().store.get_source("after") is not None


def test_writes_are_refused_while_a_staging_dir_and_the_sync_lock_are_both_there(corpus, models):
    h = built(corpus, models)
    db = h.config.db_path
    store = h.get().store
    a_md = str(corpus / "a.md")
    expected = (
        f"{db} is being rebuilt into {staging_dir(db)}; writes to it are refused until the "
        "rebuild finishes, since the rebuilt index replaces it. Retry once sync_status "
        "reports the sync finished."
    )
    staging_dir(db).mkdir()
    writes = {
        "ingest": lambda: h.ingest_pipeline().ingest_data("x", source="s"),
        "delete": lambda: store.delete_source(a_md),
        "move": lambda: store.move_source(a_md, str(corpus / "moved.md"), mtime=1.0),
        "labels": lambda: store.set_document_labels(
            a_md, labels_json="", label_keys="", label_hash="h", doc_date="", doc_date_source=""
        ),
    }
    with sync_lock(db):
        for name, write in writes.items():
            with pytest.raises(IndexBusyError) as refused:
                write()
            assert str(refused.value) == expected, name
    assert store.get_source(a_md).label_hash != "h"
    h.ingest_pipeline().ingest_data("x", source="s")
    assert store.get_source("s") is not None


def test_writes_are_refused_while_a_rebuild_runs(corpus, models):
    built(corpus, models)
    serving = handle(corpus, models, MODEL_NAME=MINILM)
    a_md = str(corpus / "a.md")
    refused: dict[str, str] = {}

    def during() -> None:
        for name, write in (
            ("ingest", lambda: serving.ingest_pipeline().ingest_data("x", source="late")),
            ("delete", lambda: serving.get().store.delete_source(a_md)),
        ):
            with pytest.raises(IndexBusyError) as e:
                write()
            refused[name] = str(e.value)

    models[E5_LARGE] = Hooked(E5_LARGE, salt="e5", during=during)
    b = handle(corpus, models)
    b.run_sync(None)

    assert set(refused) == {"ingest", "delete"}
    assert all("is being rebuilt into" in why for why in refused.values())
    store = b.get().store
    assert store.get_source("late") is None and store.get_source(a_md) is not None
    b.ingest_pipeline().ingest_data("x", source="late")
    assert store.get_source("late") is not None


def test_a_write_under_way_when_a_rebuild_starts_is_in_the_rebuilt_index(
    corpus, models, monkeypatch
):
    """It passed the check before the staging folder existed and commits after: the
    rebuild waits for it before it reads the index, or the swap loses it."""
    built(corpus, models)
    serving = handle(corpus, models, MODEL_NAME=MINILM)
    checked, go = threading.Event(), threading.Event()
    real = store_mod.rebuild_running

    def held_after_the_check(db):
        running = real(db)
        if threading.current_thread().name == "writer":
            checked.set()
            assert go.wait(30)
        return running

    monkeypatch.setattr(store_mod, "rebuild_running", held_after_the_check)
    writer = threading.Thread(
        target=lambda: serving.ingest_pipeline().ingest_data("Late text.", source="late"),
        name="writer",
    )
    writer.start()
    assert checked.wait(30)
    b = handle(corpus, models)
    rebuilding = threading.Thread(target=lambda: b.run_sync(None), name="rebuild")
    rebuilding.start()
    time.sleep(0.5)
    go.set()
    writer.join(30)
    rebuilding.join(60)

    assert not rebuilding.is_alive() and b.get().identity.generation == 2
    late = [r for r in chunk_rows(b) if r["id"].startswith("late#")]
    assert late and same_vectors(late, models[E5_LARGE])


def _crash(root: Path, crash_in: str) -> subprocess.CompletedProcess:
    """Run in another process a rebuild of `root` to e5 that dies (os._exit) in its first
    `crash_in` call to the new embedder."""
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from conftest import FakeEmbedder\n"
        "from minirag_mcp.config import load_config\n"
        "from minirag_mcp.handle import IndexHandle\n"
        "class Dying(FakeEmbedder):\n"
        f"    def {crash_in}(self, *args):\n"
        "        os._exit(9)\n"
        f"models = {{{MINILM!r}: FakeEmbedder({MINILM!r}, salt='minilm'),\n"
        f"          {E5_LARGE!r}: Dying({E5_LARGE!r}, salt='e5')}}\n"
        f"env = {{'BASE_DIR': {str(root)!r}, 'CHUNK_TOKEN_BUDGET': {SAME_BUDGET!r}}}\n"
        f"config = load_config(env, cwd=Path({str(root)!r}))\n"
        "IndexHandle(config, embedder_factory=lambda s: models[s.model]).run_sync(None)\n"
    )
    unset = ("BASE_DIR", "BASE_DIRS", "DB_PATH", "MODEL_NAME", "CHUNK_TOKEN_BUDGET")
    env = {k: v for k, v in os.environ.items() if k not in unset}
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180, env=env
    )


def test_a_leftover_staging_dir_never_blocks_a_sync(corpus, models):
    a = built(corpus, models)
    proc = _crash(corpus, "embed_documents")
    assert proc.returncode == 9, proc.stderr
    db = a.config.db_path
    staging = staging_dir(db)
    assert (staging / STAGING_MARKER).is_file() and read_identity(staging) is None
    assert read_identity(db).generation == 1 and not previous_dir(db).exists()

    (corpus / "d.md").write_text("# Delta\n\nDelta body about cranes.")
    h = handle(corpus, models, MODEL_NAME=MINILM)
    h.ingest_pipeline().ingest_data("Written beside the debris.", source="beside")
    counts, errors, info = h.run_sync(None)

    assert (counts["ingested"], info, errors) == (1, None, [])
    assert not staging.exists()
    assert h.get().store.get_source(str(corpus / "d.md")) is not None


def test_a_staging_folder_no_rebuild_made_is_never_deleted(corpus, models):
    h = built(corpus, models)
    staging = staging_dir(h.config.db_path)
    staging.mkdir()
    (staging / "notes.txt").write_text("the user's", encoding="utf-8")
    with pytest.raises(IdentityError, match=re.escape(str(staging))):
        h.run_sync(None)
    assert (staging / "notes.txt").read_text(encoding="utf-8") == "the user's"


def test_a_rebuild_whose_swap_would_be_refused_stops_before_any_work(corpus, models):
    built(corpus, models)
    models[E5_LARGE] = Hooked(E5_LARGE, salt="e5")
    b = handle(corpus, models)
    db = b.config.db_path
    previous_dir(db).mkdir()
    (previous_dir(db) / "notes.txt").write_text("the user's", encoding="utf-8")
    with pytest.raises(IdentityError, match=re.escape(str(previous_dir(db)))):
        b.run_sync(None)
    assert models[E5_LARGE].calls == 0
    assert not staging_dir(db).exists() and read_identity(db).generation == 1
    assert (previous_dir(db) / "notes.txt").read_text(encoding="utf-8") == "the user's"


def test_no_sync_of_the_new_index_starts_before_the_rebuild_returns(corpus, models, monkeypatch):
    """The sync lock lives in the index folder, and the swap moves the one the rebuild
    holds into `<db>.previous`; the rebuild holds the new folder's from before the swap."""
    built(corpus, models)
    other = handle(corpus, models)
    outcomes: list[str] = []
    real = rebuild_mod.swap_in

    def swap(db) -> list[Path]:
        left = real(db)
        for start in (lambda: other.run_sync(None), lambda: other.start_sync(None)):
            try:
                start()
            except SyncLockBusy:
                outcomes.append("refused")
            else:
                outcomes.append("started")
        return left

    monkeypatch.setattr(rebuild_mod, "swap_in", swap)
    b = handle(corpus, models)
    _, _, info = b.run_sync(None)

    assert info.generation == 2
    assert outcomes == ["refused", "refused"]
    assert other.run_sync(None)[2] is None


def test_a_swap_by_another_process_is_picked_up_before_the_next_request(
    corpus, models, model_factory
):
    built(corpus, models)
    serving = handle(corpus, models, model_factory)
    first = serving.get()
    manager = serving.sync
    assert first.identity.generation == 1 and first.embedder is models[MINILM]

    handle(corpus, models).run_sync(None)

    opened = serving.get()
    assert opened is not first and opened.identity.generation == 2
    assert opened.embedder is models[E5_LARGE] and opened.pipeline.embedder is models[E5_LARGE]
    assert serving.sync is manager and opened.labels is first.labels
    assert serving.state() == (IndexState.READY, "")
    assert serving.get() is opened


def test_a_write_prepared_before_a_swap_is_refused(corpus, models):
    """Its chunks were made the way the replaced index was, and would land in the new one."""
    built(corpus, models)
    serving = handle(corpus, models, MODEL_NAME=MINILM)
    pipeline = serving.ingest_pipeline()
    handle(corpus, models).run_sync(None)

    with pytest.raises(IndexBusyError, match="replaced by a rebuilt index"):
        pipeline.ingest_data("Late text.", source="late")
    assert handle(corpus, models).get().store.get_source("late") is None


def test_sync_start_after_a_swap_embeds_with_the_new_scheme(corpus, models):
    """F5: the job resolves the index when it runs, never with a pipeline taken before."""
    built(corpus, models)
    serving = handle(corpus, models)
    manager = serving.sync
    handle(corpus, models).run_sync(None)
    (corpus / "d.md").write_text("# Delta\n\nDelta body about cranes.")

    job = manager.start(None)
    manager.wait()

    status = manager.status(job)
    assert (status.state, status.counts["ingested"], status.rebuild) == ("succeeded", 1, None)
    assert status.to_dict()["rebuild"] is None
    assert serving.sync is manager
    delta = [r for r in chunk_rows(serving) if r["id"].startswith(str(corpus / "d.md"))]
    assert delta and same_vectors(delta, models[E5_LARGE])


def test_an_unscoped_sync_start_rebuilds_and_reports_it(corpus, models):
    built(corpus, models)
    serving = handle(corpus, models)
    job = serving.start_sync(None)
    serving.sync.wait()
    status = serving.sync.status(job)
    assert status.state == "succeeded", status.error
    assert status.to_dict()["rebuild"] == {
        "kind": "reembed",
        "reasons": [f"model {MINILM} -> {E5_LARGE}"],
        "generation": 2,
        "reembedded": status.rebuild.reembedded,
        "restored": 0,
    }
    assert status.rebuild.reembedded > 0
    assert serving.get().identity.generation == 2


def test_an_interrupted_swap_is_finished_on_open(corpus, models, monkeypatch):
    """Review Focus 3: a rebuild killed between the swap's renames leaves no `db`. Opened
    as it is, that would be a fresh empty index answering "nothing indexed"."""
    a = built(corpus, models)
    chunks = a.get().store.chunk_count()
    db = a.config.db_path
    real = identity_mod.os.rename

    def rename(src, dst) -> None:
        if Path(src) == staging_dir(db):
            raise Killed
        real(src, dst)

    monkeypatch.setattr(identity_mod.os, "rename", rename)
    with pytest.raises(Killed):
        handle(corpus, models).run_sync(None)
    monkeypatch.setattr(identity_mod.os, "rename", real)
    assert not db.exists() and previous_dir(db).is_dir()
    assert read_identity(staging_dir(db)) is not None

    opened = a.get()  # open across the crash
    assert opened.identity.generation == 2 and opened.identity.embedding.model == E5_LARGE
    assert opened.store.chunk_count() == chunks
    assert not staging_dir(db).exists()
    fresh = handle(corpus, models).get()
    assert fresh.identity.generation == 2 and fresh.store.chunk_count() == chunks


def test_an_interrupted_swap_is_finished_by_the_next_sync(corpus, models, monkeypatch):
    a = built(corpus, models)
    chunks = a.get().store.chunk_count()
    db = a.config.db_path
    manager = handle(corpus, models).sync  # a server's, taken before the crash
    real = identity_mod.os.rename

    def rename(src, dst) -> None:
        if Path(src) == staging_dir(db):
            raise Killed
        real(src, dst)

    monkeypatch.setattr(identity_mod.os, "rename", rename)
    with pytest.raises(Killed):
        handle(corpus, models).run_sync(None)
    monkeypatch.setattr(identity_mod.os, "rename", real)

    job = manager.start(None)
    manager.wait()
    status = manager.status(job)
    assert (status.state, status.rebuild) == ("succeeded", None), status.error
    assert read_identity(db).generation == 2
    assert handle(corpus, models).get().store.chunk_count() == chunks
    assert not staging_dir(db).exists()


def _late_record(models) -> ChunkRecord:
    return ChunkRecord(
        id="late#0",
        source="late",
        source_type="data",
        title="late",
        chunk_index=0,
        text="Late text.",
        vector=models[MINILM].embed_documents(["Late text."])[0],
        file_hash="",
        mtime=0.0,
        ingested_at=datetime.now(UTC).isoformat(),
    )


def test_a_swap_between_the_hook_and_the_write_refuses_the_write(corpus, models):
    """The hook passed against generation 1; the store checks again under the lock a swap
    needs, since a store opened before a swap writes into the new index at its path."""
    built(corpus, models)
    pipeline = handle(corpus, models, MODEL_NAME=MINILM).ingest_pipeline()
    hook = pipeline.before_store

    def then_swap() -> None:
        hook()
        handle(corpus, models).run_sync(None)

    pipeline.before_store = then_swap
    with pytest.raises(IndexBusyError, match="replaced by a rebuilt index"):
        pipeline.ingest_data("Late text.", source="late")
    assert handle(corpus, models).get().store.get_source("late") is None


def test_a_write_waiting_out_a_swap_is_refused_after_it(corpus, models, monkeypatch):
    built(corpus, models)
    serving = handle(corpus, models, MODEL_NAME=MINILM)
    store = serving.get().store
    db = serving.config.db_path
    outcome: dict[str, str] = {}

    def write() -> None:
        try:
            store.replace_source("late", [_late_record(models)])
            outcome["late"] = "written"
        except IndexBusyError as e:
            outcome["late"] = str(e)

    writer = threading.Thread(target=write)
    real = identity_mod.os.rename

    def rename(src, dst) -> None:
        if Path(src) == db and writer.ident is None:
            writer.start()  # under the swap's exclusive lock: the write waits for it
            time.sleep(0.3)
        real(src, dst)

    monkeypatch.setattr(identity_mod.os, "rename", rename)
    handle(corpus, models).run_sync(None)
    writer.join(30)

    assert "replaced by a rebuilt index" in outcome["late"]
    assert handle(corpus, models).get().store.get_source("late") is None


def test_a_reingest_without_ocr_keeps_the_text_ocr_read(corpus, models, monkeypatch):
    """A PDF with a text layer reads without error on an install without OCR, and the
    text OCR gave its scanned pages would silently be gone."""
    scan = corpus / "scan.pdf"
    scan.write_bytes(b"%PDF-1.4 stand-in")
    with_ocr = ParsedDoc(
        "# Scan\n\nText layer page.\n\nScanned page read by OCR: invoice 4711.",
        "Scan",
        ocr_engine="rapidocr",
    )
    without = ParsedDoc("# Scan\n\nText layer page.", "Scan")
    reads = {"ocr": True}
    real = pipeline_mod.parse_file

    def parse(path, config=None):
        if Path(path) == scan:
            return with_ocr if reads["ocr"] else without
        return real(path, config)

    monkeypatch.setattr(pipeline_mod, "parse_file", parse)
    built(corpus, models, CHUNK_TOKEN_BUDGET="32")
    reads["ocr"] = False
    monkeypatch.setattr(ocr_mod, "available", lambda: False)

    _, errors, info = handle(corpus, models, MODEL_NAME=MINILM).run_sync(None)

    store = handle(corpus, models, MODEL_NAME=MINILM).get().store
    assert "invoice 4711" in join_document(store.all_chunks(str(scan)))
    assert store.get_source(str(scan)).ocr_engine == "rapidocr"
    assert {"source": str(scan), "error": KEPT} in errors
    assert info.restored == 1


def test_a_rebuild_that_fails_mid_build_leaves_the_previous_index_as_it_was(corpus, models):
    """`<db>.previous` is the user's rollback, kept until the next rebuild is swapped in."""
    built(corpus, models)
    handle(corpus, models).run_sync(None)
    db = handle(corpus, models).config.db_path
    kept = read_identity(previous_dir(db))
    files = sorted(p.relative_to(previous_dir(db)) for p in previous_dir(db).rglob("*"))
    seen: dict[str, bool] = {}

    def fail() -> None:
        seen["previous"] = previous_dir(db).exists()
        raise RuntimeError("the embedding endpoint went away")

    models[MINILM] = Hooked(MINILM, salt="minilm", during=fail)
    with pytest.raises(RuntimeError, match="went away"):
        handle(corpus, models, MODEL_NAME=MINILM).run_sync(None)

    assert seen == {"previous": True}
    assert read_identity(previous_dir(db)) == kept and kept.generation == 1
    assert sorted(p.relative_to(previous_dir(db)) for p in previous_dir(db).rglob("*")) == files
    assert read_identity(db).generation == 2
    _, _, info = handle(corpus, models, MODEL_NAME=MINILM).run_sync(None)
    assert info.generation == 3 and read_identity(previous_dir(db)).generation == 2


def test_a_kill_while_the_replaced_previous_is_deleted_keeps_the_rebuild(
    corpus, models, monkeypatch
):
    """The delete of the index kept from the swap before takes seconds; the rebuild is in
    place before it starts, so a kill there never throws the rebuild away."""
    built(corpus, models)
    handle(corpus, models).run_sync(None)
    db = handle(corpus, models).config.db_path
    real = shutil.rmtree

    def dies_midway(path, *args, **kwargs) -> None:
        if Path(path).name.startswith(previous_dir(db).name):
            (Path(path) / INDEX_FILE).unlink()
            raise Killed
        real(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", dies_midway)
    with pytest.raises(Killed):
        handle(corpus, models, MODEL_NAME=MINILM).run_sync(None)
    monkeypatch.setattr(shutil, "rmtree", real)

    assert read_identity(db).generation == 3 and read_identity(db).embedding.model == MINILM
    assert read_identity(previous_dir(db)).generation == 2
    _, _, info = handle(corpus, models).run_sync(None)
    assert info.generation == 4 and read_identity(previous_dir(db)).generation == 3
    assert sorted(p.name for p in db.parent.iterdir() if p.name.startswith(db.name)) == [
        db.name,
        previous_dir(db).name,
    ]


def test_a_previous_that_cannot_be_deleted_is_reported_and_the_rebuild_kept(
    corpus, models, monkeypatch
):
    built(corpus, models)
    handle(corpus, models).run_sync(None)
    db = handle(corpus, models).config.db_path
    real = shutil.rmtree

    def undeletable(path, *args, **kwargs) -> None:
        if Path(path).name.startswith(previous_dir(db).name):
            if not kwargs.get("ignore_errors", args[0] if args else False):
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
            return
        real(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", undeletable)
    counts, errors, info = handle(corpus, models, MODEL_NAME=MINILM).run_sync(None)

    assert info.generation == 3 and read_identity(db).generation == 3
    assert read_identity(previous_dir(db)).generation == 2
    [trash] = [p for p in db.parent.iterdir() if p.name.endswith(".trash")]
    assert read_identity(trash).generation == 1
    [error] = errors
    assert error["source"] == str(trash) and "could not be deleted" in error["error"]
    assert counts["failed"] == 0


def test_a_swap_that_fails_keeps_the_previous_index(corpus, models, monkeypatch):
    """`<db>.previous` is dropped only by a swap that succeeded."""
    built(corpus, models)
    handle(corpus, models).run_sync(None)
    db = handle(corpus, models).config.db_path
    kept = read_identity(previous_dir(db))
    real = identity_mod.os.rename

    def rename(src, dst) -> None:
        if Path(src) == staging_dir(db):
            raise OSError(errno.EXDEV, "Cross-device link", str(src))
        real(src, dst)

    monkeypatch.setattr(identity_mod.os, "rename", rename)
    with pytest.raises(OSError, match="Cross-device"):
        handle(corpus, models, MODEL_NAME=MINILM).run_sync(None)
    monkeypatch.setattr(identity_mod.os, "rename", real)

    assert read_identity(db).generation == 2 and read_identity(previous_dir(db)) == kept
    assert not [p for p in db.parent.iterdir() if p.name.endswith(".trash")]


def test_a_rebuild_says_what_it_rebuilds_before_it_starts(corpus, models):
    """A changed default turns a routine sync into a reingest of hours."""
    built(corpus, models)
    events: list[str] = []
    h = handle(corpus, models)
    h.run_sync(None, on_event=events.append)
    reason = f"model {MINILM} -> {E5_LARGE}"
    assert events[0].startswith(f"rebuilding {h.config.db_path} (reembed: {reason})")


def test_a_running_rebuild_names_its_kind_and_reasons_in_sync_status(corpus, models):
    built(corpus, models)
    serving = handle(corpus, models)
    jobs: list[str] = []
    started = threading.Event()
    seen: dict = {}

    def during() -> None:
        assert started.wait(30)
        seen.update(serving.sync.status(jobs[0]).to_dict())

    models[E5_LARGE] = Hooked(E5_LARGE, salt="e5", during=during)
    jobs.append(serving.start_sync(None))
    started.set()
    serving.sync.wait()

    assert seen["state"] == "running"
    assert seen["rebuild"] == {
        "kind": "reembed",
        "reasons": [f"model {MINILM} -> {E5_LARGE}"],
        "generation": None,
        "reembedded": None,
        "restored": None,
    }
    assert serving.sync.status(jobs[0]).to_dict()["rebuild"]["generation"] == 2


def test_a_reembed_embeds_the_stored_text_with_the_new_schemes_prefixes(
    tmp_path, fake_llama, tokenizer_file, reference_probes
):
    """Through the embedder `make_embedder` builds for the scheme, as a real rebuild does:
    a factory that hands out one embedder per model would never show a prefix lost."""
    from dataclasses import replace

    from minirag_mcp.endpoints import Endpoint
    from minirag_mcp.handle import make_embedder
    from minirag_mcp.identity import EmbeddingScheme

    fake_llama.dim = 1024
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# Alpha\n\nAlpha body about budgets and reviews.")
    config = replace(
        load_config({"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}, cwd=root),
        embed_endpoints={E5_LARGE: Endpoint(fake_llama.url("/v1/embeddings"), "e5")},
    )
    unprefixed = EmbeddingScheme(E5_LARGE, 1024, "none")
    shipped = dict(reference_probes)
    reference_probes.clear()
    with pytest.warns(RuntimeWarning, match="cannot be verified"):
        IndexHandle(config, embedder_factory=lambda s: make_embedder(config, unprefixed)).run_sync(
            None
        )
    reference_probes.update(shipped)
    h = IndexHandle(config)
    texts = [r["text"] for r in chunk_rows(h)]
    fake_llama.requests.clear()

    _, _, info = h.run_sync(None)

    assert info.kind == "reembed"
    assert info.reasons == ("query/passage prefixes none -> e5",)
    sent = [t for r in fake_llama.requests for t in r.body["input"]]
    assert sorted(t for t in sent if not t.startswith("passage: " + "minirag index probe")) == [
        "passage: " + t for t in sorted(texts)
    ]
    rows = chunk_rows(h)
    expected = [FakeEmbedder(dim=1024)._vec("passage: " + r["text"]) for r in rows]
    assert np.allclose([r["vector"] for r in rows], expected, atol=1e-6)
    fake_llama.requests.clear()
    opened = h.get()
    run_query(opened.store, opened.embedder, h.config, "budgets")
    assert [t for r in fake_llama.requests for t in r.body["input"]][-1] == "query: budgets"


def test_a_short_copy_is_never_swapped_in(corpus, models):
    built(corpus, models)
    b = handle(corpus, models)
    store = b.get().store
    n = store.chunk_count()
    store.chunk_count = lambda: n + 1
    with pytest.raises(RuntimeError, match=f"read {n} of the {n + 1} chunks"):
        b.run_sync(None)
    db = b.config.db_path
    assert read_identity(db).generation == 1 and not staging_dir(db).exists()
    assert b.state()[0] is IndexState.REBUILD_PENDING


def test_a_rebuild_whose_new_generation_cannot_be_opened_still_reports_it(
    corpus, models, monkeypatch
):
    """Generation 2 is in place: the sync succeeded, and the next call opens it."""
    built(corpus, models)
    b = handle(corpus, models)
    real = rebuild_mod.swap_in

    def boom(pool, reranker):
        raise RuntimeError("cannot open it now")

    def swap(db) -> list[Path]:
        left = real(db)
        b._build = boom
        return left

    monkeypatch.setattr(rebuild_mod, "swap_in", swap)
    counts, errors, info = b.run_sync(None)
    del b._build

    db = b.config.db_path
    assert info.generation == 2 and read_identity(db).generation == 2
    assert errors == [
        {
            "source": str(db),
            "error": "rebuilt as generation 2, but opening it failed: cannot open it now",
        }
    ]
    assert b.get().identity.generation == 2 and b.state() == (IndexState.READY, "")
