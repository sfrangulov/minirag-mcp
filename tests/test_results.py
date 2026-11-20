"""`displayPath` — what a Sources list is written from, beside the `source` it derives from.

The two fields are deliberately not the same string and are not interchangeable. `source`
is the identity key: `read_file`, `delete_file` and re-ingest all address a document by
it, so it stays the absolute path it has always been. `displayPath` exists only to be
printed, and the citation policy asks for it by name, copied verbatim.

Why the relative path and not the `title`: the title is derived — underscores become
spaces, the extension is dropped — so `И-112_ЗПС_Хранение ТМЗ.docx` reaches the user as
`И-112 ЗПС Хранение ТМЗ`, which is not the name of anything on disk. The relative path
carries the real filename, so what the user reads matches what they can find.

What these tests pin is that the field is always present, never empty, always a byte-wise
suffix of the `source` it came from when that source is under a root, and never silently
degraded to a bare filename or a re-cased one.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import minirag_mcp.embedder as embedder_mod
from conftest import FakeEmbedder, FakeModelFactory
from label_helpers import L, record
from minirag_mcp.config import Config
from minirag_mcp.embedder import E5_LARGE, Embedder
from minirag_mcp.endpoints import EmbeddingEndpoint, Endpoint, RerankEndpoint
from minirag_mcp.labels import LabelerSource
from minirag_mcp.results import (
    QueryError,
    aggregate_sources,
    display_path,
    documents_status,
    labels_status,
    list_payload,
    rerank_status,
    run_query,
)
from minirag_mcp.store import LABEL_COLUMNS, SearchResult, Store
from store_helpers import _older_table, _raise_commit_conflict, _row

ROOT = "/Users/x/Documents/corpus"

#: A real relative path from the indexed corpus: a numbered directory, a space in the
#: directory name, an underscore-and-space filename, Cyrillic, and a `.docx` extension —
#: every character class the `title` derivation would have flattened.
REAL_REL = "instructions/11. Учет затрат/И-112_ЗПС_Хранение ТМЗ.docx"


def hit(source: str, title: str = "T", chunk_index: int = 0) -> SearchResult:
    return SearchResult(
        text="body",
        source=source,
        title=title,
        chunk_index=chunk_index,
        score=1.0,
        distance=0.1,
    )


def test_every_source_carries_a_non_empty_display_path():
    """The policy says "copy `displayPath`", so a missing one is a citation that cannot
    be written. There is no fallback for the model to reach for, by design."""
    rows = aggregate_sources([hit(f"{ROOT}/a.md"), hit(f"{ROOT}/notes/b.md")], roots=[Path(ROOT)])
    assert [r["displayPath"] for r in rows] == ["a.md", "notes/b.md"]
    assert all(r["displayPath"] for r in rows)


def test_the_filename_survives_unmodified():
    """The whole reason the field is the path and not the `title`.

    `title` normalises underscores to spaces and strips the extension; a user looking for
    the file would be looking for a name that is not on disk. The path is copied byte for
    byte — separators, case, spaces, digits, extension.
    """
    source = f"{ROOT}/{REAL_REL}"
    (row,) = aggregate_sources([hit(source, "И-112 ЗПС Хранение ТМЗ")], roots=[Path(ROOT)])
    assert row["displayPath"] == REAL_REL
    assert row["displayPath"].endswith("И-112_ЗПС_Хранение ТМЗ.docx")
    assert row["source"] == source  # the identity key is untouched


def test_display_path_is_a_suffix_of_source():
    """Not a re-rendered path: the same bytes, with the root's prefix removed.

    Anything that round-tripped through a normaliser could change a separator or a
    Unicode composition and still look right in a terminal.
    """
    source = f"{ROOT}/{REAL_REL}"
    (row,) = aggregate_sources([hit(source)], roots=[Path(ROOT)])
    assert source.endswith(row["displayPath"])
    assert source == f"{ROOT}/{row['displayPath']}"


def test_the_root_that_contains_the_document_is_the_one_used():
    """Several roots are ordinary. Each document is measured against its own."""
    roots = [Path("/a/notes"), Path("/b/specs")]
    rows = aggregate_sources([hit("/b/specs/x/y.md"), hit("/a/notes/z.md")], roots=roots)
    assert [r["displayPath"] for r in rows] == ["x/y.md", "z.md"]


def test_nested_roots_resolve_to_the_innermost_one():
    """When two roots both contain the file, the more specific one owns it.

    Deterministic beats first-wins: the answer must not depend on the order the roots
    happened to be configured in.
    """
    roots = [Path("/a"), Path("/a/b")]
    (row,) = aggregate_sources([hit("/a/b/c.md")], roots=roots)
    assert row["displayPath"] == "c.md"
    assert (
        aggregate_sources([hit("/a/b/c.md")], roots=list(reversed(roots)))[0]["displayPath"]
        == "c.md"
    )


def test_a_source_outside_every_root_falls_back_to_itself():
    """A document indexed under a root that has since been reconfigured away.

    The absolute path is worse to read than a relative one and better than nothing;
    an empty string would leave the model to invent the line, which is the failure the
    field exists to remove.
    """
    (row,) = aggregate_sources([hit("/elsewhere/a.md")], roots=[Path(ROOT)])
    assert row["displayPath"] == "/elsewhere/a.md"


def test_no_roots_configured_falls_back_to_the_source():
    """`aggregate_sources` is called from the CLI too, and a caller may have none."""
    (row,) = aggregate_sources([hit("/a/b.md")])
    assert row["displayPath"] == "/a/b.md"


@pytest.mark.parametrize(
    "source",
    ["https://example.com/docs/onboarding", "notes-pasted-2026-08", "data:inline"],
)
def test_a_data_or_url_source_cites_the_id_it_was_ingested_under(source):
    """`data` and `url` items have no filesystem path at all — `source` is the id, and
    for a URL that id is the URL, which is already the most locating string there is."""
    (row,) = aggregate_sources([hit(source)], roots=[Path(ROOT)])
    assert row["displayPath"] == source


def test_a_root_prefix_that_is_not_a_path_boundary_is_not_stripped():
    """`/data` must not claim `/data_archive/x.md`. String-prefix logic gets this wrong."""
    (row,) = aggregate_sources([hit("/data_archive/x.md")], roots=[Path("/data")])
    assert row["displayPath"] == "/data_archive/x.md"


def test_the_root_itself_as_a_source_keeps_a_usable_string():
    """Degenerate but reachable: relpath would be `.`, which locates nothing."""
    (row,) = aggregate_sources([hit(ROOT)], roots=[Path(ROOT)])
    assert row["displayPath"] == ROOT


def test_display_path_helper_matches_what_aggregate_puts_on_the_row():
    """The helper is the whole derivation; `aggregate_sources` only calls it."""
    source = f"{ROOT}/{REAL_REL}"
    assert display_path(source, [Path(ROOT)]) == REAL_REL
    (row,) = aggregate_sources([hit(source)], roots=[Path(ROOT)])
    assert row["displayPath"] == display_path(source, [Path(ROOT)])


def test_aggregation_still_dedupes_in_rank_order():
    """The field is additive; the list it hangs off is unchanged."""
    rows = aggregate_sources(
        [hit(f"{ROOT}/b.md"), hit(f"{ROOT}/a.md"), hit(f"{ROOT}/b.md", chunk_index=1)],
        roots=[Path(ROOT)],
    )
    assert [r["source"] for r in rows] == [f"{ROOT}/b.md", f"{ROOT}/a.md"]
    assert rows[0]["hits"] == 2
    assert set(rows[0]) == {"source", "title", "hits", "displayPath"}


LABEL_TOML = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.type]
question = "What kind of document is this?"
values.spec = { description = "Spec. Requirements.", paths = ["specs"] }
values.minutes = { description = "Minutes. Decisions.", paths = ["minutes"] }

[facets.stage]
within = ["pilot"]
values.draft = { description = "A draft." }
values.final = { description = "A final version." }
"""


@pytest.fixture
def root(tmp_path) -> Path:
    root = (tmp_path / "docs").resolve()
    root.mkdir()
    return root


@pytest.fixture
def with_file(root, tmp_path):
    # A real file: a missing one makes the model "unavailable", whatever the fake does.
    model_file = tmp_path / "models" / "fake.gguf"
    model_file.parent.mkdir()
    model_file.write_bytes(b"gguf")
    (root / ".minirag").mkdir()
    (root / ".minirag" / "classify.toml").write_text(
        LABEL_TOML.replace("/models/fake.gguf", str(model_file))
    )


@pytest.fixture
def label_source(root, tmp_path, with_file) -> LabelerSource:
    return LabelerSource((root,), tmp_path / "cache", FakeModelFactory())


def make_config(root, tmp_path) -> Config:
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


def index(store, embedder, root, name, labels, doc_date="", create=True):
    path = root / name
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("budget review")
    store.replace_source(
        str(path),
        [
            record(
                str(path),
                labels,
                doc_date,
                "header" if doc_date else "",
                text="budget review",
                vector=embedder.embed_documents(["budget review"])[0],
                mtime=path.stat().st_mtime,
            )
        ],
    )
    return str(path)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db", dim=8)


def test_a_filtered_query_returns_matching_sources_with_labels_and_counts(
    tmp_path, root, store, fake_embedder, label_source, monkeypatch
):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    a = index(store, fake_embedder, root, "minutes/a.md", {"type": L("minutes")}, "2026-10-14")
    index(store, fake_embedder, root, "specs/b.md", {"type": L("spec")}, "2026-10-14")
    index(store, fake_embedder, root, "c.md", {})

    payload = run_query(
        store,
        fake_embedder,
        make_config(root, tmp_path),
        "budget review",
        labels={"type": "minutes"},
        label_source=label_source,
    )

    assert [s["source"] for s in payload["sources"]] == [a]
    assert payload["sources"][0]["labels"]["type"]["value"] == "minutes"
    assert payload["sources"][0]["date"] == {"value": "2026-10-14", "source": "header"}
    assert payload["filterCounts"] == {"unlabelled": 1, "undated": 0}
    assert "filterNotice" not in payload


def test_an_unfiltered_query_has_exactly_the_old_keys(
    tmp_path, root, store, fake_embedder, label_source
):
    index(store, fake_embedder, root, "minutes/a.md", {"type": L("minutes")}, "2026-10-14")
    for source in (None, label_source):
        payload = run_query(
            store,
            fake_embedder,
            make_config(root, tmp_path),
            "budget review",
            label_source=source,
        )
        assert set(payload) == {"results", "sources", "parents", "embeddingBackend", "reranked"}
    # Labels were stored, so the source row shows them; the shell keys stay unchanged.
    assert "labels" in payload["sources"][0]


def test_an_unlabelled_corpus_has_no_label_keys_on_a_source(tmp_path, root, store, fake_embedder):
    index(store, fake_embedder, root, "a.md", {})
    payload = run_query(store, fake_embedder, make_config(root, tmp_path), "budget review")
    assert "labels" not in payload["sources"][0] and "date" not in payload["sources"][0]


def test_a_filter_without_the_file_is_a_query_error(tmp_path, root, store, fake_embedder):
    with pytest.raises(QueryError, match="document labels are off"):
        run_query(
            store,
            fake_embedder,
            make_config(root, tmp_path),
            "budget review",
            labels={"type": "spec"},
            label_source=LabelerSource((root,), tmp_path / "cache", FakeModelFactory()),
        )


def test_list_sorted_by_date_puts_undated_last(tmp_path, root, store, fake_embedder):
    index(store, fake_embedder, root, "a_undated.md", {})
    index(store, fake_embedder, root, "b_old.md", {}, "2026-10-01")
    index(store, fake_embedder, root, "c_new.md", {}, "2026-10-20")
    index(store, fake_embedder, root, "d_tie.md", {}, "2026-10-01")

    payload = list_payload(store, [root], sort="date", exclude=())

    assert [Path(f["source"]).name for f in payload["files"]] == [
        "c_new.md",
        "b_old.md",
        "d_tie.md",
        "a_undated.md",
    ]
    with pytest.raises(QueryError, match="sort"):
        list_payload(store, [root], sort="size", exclude=())


def test_a_filtered_listing_keeps_matches_and_counts_unread_files_as_unlabelled(
    tmp_path, root, store, fake_embedder, label_source, monkeypatch
):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    a = index(store, fake_embedder, root, "minutes/a.md", {"type": L("minutes")})
    index(store, fake_embedder, root, "specs/b.md", {"type": L("spec")})
    index(store, fake_embedder, root, "c.md", {})
    index(store, fake_embedder, root, "pilot/d.md", {}, create=True)
    (root / "new.md").write_text("never ingested")

    payload = list_payload(
        store, [root], labels={"type": "minutes"}, label_source=label_source, exclude=()
    )

    assert [f["source"] for f in payload["files"]] == [a]
    assert payload["files"][0]["labels"]["type"]["value"] == "minutes"
    # c.md and pilot/d.md are indexed without a type; new.md is unread.
    assert payload["filterCounts"]["unlabelled"] == 3
    assert "filterNotice" not in payload


def test_a_query_names_its_embedding_backend_and_warns_on_a_fallback(
    tmp_path, root, store, fake_embedder
):
    index(store, fake_embedder, root, "a.md", {})
    payload = run_query(store, fake_embedder, make_config(root, tmp_path), "budget review")
    assert payload["embeddingBackend"] == "in-process" and "warnings" not in payload
    fallen = FakeEmbedder()
    fallen.backend = "in-process (endpoint unreachable)"
    fallen.warning = "W"
    payload = run_query(store, fallen, make_config(root, tmp_path), "budget review")
    assert payload["embeddingBackend"] == "in-process (endpoint unreachable)"
    assert payload["warnings"] == ["W"]


def test_a_query_reads_the_backend_after_embedding(tmp_path, root, store, fake_embedder):
    """The query's own embedding is what falls back, so the flag must describe it."""

    class FallsBack(FakeEmbedder):
        def embed_query(self, text):
            self.backend, self.warning = "in-process (endpoint unreachable)", "W"
            return super().embed_query(text)

    payload = run_query(store, FallsBack(), make_config(root, tmp_path), "budget review")
    assert payload["embeddingBackend"] == "in-process (endpoint unreachable)"
    assert payload["warnings"] == ["W"]


def test_only_the_reply_to_a_query_embedded_in_process_says_so(
    tmp_path, root, fake_llama, stub_text_embedding, tokenizer_file, monkeypatch
):
    """Per reply, not from the shared embedder: another query may run meanwhile."""
    monkeypatch.setattr(embedder_mod, "QUERY_TIMEOUT", 0.2)
    fake_llama.dim = 1024
    url = fake_llama.url("/v1/embeddings")
    embedder = Embedder(E5_LARGE, tmp_path, endpoint=EmbeddingEndpoint(Endpoint(url, "e5")))
    store = Store(tmp_path / "db", dim=1024)
    index(store, embedder, root, "a.md", {})
    config = make_config(root, tmp_path)
    fake_llama.delay = 1.0
    slow = run_query(store, embedder, config, "budget review")
    fake_llama.delay = 0.0
    fast = run_query(store, embedder, config, "budget review")
    assert slow["embeddingBackend"] == "in-process (endpoint timed out)"
    [note] = slow["warnings"]
    assert url in note and "in-process" in note
    assert slow["results"] and slow["results"][0]["source"] == fast["results"][0]["source"]
    assert fast["embeddingBackend"] == "endpoint" and "warnings" not in fast


def test_an_embedder_without_backend_flags_reads_as_in_process(tmp_path, root, store):
    class Bare:
        dim = 8

        def embed_query(self, text):
            return FakeEmbedder().embed_query(text)

    payload = run_query(store, Bare(), make_config(root, tmp_path), "budget review")
    assert payload["embeddingBackend"] == "in-process" and "warnings" not in payload


def test_a_listing_without_a_filter_has_exactly_the_old_keys(tmp_path, root, store, fake_embedder):
    index(store, fake_embedder, root, "a.md", {})
    payload = list_payload(store, [root], exclude=())
    assert set(payload) == {"files", "unlistedFolders"}
    assert set(payload["files"][0]) == {
        "source",
        "sourceType",
        "title",
        "state",
        "chunkCount",
        "ocrEngine",
    }


def test_a_filter_on_an_index_without_label_columns_is_refused(
    tmp_path, root, fake_embedder, label_source, monkeypatch
):
    db_path = tmp_path / "old"
    table = _older_table(db_path, set(LABEL_COLUMNS), [_row(str(root / "a.md"), 0, "x")])
    monkeypatch.setattr(type(table), "add_columns", _raise_commit_conflict)
    with pytest.warns(RuntimeWarning, match="labels_json"):
        old = Store(db_path, dim=8)
    assert set(old.missing_columns) == set(LABEL_COLUMNS)

    with pytest.raises(QueryError, match="write access"):
        run_query(
            old,
            fake_embedder,
            make_config(root, tmp_path),
            "x",
            labels={"type": "spec"},
            label_source=label_source,
        )
    with pytest.raises(QueryError, match="write access"):
        list_payload(old, [root], date_from="2026-10-01", label_source=label_source, exclude=())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert run_query(old, fake_embedder, make_config(root, tmp_path), "x")["results"]


def test_labels_status_is_empty_without_the_file_and_names_an_invalid_one(tmp_path, root, store):
    absent = LabelerSource((root,), tmp_path / "cache", FakeModelFactory())
    assert labels_status(absent, store) == {}
    assert labels_status(None, store) == {}

    (root / ".minirag").mkdir()
    (root / ".minirag" / "classify.toml").write_text("not = [valid")
    status = labels_status(absent, store)
    assert set(status) == {"labels"}
    assert status["labels"]["file"] == str(root / ".minirag" / "classify.toml")
    assert status["labels"]["error"]


def test_labels_status_counts_values_unlabelled_and_uncertain(
    tmp_path, root, store, fake_embedder, label_source, monkeypatch
):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    index(store, fake_embedder, root, "minutes/a.md", {"type": L("minutes")}, "2026-10-14")
    index(store, fake_embedder, root, "minutes/b.md", {"type": L("minutes")})
    index(
        store,
        fake_embedder,
        root,
        "c.md",
        {"type": L("spec", second="minutes", uncertain=True)},
    )
    index(store, fake_embedder, root, "d.md", {})
    index(store, fake_embedder, root, "pilot/e.md", {"stage": L("draft")}, "2026-10-02")

    labels = labels_status(label_source, store)["labels"]

    assert labels["model"] == "on demand" and "modelHint" not in labels
    type_ = labels["facets"]["type"]
    assert type_["question"] == "What kind of document is this?"
    assert type_["values"]["minutes"] == {"description": "Minutes. Decisions.", "count": 2}
    assert type_["values"]["spec"]["count"] == 1
    assert type_["unlabelled"] == 2  # d.md and pilot/e.md
    assert type_["uncertain"] == 1
    stage = labels["facets"]["stage"]
    assert stage["values"]["draft"]["count"] == 1
    assert stage["unlabelled"] == 0  # the other documents are outside `within`
    assert labels["undated"] == 3


def test_a_filter_on_a_model_facet_without_the_extra_says_what_to_install(
    tmp_path, root, store, fake_embedder, label_source, monkeypatch
):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: False)
    index(store, fake_embedder, root, "minutes/a.md", {"type": L("minutes")})

    payload = run_query(
        store,
        fake_embedder,
        make_config(root, tmp_path),
        "budget review",
        labels={"type": "minutes"},
        label_source=label_source,
    )
    assert re.search(r"minirag-mcp\[classify\]", payload["filterNotice"])
    assert "type" in payload["filterNotice"]
    assert labels_status(label_source, store)["labels"]["model"] == "unavailable"
    assert "modelHint" in labels_status(label_source, store)["labels"]


def test_status_and_a_filter_name_a_model_file_that_is_not_there(
    root, tmp_path, monkeypatch, fake_embedder
):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    missing = tmp_path / "models" / "absent.gguf"
    (root / ".minirag").mkdir()
    (root / ".minirag" / "classify.toml").write_text(
        LABEL_TOML.replace("/models/fake.gguf", str(missing))
    )
    source = LabelerSource((root,), tmp_path / "cache", FakeModelFactory())
    store = Store(tmp_path / "db", dim=fake_embedder.dim)
    reason = f"the classify model file {missing} does not exist"
    labels = labels_status(source, store)["labels"]
    assert labels["model"] == "unavailable" and labels["modelHint"] == reason
    got = run_query(
        store,
        fake_embedder,
        make_config(root, tmp_path),
        "x",
        labels={"type": "spec"},
        label_source=source,
    )
    assert got["filterNotice"].endswith(f"which cannot run here: {reason}")


def test_status_counts_sources_without_a_document_row(tmp_path, root, store, fake_embedder):
    index(store, fake_embedder, root, "a.md", {})
    assert documents_status(store) == {}

    path = root / "b.md"
    path.write_text("budget review")
    store._table.add([asdict(record(str(path), {}, "", "", text="budget review"))])
    status = documents_status(store)
    assert status["undocumentedSourceCount"] == 1
    assert "sync" in status["documentsWarning"]

    store.backfill_documents()
    assert documents_status(store) == {}


class _Fixed:
    """Embeds every query as the same vector, so the keyword-free ranking is known."""

    dim = 8

    def embed_query(self, text):
        return [0.0, 1.0] + [0.0] * 6


def test_a_query_ranks_with_the_configured_document_weight(tmp_path, root, store):
    """/a.md holds both query terms only as a whole document; /b.md's chunk is nearest."""
    near, apart = [0.0, 1.0] + [0.0] * 6, [1.0, 0.0] + [0.0] * 6
    for name, texts, vector in (
        ("b.md", ["gamma delta"], near),
        ("a.md", ["alpha", "beta"], apart),
    ):
        source = str(root / name)
        store.replace_source(
            source,
            [
                replace(
                    record(source, {}, "", "", text=text, vector=vector),
                    id=f"{source}#{i}",
                    chunk_index=i,
                )
                for i, text in enumerate(texts)
            ],
        )

    def first(document_weight):
        config = replace(
            make_config(root, tmp_path), hybrid_weight=0.0, document_weight=document_weight
        )
        payload = run_query(store, _Fixed(), config, "alpha beta")
        return Path(payload["sources"][0]["source"]).name

    assert first(0.0) == "b.md"
    assert first(0.5) == "a.md"


def _two_chunks(store, embedder, root):
    path = root / "r.md"
    path.write_text("x")
    texts = ["budget alpha", "budget review plan"]
    store.replace_source(
        str(path),
        [
            replace(
                record(
                    str(path),
                    {},
                    text=t,
                    vector=embedder.embed_documents([t])[0],
                    mtime=path.stat().st_mtime,
                ),
                id=f"{path}#{i}",
                chunk_index=i,
            )
            for i, t in enumerate(texts)
        ],
    )


def test_a_served_reranker_reorders_and_says_so(tmp_path, root, store, fake_embedder, fake_llama):
    _two_chunks(store, fake_embedder, root)
    rr = RerankEndpoint(Endpoint(fake_llama.url("/v1/rerank"), "rr"))
    cfg = make_config(root, tmp_path)
    plain = run_query(store, fake_embedder, cfg, "review plan")
    payload = run_query(store, fake_embedder, cfg, "review plan", reranker=rr)
    assert payload["reranked"] is True and "warnings" not in payload
    assert payload["results"][0]["text"] == "budget review plan"
    assert payload["results"][0]["rerankScore"] == 0.0  # fake: 2 shared words - 2.0
    assert payload["results"][0]["score"] == next(
        r["score"] for r in plain["results"] if r["text"] == "budget review plan"
    )
    assert all("rerankScore" not in r for r in plain["results"])


def test_an_unreachable_reranker_is_flagged_in_the_result_and_in_status(
    tmp_path, root, store, fake_embedder, dead_url
):
    _two_chunks(store, fake_embedder, root)
    rr = RerankEndpoint(Endpoint(dead_url + "/v1/rerank", "rr"))
    payload = run_query(
        store, fake_embedder, make_config(root, tmp_path), "review plan", reranker=rr
    )
    assert payload["reranked"] is False and payload["results"]
    assert payload["warnings"][0].startswith("Reranking was skipped")
    assert rerank_status(rr) == {"reranker": "unreachable", "rerankWarning": rr.last_error}


def test_without_a_reranker_reranked_is_false(tmp_path, root, store, fake_embedder):
    _two_chunks(store, fake_embedder, root)
    payload = run_query(store, fake_embedder, make_config(root, tmp_path), "review plan")
    assert payload["reranked"] is False and "warnings" not in payload
    assert rerank_status(None) == {"reranker": "off"}


def test_an_oversize_passage_is_capped_in_the_request_and_reranking_succeeds(
    tmp_path, root, store, fake_embedder, fake_llama
):
    path = root / "big.md"
    path.write_text("x")
    text = "review " * 3000
    store.replace_source(
        str(path),
        [
            record(
                str(path),
                {},
                text=text,
                vector=fake_embedder.embed_documents([text])[0],
                mtime=path.stat().st_mtime,
            )
        ],
    )
    rr = RerankEndpoint(Endpoint(fake_llama.url("/v1/rerank"), "rr"))
    payload = run_query(store, fake_embedder, make_config(root, tmp_path), "review", reranker=rr)
    (sent,) = [r for r in fake_llama.requests if r.path == "/v1/rerank"]
    assert all(len(d) <= 4000 for d in sent.body["documents"])
    assert payload["reranked"] is True and payload["results"][0]["text"] == text


def test_a_wrong_score_count_is_flagged_not_silent(tmp_path, root, store, fake_embedder):
    _two_chunks(store, fake_embedder, root)

    class Short:
        last_error = None

        def scores(self, query, passages):
            return [1.0]

    payload = run_query(
        store, fake_embedder, make_config(root, tmp_path), "review", reranker=Short()
    )
    assert payload["reranked"] is False
    assert payload["warnings"][0].startswith("Reranking was skipped")


def test_a_hung_reranker_costs_one_short_wait_then_is_not_called_until_the_window_passes(
    tmp_path, root, store, fake_embedder, fake_llama, monkeypatch
):
    import time

    import minirag_mcp.endpoints as endpoints

    _two_chunks(store, fake_embedder, root)
    monkeypatch.setattr(endpoints, "RERANK_TIMEOUT", 0.3)
    now = [1000.0]
    monkeypatch.setattr(endpoints.time, "monotonic", lambda: now[0])
    rr = RerankEndpoint(Endpoint(fake_llama.url("/v1/rerank"), "rr"))  # default 120 s
    cfg = make_config(root, tmp_path)
    fake_llama.delay = 1.5

    started = time.perf_counter()
    first = run_query(store, fake_embedder, cfg, "review plan", reranker=rr)
    assert time.perf_counter() - started < 1.2
    assert first["reranked"] is False and first["results"]
    assert first["warnings"][0].startswith("Reranking was skipped")
    assert "not retried for 60 s" in first["warnings"][0]
    calls = len(fake_llama.requests)

    fake_llama.delay = 0.0
    now[0] += 30
    second = run_query(store, fake_embedder, cfg, "review plan", reranker=rr)
    assert len(fake_llama.requests) == calls
    assert second["reranked"] is False
    assert "not retried for another 30 s" in second["warnings"][0]
    assert rr.last_error and "30 s" not in rr.last_error

    now[0] += 31
    third = run_query(store, fake_embedder, cfg, "review plan", reranker=rr)
    assert len(fake_llama.requests) == calls + 1
    assert third["reranked"] is True and "warnings" not in third
