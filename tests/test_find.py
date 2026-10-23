import unicodedata
from pathlib import Path

import pytest

from conftest import FakeModelFactory
from label_helpers import L, record
from minirag_mcp.config import load_config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.labels import LabelerSource
from minirag_mcp.literal import matches, sql_pattern, sql_subject, term
from minirag_mcp.results import (
    EMPTY_INDEX_NOTICE,
    FILTERED_OUT_NOTICE,
    MAX_FIND_LIMIT,
    MAX_TERMS,
    SNIPPET_CHARS,
    QueryError,
    run_find,
)
from minirag_mcp.store import MAX_TOP_K, ChunkRecord, Store


def rec(source, i, text, title="T", labels_json=""):
    return ChunkRecord(
        id=f"{source}#{i}",
        source=source,
        source_type="file",
        title=title,
        chunk_index=i,
        text=text,
        vector=[1.0] + [0.0] * 7,
        file_hash="h",
        mtime=1.0,
        ingested_at="2026-10-23T00:00:00+00:00",
        labels_json=labels_json,
    )


def test_candidates_are_not_capped_at_top_k(tmp_path):
    store = Store(tmp_path / "db", dim=8)
    n = MAX_TOP_K + 20
    for i in range(n):
        store.replace_source(f"/d/{i:03}.md", [rec(f"/d/{i:03}.md", 0, f"Иванов, письмо {i}")])
    store.replace_source("/d/other.md", [rec("/d/other.md", 0, "Петров")])
    got = store.literal_candidates([term("иванов")])
    assert len({r.source for r in got}) == n


def test_candidates_cover_an_nfd_title_and_yo(tmp_path):
    store = Store(tmp_path / "db", dim=8)
    title = unicodedata.normalize("NFD", "Свод показателей")
    store.replace_source("/a.md", [rec("/a.md", 0, "body", title=title)])
    store.replace_source("/b.md", [rec("/b.md", 0, "Годовой ОТЧЁТ")])
    store.replace_source("/c.md", [rec("/c.md", 0, "nothing")])
    store.replace_source("/d.md", [rec("/d.md", 0, unicodedata.normalize("NFD", "отчёт"))])
    assert {r.source for r in store.literal_candidates([term("показателей")])} == {"/a.md"}
    assert {r.source for r in store.literal_candidates([term("отчет")])} == {"/b.md", "/d.md"}


def test_candidates_respect_scope_and_filter(tmp_path):
    store = Store(tmp_path / "db", dim=8)
    store.replace_source("/x/a.md", [rec("/x/a.md", 0, "акт", labels_json="A")])
    store.replace_source("/x/b.md", [rec("/x/b.md", 0, "акт", labels_json="B")])
    store.replace_source("/y/c.md", [rec("/y/c.md", 0, "акт", labels_json="A")])
    got = store.literal_candidates([term("акт")], scopes=("/x",), filter_sql="labels_json = 'A'")
    assert [r.source for r in got] == ["/x/a.md"]


def test_a_quote_in_a_term_does_not_break_the_query(tmp_path):
    store = Store(tmp_path / "db", dim=8)
    store.replace_source("/a.md", [rec("/a.md", 0, "it's O'Brien's memo")])
    assert [r.source for r in store.literal_candidates([term("o'brien")])] == ["/a.md"]


def test_candidates_come_back_in_document_order(tmp_path):
    store = Store(tmp_path / "db", dim=8)
    store.replace_source("/a.md", [rec("/a.md", i, f"акт {i}") for i in (2, 0, 1)])
    got = store.literal_candidates([term("акт")])
    assert [(r.source, r.chunk_index) for r in got] == [("/a.md", 0), ("/a.md", 1), ("/a.md", 2)]


# --- run_find: the payload both shells return ------------------------------------------

ROOT = Path("/d")


def docs(store, texts: dict[str, list[str] | str], title="T"):
    for name, chunks in texts.items():
        chunks = [chunks] if isinstance(chunks, str) else chunks
        src = f"/d/{name}"
        store.replace_source(src, [rec(src, i, t, title=title) for i, t in enumerate(chunks)])


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db", dim=8)


def find(store, terms, **kw):
    return run_find(store, (ROOT,), terms, **kw)


def test_every_matching_document_is_listed_beyond_any_top_k(store):
    n = MAX_TOP_K + 20
    docs(store, {f"{i:03}.md": f"Акт № {i} подписан" for i in range(n)})
    docs(store, {"other.md": "ничего похожего"})
    got = find(store, "акт", limit=MAX_FIND_LIMIT)
    assert got["searchedDocuments"] == n + 1
    assert got["matchedDocuments"] == n
    assert len(got["documents"]) == n
    assert "nextOffset" not in got


def test_paging_walks_every_match_once(store):
    docs(store, {f"{i}.md": "акт" for i in range(7)})
    seen, offset = [], 0
    while offset is not None:
        page = find(store, "акт", limit=3, offset=offset)
        seen += [d["source"] for d in page["documents"]]
        offset = page.get("nextOffset")
    assert sorted(seen) == sorted(f"/d/{i}.md" for i in range(7))
    assert len(seen) == 7


def test_a_term_inside_another_word_is_not_a_mention(store):
    docs(store, {"a.md": "акт сверки", "b.md": "контакт и фактически"})
    got = find(store, "акт")
    assert [d["source"] for d in got["documents"]] == ["/d/a.md"]
    assert got["matchedDocuments"] == 1


def test_match_all_needs_every_term_somewhere_in_the_document(store):
    docs(store, {"both.md": ["Иванов И.И.", "проект договора"], "one.md": "Иванов"})
    assert [d["source"] for d in find(store, ["иванов", "договор"])["documents"]] == ["/d/both.md"]
    got = find(store, ["иванов", "договор"], match="any")
    rows = {d["source"]: d for d in got["documents"]}
    assert set(rows) == {"/d/both.md", "/d/one.md"}
    assert rows["/d/one.md"]["terms"] == ["иванов"]
    assert rows["/d/both.md"]["terms"] == ["иванов", "договор"]


def test_a_single_term_request_carries_no_per_document_terms(store):
    docs(store, {"a.md": "акт"})
    assert "terms" not in find(store, "акт")["documents"][0]


def test_hits_and_first_match_locate_the_mention(store):
    docs(store, {"a.md": ["вступление", "Подписан\n\n  АКТ сверки №5", "ещё акты"]})
    (row,) = find(store, "акт")["documents"]
    assert row["hits"] == 2
    assert row["inTitle"] is False
    assert row["displayPath"] == "a.md"
    assert row["firstMatch"]["chunkIndex"] == 1
    assert row["firstMatch"]["snippet"] == "Подписан АКТ сверки №5"


def test_a_long_chunk_gives_a_window_around_the_mention(store):
    text = "слово " * 100 + "ДОГОВОР поставки " + "хвост " * 100
    docs(store, {"a.md": text})
    snippet = find(store, "договор")["documents"][0]["firstMatch"]["snippet"]
    assert "ДОГОВОР поставки" in snippet
    assert snippet.startswith("…") and snippet.endswith("…")
    assert len(snippet) <= SNIPPET_CHARS + 2


def test_a_title_match_is_listed_first_and_has_no_snippet(store):
    docs(store, {"body.md": ["акт", "акт", "акт"]})
    store.replace_source("/d/title.md", [rec("/d/title.md", 0, "текст", title="Акт сверки")])
    got = find(store, "акт")
    assert [d["source"] for d in got["documents"]] == ["/d/title.md", "/d/body.md"]
    title_row = got["documents"][0]
    assert title_row["inTitle"] is True
    assert title_row["hits"] == 0
    assert title_row["firstMatch"] is None


def test_matched_chunks_counts_text_matches_across_documents(store):
    docs(store, {"a.md": ["акт", "нет", "акт"], "b.md": "акт"})
    assert find(store, "акт")["matchedChunks"] == 3


def test_absence_is_stated_with_what_was_searched(store):
    docs(store, {"a.md": "контакт", "b.md": "ничего"})
    got = find(store, "акт")
    assert got["documents"] == [] and got["matchedDocuments"] == 0
    assert "None of the 2 documents searched" in got["notice"]
    assert "from the start of a word" in got["notice"]


def test_an_empty_index_says_so_rather_than_claiming_absence(store):
    got = find(store, "акт")
    assert got["searchedDocuments"] == 0
    assert got["notice"] == EMPTY_INDEX_NOTICE


def test_a_scope_with_no_documents_says_so(store):
    docs(store, {"a.md": "акт"})
    got = run_find(store, (ROOT,), "акт", scopes=("/elsewhere",))
    assert "No indexed document is under scope" in got["notice"]


@pytest.mark.parametrize(
    ("terms", "kw", "message"),
    [
        ([], {}, "terms must not be empty"),
        (["  "], {}, "a term must not be empty"),
        ([f"t{i}" for i in range(MAX_TERMS + 1)], {}, f"at most {MAX_TERMS} terms"),
        ("акт", {"match": "some"}, "match must be 'all' or 'any'"),
        ("акт", {"limit": 0}, "limit must be at least 1"),
        ("акт", {"offset": -1}, "offset must not be negative"),
    ],
)
def test_requests_that_cannot_run_are_refused(store, terms, kw, message):
    with pytest.raises(QueryError, match=message):
        find(store, terms, **kw)


def test_an_oversized_limit_is_clamped(store):
    docs(store, {f"{i:03}.md": "акт" for i in range(MAX_FIND_LIMIT + 5)})
    got = find(store, "акт", limit=10**6)
    assert len(got["documents"]) == MAX_FIND_LIMIT
    assert got["nextOffset"] == MAX_FIND_LIMIT


TAXONOMY = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.type]
values.minutes = { description = "Minutes." }
values.spec = { description = "Spec." }
"""


def test_a_label_filter_narrows_what_is_searched(tmp_path):
    root = tmp_path / "docs"
    (root / ".minirag").mkdir(parents=True)
    (root / ".minirag" / "classify.toml").write_text(TAXONOMY)
    store = Store(tmp_path / "db", dim=8)
    for name, kind in (("m.md", "minutes"), ("s.md", "spec")):
        src = str(root / name)
        store.replace_source(src, [record(src, {"type": L(kind)}, text="акт сверки")])
    source = LabelerSource((root,), tmp_path / "cache", FakeModelFactory())
    got = run_find(store, (root,), "акт", labels={"type": "minutes"}, label_source=source)
    assert [d["displayPath"] for d in got["documents"]] == ["m.md"]
    assert got["searchedDocuments"] == 1
    assert got["filterCounts"]["unlabelled"] == 0


def test_forms_count_the_documents_each_word_form_appears_in(store):
    docs(store, {"a.md": "акт и акта", "b.md": "актуально", "c.md": ["акт", "акт"]})
    forms = find(store, "акт")["forms"]
    assert forms == {"акт": {"акт": 2, "акта": 1, "актуально": 1}}
    assert list(forms["акт"]) == ["акт", "акта", "актуально"]


def test_a_quoted_term_excludes_other_forms(store):
    docs(store, {"a.md": "Акт сверки", "b.md": "два акта", "c.md": "актуально"})
    got = find(store, '"акт"')
    assert [d["source"] for d in got["documents"]] == ["/d/a.md"]
    assert got["forms"] == {'"акт"': {"акт": 1}}


def test_an_absent_quoted_term_is_named_once_quoted(store):
    docs(store, {"a.md": "два акта"})
    notice = find(store, '"акт"')["notice"]
    assert 'contain "акт" as a whole word' in notice
    assert '""' not in notice


# --- the SQL prefilter never drops what Python accepts ---------------------------------


def _sweep_words():
    """Every BMP character the fold could get wrong, in a word and at a word's end.

    Left out: characters with no case, no decomposition and no special handling, such
    as CJK ideographs, which both sides pass through unchanged. Checking those too
    triples the run time and has nothing to find.
    """
    for cp in range(0x20, 0x10000):
        c = chr(cp)
        category = unicodedata.category(c)
        if category in ("Cs", "Co", "Cn", "Cc"):
            continue
        if not (
            cp < 0x80
            or category in ("Lu", "Ll", "Lt", "Mn", "Mc", "Me", "Cf")
            or c.lower() != c
            or c.upper() != c
            or unicodedata.normalize("NFD", c) != c
            or unicodedata.normalize("NFC", c) != c
        ):
            continue
        yield "a" + c + "b"
        yield "AB" + c


def test_the_sql_prefilter_keeps_every_row_the_rule_accepts(tmp_path):
    """A row the SQL drops is never shown to the Python rule, so the tool would answer
    "none" for a document that has the term. Checked by DataFusion's own `lower` and Rust
    regex, which are what the store runs, not by a Python stand-in for them."""
    rows = []
    for w in _sweep_words():
        if unicodedata.normalize("NFC", w) != w and unicodedata.normalize("NFD", w) != w:
            continue  # stored in neither form: the documented limit
        try:
            t = term(w)
        except ValueError:
            continue
        if matches([t], w)[0] is not None:
            rows.append({"text": w, "pat": sql_pattern([t])})
    import lancedb

    table = lancedb.connect(tmp_path).create_table("t", rows)
    missed = table.search().where(f"NOT regexp_like({sql_subject('text')}, pat)").limit(len(rows))
    assert [r["text"] for r in missed.to_list()] == []
    assert len(rows) > 25_000


# --- what real ingestion stores ----------------------------------------------------------


def _pipeline(tmp_path, fake_embedder):
    root = tmp_path / "docs"
    root.mkdir()
    cfg = load_config({"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")})
    store = Store(cfg.db_path, dim=fake_embedder.dim)
    return root, store, Pipeline(store, fake_embedder, cfg)


def test_a_code_in_converted_html_is_found(tmp_path, fake_embedder):
    root, store, pipeline = _pipeline(tmp_path, fake_embedder)
    pipeline.ingest_data("<p>Согласно ТЗ_2024_015 итого 5*3.</p>", source="memo", fmt="html")
    for raw in ("ТЗ_2024_015", "5*3"):
        got = run_find(store, (root,), raw)
        assert [d["source"] for d in got["documents"]] == ["memo"], raw


def test_a_heading_repeated_on_every_chunk_counts_once(tmp_path, fake_embedder):
    root, store, pipeline = _pipeline(tmp_path, fake_embedder)
    body = " ".join(f"Слово номер {i} в длинном абзаце про поставки." for i in range(60))
    doc = root / "doc.md"
    doc.write_text(f"# Регламент\n\n## Акт сверки\n\n{body}\n\n{body}\n\n## Склад\n\n{body}\n")
    pipeline.ingest_file(doc)
    assert len(store.all_chunks(str(doc))) > 6
    (row,) = run_find(store, (root,), "сверки")["documents"]
    assert row["hits"] == 1
    assert run_find(store, (root,), "поставки")["documents"][0]["hits"] == len(
        store.all_chunks(str(doc))
    )


def test_paging_ends_exactly_at_a_multiple_of_the_limit(store):
    docs(store, {f"{i}.md": "акт" for i in range(6)})
    assert find(store, "акт", limit=3)["nextOffset"] == 3
    assert "nextOffset" not in find(store, "акт", limit=3, offset=3)


def test_a_filter_that_excludes_everything_says_so(tmp_path):
    root = tmp_path / "docs"
    (root / ".minirag").mkdir(parents=True)
    (root / ".minirag" / "classify.toml").write_text(TAXONOMY)
    store = Store(tmp_path / "db", dim=8)
    src = str(root / "m.md")
    store.replace_source(src, [record(src, {"type": L("minutes")}, text="акт")])
    source = LabelerSource((root,), tmp_path / "cache", FakeModelFactory())
    got = run_find(store, (root,), "акт", labels={"type": "spec"}, label_source=source)
    assert got["notice"] == FILTERED_OUT_NOTICE


def test_a_snippet_from_decomposed_text_is_composed(store):
    text = unicodedata.normalize("NFD", "Годовой отчёт за квартал")
    docs(store, {"a.md": text})
    snippet = find(store, "отчет")["documents"][0]["firstMatch"]["snippet"]
    assert snippet == unicodedata.normalize("NFC", text)


def test_a_snippet_stays_bounded_around_a_huge_token(store):
    docs(store, {"a.md": "начало акт" + "я" * 5000 + " конец"})
    snippet = find(store, "акт")["documents"][0]["firstMatch"]["snippet"]
    assert len(snippet) <= SNIPPET_CHARS + 2
