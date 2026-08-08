import pytest

from minirag_mcp.store import ChunkRecord, Store, relevance_cutoff


def rec(source, i, text, vec):
    return ChunkRecord(
        id=f"{source}#{i}",
        source=source,
        source_type="file",
        title="T",
        chunk_index=i,
        text=text,
        vector=vec,
        file_hash="h",
        mtime=1.0,
        ingested_at="2026-08-07T00:00:00+00:00",
    )


def V(x):  # 8-dim vector pointing "x of the way" between axis 0 and axis 1
    return [1.0 - x, x] + [0.0] * 6


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db", dim=8)
    s.replace_source(
        "/auth.md",
        [
            rec("/auth.md", 0, "OAuth2 token authentication flow", V(0.0)),
            rec("/auth.md", 1, "The API returns ERR_CONNECTION_REFUSED when down", V(0.1)),
        ],
    )
    s.replace_source("/cook.md", [rec("/cook.md", 0, "Borscht recipe with beets", V(1.0))])
    return s


def test_vector_only_ordering(store):
    got = store.search("anything", V(0.05), top_k=3, hybrid_weight=0.0)
    assert got[0].source == "/auth.md"
    assert got[0].distance is not None
    assert [g.source for g in got].count("/cook.md") == 1


def test_keyword_boost_lifts_exact_term(store):
    # Query vector points AT the recipe; only the keyword boost can lift the auth chunk.
    got = store.search("ERR_CONNECTION_REFUSED", V(1.0), top_k=3, hybrid_weight=1.0)
    assert got[0].text.startswith("The API returns ERR_CONNECTION_REFUSED")


def test_scope_filter(store):
    got = store.search("token", V(0.0), top_k=5, hybrid_weight=0.5, scopes=("/auth.md",))
    assert got and all(g.source == "/auth.md" for g in got)


def test_query_scope_does_not_reach_a_sibling_directory(tmp_path):
    """`/data/proj` must not pull in `/data/project-secret` — scope is the only thing
    narrowing what a caller can see, so a bare string prefix is a disclosure."""
    s = Store(tmp_path / "db", dim=8)
    s.replace_source("/data/proj/f.md", [rec("/data/proj/f.md", 0, "shared body", V(0.0))])
    s.replace_source(
        "/data/project-secret/f.md",
        [rec("/data/project-secret/f.md", 0, "shared body", V(0.0))],
    )
    got = s.search("shared body", V(0.0), top_k=5, hybrid_weight=0.5, scopes=("/data/proj",))
    assert [g.source for g in got] == ["/data/proj/f.md"]


def test_max_distance_filter(store):
    all_rows = store.search("x", V(0.0), top_k=5, hybrid_weight=0.0)
    far = max(r.distance for r in all_rows)
    got = store.search("x", V(0.0), top_k=5, hybrid_weight=0.0, max_distance=far - 1e-6)
    assert len(got) < len(all_rows)


def test_max_files_limits_distinct_sources(store):
    got = store.search("x", V(0.05), top_k=5, hybrid_weight=0.0, max_files=1)
    assert len({g.source for g in got}) == 1


def test_top_k_truncates(store):
    got = store.search("x", V(0.0), top_k=1, hybrid_weight=0.0)
    assert len(got) == 1


def test_relevance_cutoff():
    assert relevance_cutoff([], "similar") == 0
    assert relevance_cutoff([0.1, 0.11], "similar") == 2  # <3 items: no cut
    d = [0.10, 0.11, 0.12, 0.55, 0.56]  # one big gap after index 2
    assert relevance_cutoff(d, "similar") == 3
    assert relevance_cutoff(d, "related") == 5  # only one boundary => keep all
    d2 = [0.10, 0.11, 0.40, 0.41, 0.80, 0.81]  # two big gaps
    assert relevance_cutoff(d2, "similar") == 2
    assert relevance_cutoff(d2, "related") == 4


def test_keyword_boost_disabled_keeps_vector_order(store):
    got = store.search("ERR_CONNECTION_REFUSED", V(1.0), top_k=3, hybrid_weight=0.0)
    assert got[0].source == "/cook.md"  # pure vector: the recipe wins


def test_malformed_fts_query_returns_no_rows_rather_than_raising(store):
    """Characterizes LanceDB: malformed FTS syntax yields 0 rows, it does not raise.

    Asserted against the table directly — through Store.search the except clause would
    mask a change here. The degradation path itself is covered by the injection test
    below, since no input we could find actually raises.
    """
    rows = store._table.search('"unbalanced quote AND (', query_type="fts").limit(5).to_list()
    assert rows == []
    got = store.search('"unbalanced quote AND (', V(0.0), top_k=3, hybrid_weight=0.6)
    assert got  # the vector side still answers


def test_relevance_cutoff_ignores_jitter():
    assert relevance_cutoff([0.10, 0.11, 0.20, 0.21, 0.30], "similar") == 5


def test_relevance_cutoff_no_cut_when_threshold_non_positive():
    assert relevance_cutoff([0.5, 0.5, 0.5, 0.5], "similar") == 4  # mean gap 0 => no boundary


# --- hybrid-mode filters: FTS-only hits (no distance) at the DEFAULT weight ----------
#
# `fetch = max(top_k * 4, 50)`, so with top_k=12 the vector side only ever fetches 50 rows.
# The corpus below puts 56 rows closer to the query than the three "keyed" rows, so the
# keyed rows can only ever arrive via FTS — they come back with `distance is None`, which
# is the exact case both distance filters have to handle. Keep `top_k` at 12 in these
# tests: a larger top_k widens the fetch window and the FTS-only rows disappear.


def FARAWAY(i):  # farther from V(0.0) than any V(x) can be, so it never enters the window
    return [-1.0 - i * 0.01, 0.0] + [0.0] * 6


@pytest.fixture
def hybrid_store(tmp_path):
    s = Store(tmp_path / "hybrid-db", dim=8)
    # Two tight clusters near the query; both match the keyword.
    s.replace_source(
        "/near.md",
        [rec("/near.md", i, "widget " * 100 + f"near {i}", V(i * 0.001)) for i in range(3)],
    )
    s.replace_source(
        "/mid.md",
        [rec("/mid.md", i, "widget " * 100 + f"mid {i}", V(0.30 + i * 0.001)) for i in range(3)],
    )
    # Filler: far but inside the fetch window, and deliberately keyword-free.
    s.replace_source(
        "/filler.md",
        [rec("/filler.md", i, f"sprocket filler body {i}", V(1.0 - i * 0.0005)) for i in range(50)],
    )
    # Keyed: strong keyword match, but too far to ever be fetched by the vector side.
    s.replace_source(
        "/keyed.md",
        [rec("/keyed.md", i, "widget " * 50 + f"keyed {i}", FARAWAY(i)) for i in range(3)],
    )
    return s


def test_hybrid_corpus_really_yields_fts_only_hits(hybrid_store):
    """Guard for the fixture: without distance-less hits the tests below prove nothing."""
    got = hybrid_store.search("widget", V(0.0), top_k=12, hybrid_weight=0.6)
    assert any(r.distance is None for r in got), "expected FTS-only hits with no distance"
    assert any(r.distance is not None and r.distance > 1.0 for r in got), "expected far hits"


def test_grouping_similar_cuts_far_results_at_default_hybrid_weight(hybrid_store):
    """Regression: grouping was skipped whenever any hit lacked a distance (the default)."""
    ungrouped = hybrid_store.search("widget", V(0.0), top_k=12, hybrid_weight=0.6)

    got = hybrid_store.search("widget", V(0.0), top_k=12, hybrid_weight=0.6, grouping="similar")
    kept = sorted(r.distance for r in got if r.distance is not None)
    assert len(got) < len(ungrouped), "the cut must actually drop results"
    assert len(kept) == 3, "only the near cluster survives the first boundary"
    assert max(kept) < 0.01
    assert all(r.source != "/mid.md" for r in got)  # second cluster cut away


def test_grouping_related_keeps_two_clusters_at_default_hybrid_weight(hybrid_store):
    ungrouped = hybrid_store.search("widget", V(0.0), top_k=12, hybrid_weight=0.6)

    got = hybrid_store.search("widget", V(0.0), top_k=12, hybrid_weight=0.6, grouping="related")
    kept = sorted(r.distance for r in got if r.distance is not None)
    assert len(got) < len(ungrouped), "the cut must actually drop results"
    assert len(kept) == 6, "near + mid clusters survive the second boundary"
    assert max(kept) < 0.2  # the far filler cluster (~1.98) is gone
    assert any(0.1 < d < 0.2 for d in kept)  # the mid cluster is still there


def test_grouping_keeps_fts_only_hits_and_preserves_rrf_order(hybrid_store):
    ungrouped = hybrid_store.search("widget", V(0.0), top_k=12, hybrid_weight=0.6)
    got = hybrid_store.search("widget", V(0.0), top_k=12, hybrid_weight=0.6, grouping="related")
    assert any(r.distance is None for r in got), "distance-less hits are never cut by grouping"
    survivors = [(r.source, r.chunk_index) for r in got]
    order = [(r.source, r.chunk_index) for r in ungrouped]
    assert survivors == [x for x in order if x in set(survivors)]


def test_max_distance_drops_hits_without_a_distance(hybrid_store):
    """Regression: FTS-only hits sailed past RAG_MAX_DISTANCE because distance was None."""
    unbounded = hybrid_store.search("widget", V(0.0), top_k=12, hybrid_weight=0.6)
    assert any(r.distance is None for r in unbounded)

    got = hybrid_store.search("widget", V(0.0), top_k=12, hybrid_weight=0.6, max_distance=0.5)
    assert got, "the near/mid clusters are within the bound"
    assert all(r.distance is not None and r.distance <= 0.5 for r in got)


# --- keyword search covers the title column too ---------------------------------------


TITLED_SRC = "/И-112_ЗПС_Хранение ТМЗ на складах.md"


@pytest.fixture
def titled_store(tmp_path):
    """A row whose distinguishing term lives ONLY in its title — the real-corpus case:
    the document code and subject are in the filename, never in the body text.

    Its vector is FARAWAY, so it can never enter the vector fetch window (see the note
    above): if search finds it at all, keyword ranking is what found it.
    """
    s = Store(tmp_path / "titled-db", dim=8)
    s.replace_source(
        TITLED_SRC,
        [
            ChunkRecord(
                id=f"{TITLED_SRC}#0",
                source=TITLED_SRC,
                source_type="file",
                title="И-112 ЗПС Хранение ТМЗ на складах",
                chunk_index=0,
                text="Общие положения. Ответственный за приёмку назначается приказом.",
                vector=FARAWAY(0),
                file_hash="h",
                mtime=1.0,
                ingested_at="2026-08-07T00:00:00+00:00",
            )
        ],
    )
    s.replace_source(
        "/filler.md",
        [rec("/filler.md", i, f"sprocket filler body {i}", V(1.0 - i * 0.0005)) for i in range(50)],
    )
    return s


def test_titled_row_is_unreachable_by_vector_search_alone(titled_store):
    """Guard for the fixture: without this the tests below would prove nothing."""
    got = titled_store.search("ТМЗ", V(1.0), top_k=12, hybrid_weight=0.0)
    assert all(r.source != TITLED_SRC for r in got)


def test_title_only_term_is_found_at_default_hybrid_weight(titled_store):
    """Regression: FTS covered only `text`, so filenames/titles were unsearchable."""
    got = titled_store.search("ТМЗ", V(1.0), top_k=12, hybrid_weight=0.6)
    assert [r.source for r in got].count(TITLED_SRC) == 1, "title-only term must be retrievable"


def test_text_only_term_still_found_when_title_is_indexed(titled_store):
    got = titled_store.search("приёмку", V(1.0), top_k=12, hybrid_weight=0.6)
    assert [r.source for r in got].count(TITLED_SRC) == 1


def test_the_rare_term_really_is_absent_from_the_text_column(titled_store):
    """Guard: prove the term is in the title only, not merely a substring of the body."""
    rows = titled_store._table.search("ТМЗ", query_type="fts", fts_columns=["text"]).to_list()
    assert rows == []


def test_fts_failure_degrades_to_vector_ranking(store, monkeypatch):
    """LanceDB tolerates malformed FTS syntax, so inject a real failure to cover the except."""
    real_search = store._table.search
    calls = {"fts": 0}

    def failing_search(*args, **kwargs):
        if kwargs.get("query_type") == "fts":
            calls["fts"] += 1
            raise RuntimeError("FTS index unavailable")
        return real_search(*args, **kwargs)

    monkeypatch.setattr(store._table, "search", failing_search)
    got = store.search("ERR_CONNECTION_REFUSED", V(1.0), top_k=3, hybrid_weight=0.6)
    assert calls["fts"] == 1, "the FTS path must actually have been taken"
    assert got and got[0].source == "/cook.md"  # degraded to pure vector order
