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
    got = store.search("token", V(0.0), top_k=5, hybrid_weight=0.5, scopes=("/auth",))
    assert got and all(g.source == "/auth.md" for g in got)


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


def test_malformed_fts_query_degrades_to_vector(store):
    got = store.search('"unbalanced quote AND (', V(0.0), top_k=3, hybrid_weight=0.6)
    assert got  # no exception, vector side still answers


def test_relevance_cutoff_ignores_jitter():
    assert relevance_cutoff([0.10, 0.11, 0.20, 0.21, 0.30], "similar") == 5
