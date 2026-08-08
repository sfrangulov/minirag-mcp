import pytest

from minirag_mcp.scope import is_under, sql_clause


@pytest.mark.parametrize(
    "path,prefix",
    [
        ("/data/proj", "/data/proj"),  # the scope path itself
        ("/data/proj/f.md", "/data/proj"),
        ("/data/proj/f.md", "/data/proj/"),  # trailing separator normalised away
        ("/data/proj/a/b.md", "/data/proj"),
        ("/anything", "/"),  # root scopes everything absolute
        ("https://example.com/docs/page", "https://example.com/docs"),
        ("note-1/part", "note-1"),  # data ids need not be paths
    ],
)
def test_under(path, prefix):
    assert is_under(path, prefix)


@pytest.mark.parametrize(
    "path,prefix",
    [
        ("/data/project-secret/f.md", "/data/proj"),
        ("/data/projX", "/data/proj"),
        ("/data/pro", "/data/proj"),
        ("/other/proj/f.md", "/data/proj"),
        ("https://example.com/docs-private/page", "https://example.com/docs"),
        ("note-10", "note-1"),
    ],
)
def test_not_under(path, prefix):
    assert not is_under(path, prefix)


def test_no_prefixes_means_no_clause():
    assert sql_clause(()) is None


def test_clause_matches_the_prefix_itself_and_its_subtree():
    assert sql_clause(("/data/proj/",)) == (
        "(source = '/data/proj' OR starts_with(source, '/data/proj/'))"
    )


def test_clause_ors_every_prefix():
    clause = sql_clause(("/a", "/b"))
    assert clause.count(" OR starts_with") == 2  # one per prefix
    assert "'/a'" in clause and "'/b'" in clause


def test_clause_escapes_single_quotes():
    """A source id may contain a quote; it must never terminate the SQL literal."""
    clause = sql_clause(("/o'brien",))
    assert "'/o''brien'" in clause
    assert clause.count("'") % 2 == 0
