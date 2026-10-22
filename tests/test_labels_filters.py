from pathlib import Path

import pytest

from conftest import FakeModelFactory
from label_helpers import L, record
from minirag_mcp.labels import LabelerSource
from minirag_mcp.labels.dates import DocDate
from minirag_mcp.labels.filters import (
    LabelFilter,
    LabelFilterError,
    filter_counts,
    parse_filter,
    source_matches,
    to_sql,
)
from minirag_mcp.labels.resolve import DocLabels
from minirag_mcp.store import SourceInfo, Store

TAXONOMY = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.project]
default = "core"
values.core = { description = "The core project." }
values.pilot = { description = "The pilot project.", paths = ["pilot"] }

[facets.type]
values.spec = { description = "Spec. Requirements.", paths = ["specs"] }
values.minutes = { description = "Minutes. Decisions." }
values.other = { description = "Other." }

[facets.stage]
within = ["pilot"]
values.draft = { description = "A draft." }
values.final = { description = "A final version." }
"""


@pytest.fixture
def roots(tmp_path) -> tuple[Path, ...]:
    root = tmp_path / "docs"
    (root / ".minirag").mkdir(parents=True)
    (root / ".minirag" / "classify.toml").write_text(TAXONOMY)
    return (root,)


@pytest.fixture
def labeler(roots, tmp_path):
    source = LabelerSource(roots, tmp_path / "cache", FakeModelFactory())
    got, error = source.get()
    assert error is None and got is not None
    return got


def parse(labeler, labels=None, date_from=None, date_to=None):
    return parse_filter(
        labels,
        date_from,
        date_to,
        labeler=labeler,
        error=None,
        taxonomy_file=Path("/x/.minirag/classify.toml"),
    )


DOCS = {
    "/r/a.md": ({"type": L("minutes"), "project": L("core")}, "2026-10-14"),
    "/r/b.md": ({"type": L("spec"), "project": L("core")}, "2026-10-14"),
    "/r/c.md": ({"type": L("other", second="minutes", uncertain=True)}, "2026-10-20"),
    "/r/d.md": ({}, ""),
    "/r/e.md": ({"project": L("pilot")}, "2026-10-12"),
    "/r/f.md": ({"type": L("minutes")}, ""),
}


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "db", dim=8)
    for source, (labels, doc_date) in DOCS.items():
        store.replace_source(source, [record(source, labels, doc_date, "header")])
    return store


def test_sql_and_python_predicates_agree(labeler, store):
    infos = store.list_sources()
    assert len(infos) == 6
    filters = [
        {"labels": {"type": "minutes"}},
        {"labels": {"type": ["spec", "minutes"]}},
        {"labels": {"type": "minutes", "project": "core"}},
        {"labels": {"project": "pilot"}},
        {"date_from": "2026-10-13"},
        {"date_to": "2026-10-14"},
        {"labels": {"type": "minutes"}, "date_from": "2026-10-14", "date_to": "2026-10-20"},
    ]
    seen = set()
    for kwargs in filters:
        f = parse(labeler, **kwargs)
        via_sql = {i.source for i in store.list_sources(filter_sql=to_sql(f))}
        via_python = {i.source for i in infos if source_matches(f, i)}
        assert via_sql == via_python, kwargs
        seen.add(frozenset(via_sql))
    assert len(seen) == len(filters)  # the cases are not all the same answer


def test_an_uncertain_document_matches_its_second_choice(labeler, store):
    f = parse(labeler, {"type": "minutes"})
    sql = {i.source for i in store.list_sources(filter_sql=to_sql(f))}
    python = {i.source for i in store.list_sources() if source_matches(f, i)}
    assert "/r/c.md" in sql and "/r/c.md" in python
    strict = parse(labeler, {"type": "spec"})
    assert not [
        i for i in store.list_sources() if i.source == "/r/c.md" and source_matches(strict, i)
    ]


def test_filters_without_a_taxonomy_file_are_refused():
    with pytest.raises(LabelFilterError, match="document labels are off"):
        parse_filter(
            {"type": "spec"},
            None,
            None,
            labeler=None,
            error=None,
            taxonomy_file=Path("/x/.minirag/classify.toml"),
        )
    with pytest.raises(LabelFilterError, match="is invalid: facets: broken"):
        parse_filter(
            None,
            "2026-10-01",
            None,
            labeler=None,
            error="facets: broken",
            taxonomy_file=Path("/x/.minirag/classify.toml"),
        )


def test_nothing_asked_is_no_filter(labeler):
    assert parse(labeler) is None
    assert parse_filter(None, None, None, labeler=None, error=None, taxonomy_file=None) is None


def test_unknown_facet_and_value_list_the_known_ones(labeler):
    with pytest.raises(LabelFilterError, match="known facets: project, type, stage"):
        parse(labeler, {"kind": "spec"})
    with pytest.raises(LabelFilterError, match="known values: spec, minutes, other"):
        parse(labeler, {"type": "memo"})


@pytest.mark.parametrize(
    "date_from, date_to, match",
    [
        ("2026-1-2", None, "YYYY-MM-DD"),
        (None, "20261002", "YYYY-MM-DD"),
        ("2026-13-01", None, "calendar date"),
        ("2026-10-18", "2026-10-12", "after"),
    ],
)
def test_bad_and_reversed_dates_are_refused(labeler, date_from, date_to, match):
    with pytest.raises(LabelFilterError, match=match):
        parse(labeler, None, date_from, date_to)


def info(source, labels=None, doc_date=""):
    doc = DocLabels(labels or {}, DocDate(doc_date, "header" if doc_date else ""), "h")
    return SourceInfo(
        source=source,
        source_type="file",
        title="T",
        chunk_count=1,
        file_hash="h",
        mtime=1.0,
        labels_json=doc.labels_json(),
        doc_date=doc_date,
    )


def test_documents_outside_within_are_not_counted_as_unlabelled(labeler, roots):
    root = roots[0]
    infos = [
        info(f"{root}/pilot/a.md"),  # in reach of `stage`, unlabelled
        info(f"{root}/pilot/b.md", {"stage": L("draft")}),
        info(f"{root}/other/c.md"),  # outside `within`: excluded legitimately
        info(f"{root}/other/d.md"),
        info("https://example.test/page"),  # under no root at all
    ]
    f = parse(labeler, {"stage": "draft"})
    assert filter_counts(f, infos, labeler.taxonomy, roots) == {"unlabelled": 1, "undated": 0}


def test_a_document_missing_two_filtered_facets_counts_once(labeler, roots):
    infos = [info(f"{roots[0]}/pilot/a.md")]
    f = parse(labeler, {"stage": "draft", "type": "spec"})
    assert filter_counts(f, infos, labeler.taxonomy, roots)["unlabelled"] == 1


def test_undated_documents_are_counted_when_dates_are_filtered(labeler, roots):
    infos = [info("/r/a.md", doc_date="2026-10-14"), info("/r/b.md"), info("/r/c.md")]
    dated = parse(labeler, None, "2026-10-01")
    assert filter_counts(dated, infos, labeler.taxonomy, roots) == {"unlabelled": 0, "undated": 2}
    labels_only = parse(labeler, {"type": "spec"})
    assert filter_counts(labels_only, infos, labeler.taxonomy, roots)["undated"] == 0


def test_to_sql_has_the_documented_shape():
    f = LabelFilter((("type", ("minutes",)),), "2026-10-12", "2026-10-18")
    assert to_sql(f) == (
        "(contains(label_keys, '|type=minutes|') OR contains(label_keys, '|type~minutes|')) "
        "AND doc_date != '' AND doc_date >= '2026-10-12' AND doc_date <= '2026-10-18'"
    )
