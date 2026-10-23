"""Label and date filters: one predicate written twice, as SQL and as Python.

`to_sql` narrows a search inside the store; `source_matches` is the same rule over a
`SourceInfo` for listings, which already hold every document. The two must agree, and
a test runs the same documents through both.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from minirag_mcp.labels.labeler import Labeler
from minirag_mcp.labels.resolve import decode_labels, root_relative
from minirag_mcp.labels.taxonomy import Facet, Taxonomy
from minirag_mcp.scope import is_under
from minirag_mcp.store import SourceInfo

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LabelFilterError(ValueError):
    """A filter that cannot run as given."""


@dataclass(frozen=True)
class LabelFilter:
    labels: tuple[tuple[str, tuple[str, ...]], ...]
    date_from: str | None = None
    date_to: str | None = None

    @property
    def has_dates(self) -> bool:
        return self.date_from is not None or self.date_to is not None


def _check_date(name: str, value: str | None) -> None:
    if value is None:
        return
    if not _DATE.match(value):
        raise LabelFilterError(f"{name} must be YYYY-MM-DD, got {value!r}")
    try:
        date.fromisoformat(value)
    except ValueError as e:
        raise LabelFilterError(f"{name} is not a calendar date: {value!r}") from e


def parse_filter(
    labels: Mapping[str, str | Sequence[str]] | None,
    date_from: str | None,
    date_to: str | None,
    *,
    labeler: Labeler | None,
    error: str | None,
    taxonomy_file: Path | None,
) -> LabelFilter | None:
    if not labels and date_from is None and date_to is None:
        return None
    if error is not None:
        raise LabelFilterError(f"{taxonomy_file} is invalid: {error}")
    if labeler is None:
        raise LabelFilterError(f"document labels are off: there is no {taxonomy_file}")
    taxonomy = labeler.taxonomy
    _check_date("dateFrom", date_from)
    _check_date("dateTo", date_to)
    if date_from is not None and date_to is not None and date_from > date_to:
        raise LabelFilterError(f"dateFrom {date_from} is after dateTo {date_to}")

    wanted: dict[str, tuple[str, ...]] = {}
    for name, raw in (labels or {}).items():
        try:
            facet = taxonomy.facet(name)
        except KeyError:
            known = ", ".join(f.name for f in taxonomy.facets)
            raise LabelFilterError(f"unknown label facet {name!r}; known facets: {known}") from None
        values = (raw,) if isinstance(raw, str) else tuple(raw)
        if not values:
            raise LabelFilterError(f"label facet {name!r} needs at least one value")
        for v in values:
            if v not in facet.value_ids:
                raise LabelFilterError(
                    f"unknown value {v!r} for label facet {name!r}; "
                    f"known values: {', '.join(facet.value_ids)}"
                )
        wanted[name] = tuple(dict.fromkeys(values))
    ordered = tuple((f.name, wanted[f.name]) for f in taxonomy.facets if f.name in wanted)
    return LabelFilter(ordered, date_from, date_to)


def to_sql(f: LabelFilter) -> str:
    """Values OR within a facet, facets AND. Ids are validated slugs, so they need no escaping.

    `~` keys are an uncertain document's runner-up: a filter on it still finds the document.
    """
    clauses = []
    for facet, values in f.labels:
        terms = []
        for v in values:
            terms.append(f"contains(label_keys, '|{facet}={v}|')")
            terms.append(f"contains(label_keys, '|{facet}~{v}|')")
        clauses.append("(" + " OR ".join(terms) + ")")
    if f.has_dates:
        parts = ["doc_date != ''"]
        if f.date_from is not None:
            parts.append(f"doc_date >= '{f.date_from}'")
        if f.date_to is not None:
            parts.append(f"doc_date <= '{f.date_to}'")
        clauses.append(" AND ".join(parts))
    return " AND ".join(clauses)


def source_matches(f: LabelFilter, info: SourceInfo) -> bool:
    labels = decode_labels(info.labels_json)
    for facet, values in f.labels:
        lab = labels.get(facet)
        if lab is None:
            return False
        if lab.value not in values and not (lab.uncertain and lab.second in values):
            return False
    if f.has_dates:
        if not info.doc_date:
            return False
        if f.date_from is not None and info.doc_date < f.date_from:
            return False
        if f.date_to is not None and info.doc_date > f.date_to:
            return False
    return True


def facet_applies(facet: Facet, rel: str | None) -> bool:
    """Whether `rel` (root-relative; None under no root) is inside the facet's `within`."""
    if not facet.within:
        return True
    return rel is not None and any(is_under(rel, w) for w in facet.within)


def filter_counts(
    f: LabelFilter,
    infos: Iterable[SourceInfo],
    taxonomy: Taxonomy,
    roots: Sequence[Path],
) -> dict[str, int]:
    """Documents in scope that a filter excludes for want of a label or a date.

    A document outside a facet's `within` is excluded legitimately, not for lack of a
    label, and counts in neither figure: counting it would make the number scale with the
    corpus rather than with the labelling gap.
    """
    unlabelled = undated = 0
    for info in infos:
        rel = root_relative(info.source, roots)
        labels = decode_labels(info.labels_json)
        if any(
            facet_applies(taxonomy.facet(name), rel) and name not in labels for name, _ in f.labels
        ):
            unlabelled += 1
        if f.has_dates and not info.doc_date:
            undated += 1
    return {"unlabelled": unlabelled, "undated": undated}


def filter_notice(f: LabelFilter, labeler: Labeler) -> str | None:
    names = [name for name, _ in f.labels if labeler.taxonomy.facet(name).needs_model]
    if not names:
        return None
    problem = labeler.model_problem()
    if problem is None:
        return None
    return f"labels for {', '.join(names)} come from the model, which cannot run here: {problem}"
