"""Resolve a facet for one document and encode the result for storage and filtering."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from minirag_mcp.labels.dates import DocDate
from minirag_mcp.labels.taxonomy import Facet
from minirag_mcp.scope import is_under


@dataclass(frozen=True)
class Label:
    value: str
    confidence: float
    source: str  # "path" | "default" | "model"
    second: str | None = None
    uncertain: bool = False


@dataclass(frozen=True)
class DocLabels:
    labels: Mapping[str, Label]
    date: DocDate
    taxonomy_hash: str

    def labels_json(self) -> str:
        if not self.labels:
            return ""
        return json.dumps(
            {
                facet: {
                    "value": lab.value,
                    "confidence": round(lab.confidence, 4),
                    "source": lab.source,
                    "second": lab.second,
                    "uncertain": lab.uncertain,
                }
                for facet, lab in self.labels.items()
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    def label_keys(self) -> str:
        """Delimited keys a SQL `contains('|facet=value|')` can match without substring hits.

        An uncertain label also carries `|facet~second|` so a filter on its runner-up
        still finds it.
        """
        entries = []
        for facet, lab in self.labels.items():
            entries.append(f"{facet}={lab.value}")
            if lab.uncertain and lab.second:
                entries.append(f"{facet}~{lab.second}")
        if not entries:
            return ""
        return "|" + "|".join(sorted(entries)) + "|"


def root_relative(source: str, roots: Sequence[Path]) -> str | None:
    """`source` relative to the longest root holding it, NFC; None when no root holds it.

    NFC because APFS stores names decomposed while rules are typed composed.
    """
    best = ""
    for root in roots:
        base = str(root).rstrip("/")
        if is_under(source, base) and source != base and len(base) > len(best):
            best = base
    if not best:
        return None
    return unicodedata.normalize("NFC", source[len(best) :].lstrip("/"))


def path_value(facet: Facet, rel: str) -> str | None:
    rel = unicodedata.normalize("NFC", rel)
    best_len = -1
    best: str | None = None
    for value in facet.values:
        for p in value.paths:
            prefix = unicodedata.normalize("NFC", p)
            if is_under(rel, prefix) and len(prefix) > best_len:
                best, best_len = value.id, len(prefix)
    return best


def _model_label(probs: Mapping[str, float], uncertain_below: float) -> Label | None:
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return None
    top, p_top = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else None
    return Label(top, p_top, "model", second=second, uncertain=p_top < uncertain_below)


def resolve_facet(
    facet: Facet,
    rel: str | None,
    ask: Callable[[], Mapping[str, float]] | None,
    uncertain_below: float,
) -> Label | None:
    if facet.within and (rel is None or not any(is_under(rel, w) for w in facet.within)):
        return None
    if rel is not None:
        value = path_value(facet, rel)
        if value is not None:
            if facet.override_at is not None and ask is not None:
                label = _model_label(ask(), uncertain_below)
                if (
                    label is not None
                    and label.value != value
                    and label.confidence >= facet.override_at
                ):
                    return label
            return Label(value, 1.0, "path")
    if facet.default is not None:
        return Label(facet.default, 1.0, "default") if rel is not None else None
    if ask is not None:
        return _model_label(ask(), uncertain_below)
    return None


def decode_labels(labels_json: str) -> dict[str, Label]:
    if not labels_json:
        return {}
    return {
        facet: Label(
            value=d["value"],
            confidence=d["confidence"],
            source=d["source"],
            second=d.get("second"),
            uncertain=bool(d.get("uncertain", False)),
        )
        for facet, d in json.loads(labels_json).items()
    }


def document_fields(labels_json: str, doc_date: str, doc_date_source: str) -> dict:
    out: dict = {}
    labels = decode_labels(labels_json)
    if labels:
        out["labels"] = {
            facet: {
                "value": lab.value,
                "confidence": lab.confidence,
                "source": lab.source,
                "uncertain": lab.uncertain,
            }
            for facet, lab in labels.items()
        }
    if doc_date:
        out["date"] = {"value": doc_date, "source": doc_date_source}
    return out
