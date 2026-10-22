import json
import unicodedata
from pathlib import Path

import pytest

from minirag_mcp.labels.dates import DocDate
from minirag_mcp.labels.resolve import (
    DocLabels,
    Label,
    decode_labels,
    document_fields,
    path_value,
    resolve_facet,
    root_relative,
)
from minirag_mcp.labels.taxonomy import Facet, Value


def _never() -> dict:
    pytest.fail("ask must not be called")


def _facet(**kw) -> Facet:
    values = kw.pop("values", (Value("spec", "a spec"), Value("instruction", "how to")))
    return Facet(name="t", values=values, **kw)


def test_root_relative_picks_the_longest_root():
    roots = [Path("/d"), Path("/d/inner")]
    assert root_relative("/d/inner/x.pdf", roots) == "x.pdf"
    assert root_relative("/d/other/x.pdf", roots) == "other/x.pdf"


def test_root_relative_outside_every_root_is_none():
    assert root_relative("/elsewhere/x.pdf", [Path("/d")]) is None
    assert root_relative("/data_archive/x.pdf", [Path("/data")]) is None
    assert root_relative("https://example.com/x", [Path("/d")]) is None


def test_longest_path_rule_wins_at_segment_boundaries():
    f = _facet(
        values=(Value("initiatives", "i", ("ai",)), Value("fx", "f", ("ai/fx",))),
    )
    assert path_value(f, "ai/fx/a.pdf") == "fx"
    assert path_value(f, "ai/other/a.pdf") == "initiatives"
    assert path_value(f, "ai-old/a.pdf") is None


def test_nfd_path_matches_an_nfc_rule():
    word = "Новый"
    source = unicodedata.normalize("NFD", f"/d/ai/{word} проект/a.pdf")
    assert source != unicodedata.normalize("NFC", source)
    f = _facet(values=(Value("p", "p", (unicodedata.normalize("NFC", f"ai/{word} проект"),)),))
    rel = root_relative(source, [Path("/d")])
    assert rel == unicodedata.normalize("NFC", f"ai/{word} проект/a.pdf")
    assert path_value(f, rel) == "p"


def test_within_excludes_without_asking():
    f = _facet(within=("legal",))
    assert resolve_facet(f, "other/x.pdf", _never, 0.7) is None
    assert resolve_facet(f, None, _never, 0.7) is None


def test_path_rule_resolves_without_asking_when_not_overridable():
    f = _facet(values=(Value("spec", "s", ("specs",)), Value("instruction", "i")))
    assert resolve_facet(f, "specs/a.pdf", _never, 0.7) == Label("spec", 1.0, "path")


def test_a_path_rule_is_not_overridden_below_override_at():
    f = _facet(
        values=(Value("spec", "s"), Value("instruction", "i", ("how",))),
        override_at=0.9,
    )
    lab = resolve_facet(f, "how/a.pdf", lambda: {"spec": 0.85, "instruction": 0.15}, 0.7)
    assert lab == Label("instruction", 1.0, "path")


def test_a_confident_model_overrides_a_path_rule():
    f = _facet(
        values=(Value("spec", "s"), Value("instruction", "i", ("how",))),
        override_at=0.9,
    )
    lab = resolve_facet(f, "how/a.pdf", lambda: {"spec": 0.95, "instruction": 0.05}, 0.7)
    assert lab == Label("spec", 0.95, "model", second="instruction", uncertain=False)


def test_a_model_agreeing_with_the_path_keeps_the_path_label():
    f = _facet(values=(Value("spec", "s", ("how",)), Value("instruction", "i")), override_at=0.9)
    lab = resolve_facet(f, "how/a.pdf", lambda: {"spec": 0.99, "instruction": 0.01}, 0.7)
    assert lab == Label("spec", 1.0, "path")


def test_default_applies_to_files_but_not_to_text_documents():
    f = _facet(default="spec")
    assert resolve_facet(f, "x.pdf", _never, 0.7) == Label("spec", 1.0, "default")
    assert resolve_facet(f, None, _never, 0.7) is None


def test_no_rule_and_no_model_gives_nothing():
    assert resolve_facet(_facet(), "x.pdf", None, 0.7) is None


def test_uncertain_answer_keeps_its_second_choice():
    f = _facet(values=(Value("a", "a"), Value("b", "b"), Value("c", "c")))
    lab = resolve_facet(f, "x", lambda: {"a": 0.5, "b": 0.3, "c": 0.2}, 0.7)
    assert lab == Label("a", 0.5, "model", second="b", uncertain=True)
    assert DocLabels({"t": lab}, DocDate("", "none"), "h").label_keys() == "|t=a|t~b|"


def test_certain_answer_has_no_soft_key():
    f = _facet(values=(Value("a", "a"), Value("b", "b")))
    lab = resolve_facet(f, "x", lambda: {"a": 0.9, "b": 0.1}, 0.7)
    assert lab is not None and not lab.uncertain
    assert DocLabels({"t": lab}, DocDate("", "none"), "h").label_keys() == "|t=a|"


def test_ask_errors_propagate():
    def boom() -> dict:
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError):
        resolve_facet(_facet(), "x", boom, 0.7)


def test_label_keys_sorted_across_facets_and_empty_when_unlabelled():
    labels = {"z": Label("v", 1.0, "path"), "a": Label("w", 1.0, "default")}
    assert DocLabels(labels, DocDate("", "none"), "h").label_keys() == "|a=w|z=v|"
    assert DocLabels({}, DocDate("", "none"), "h").label_keys() == ""
    assert DocLabels({}, DocDate("", "none"), "h").labels_json() == ""


def test_labels_json_round_trips():
    labels = {
        "t": Label("a", 0.51234567, "model", second="b", uncertain=True),
        "k": Label("спека", 1.0, "path"),
    }
    raw = DocLabels(labels, DocDate("", "none"), "h").labels_json()
    assert "спека" in raw
    assert json.loads(raw)["t"]["confidence"] == 0.5123
    back = decode_labels(raw)
    assert back["k"] == labels["k"]
    assert back["t"] == Label("a", 0.5123, "model", second="b", uncertain=True)
    assert decode_labels("") == {}


def test_document_fields_omit_what_is_empty():
    assert document_fields("", "", "") == {}
    raw = DocLabels({"t": Label("a", 0.5, "model", "b", True)}, DocDate("", "none"), "h")
    assert document_fields(raw.labels_json(), "", "") == {
        "labels": {"t": {"value": "a", "confidence": 0.5, "source": "model", "uncertain": True}}
    }
    assert document_fields("", "2024-05-01", "header") == {
        "date": {"value": "2024-05-01", "source": "header"}
    }
