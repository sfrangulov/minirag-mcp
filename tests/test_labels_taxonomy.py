import re
import unicodedata
from pathlib import Path

import pytest

from minirag_mcp.labels import taxonomy
from minirag_mcp.labels.taxonomy import TaxonomyError, load_taxonomy, taxonomy_path

MINIMAL = """
[model]
path = "/models/plumb.gguf"
temperature = 2.07

[facets.type]
values.minutes = { description = "Minutes. Date, participants, decisions." }
values.mail = { description = "Mail. Project correspondence." }
"""

REORDERED = """

[facets.type]

values.minutes = { description = "Minutes. Date, participants, decisions." }
values.mail = { description = "Mail. Project correspondence." }


[model]
temperature = 2.07
path = "/models/plumb.gguf"

"""

PATH_ONLY = """
[facets.project]
default = "alpha"
values.alpha = { description = "Alpha project." }
values.beta = { description = "Beta project.", paths = ["beta"] }
"""


def write(tmp_path: Path, text: str, encoding: str = "utf-8") -> Path:
    p = tmp_path / "classify.toml"
    p.write_text(text, encoding=encoding)
    return p


def fails(tmp_path: Path, text: str, pattern: str, encoding: str = "utf-8") -> None:
    """Assert load_taxonomy rejects `text` with a message that starts with `pattern`.

    The message is matched with the file path stripped: tmp_path embeds the test name.
    """
    path = write(tmp_path, text, encoding)
    with pytest.raises(TaxonomyError) as exc:
        load_taxonomy(path)
    message = str(exc.value)
    prefix = f"{path}: "
    assert message.startswith(prefix), message
    assert re.match(pattern, message[len(prefix) :]), message


def test_minimal_file_loads(tmp_path):
    t = load_taxonomy(write(tmp_path, MINIMAL))
    assert [f.name for f in t.facets] == ["type"]
    assert t.facet("type").value_ids == ("minutes", "mail")
    assert t.facet("type").question == "What kind of document is this?"
    assert t.model.temperature == 2.07 and t.model.uncertain_below == 0.7
    assert t.model.file_name_label == "File name"
    assert t.model.path == Path("/models/plumb.gguf")
    assert t.source == tmp_path / "classify.toml"


def test_unknown_facet_raises_key_error(tmp_path):
    t = load_taxonomy(write(tmp_path, MINIMAL))
    with pytest.raises(KeyError):
        t.facet("nope")


@pytest.mark.parametrize(
    ("text", "key"),
    [
        (
            MINIMAL.replace("[facets.type]", "[facets.type]\noveride_at = 0.9"),
            r"facets\.type\.overide_at",
        ),
        (MINIMAL.replace("temperature", "tempature = 1\ntemperature"), r"model\.tempature"),
        (MINIMAL + "\n[extra]\nx = 1\n", r"extra"),
        (
            MINIMAL.replace('{ description = "Mail.', '{ colour = "x", description = "Mail.'),
            r"facets\.type\.values\.mail\.colour",
        ),
    ],
)
def test_an_unknown_key_is_named(tmp_path, text, key):
    fails(tmp_path, text, rf"{key}: unknown key")


def test_default_must_be_one_of_the_values(tmp_path):
    text = PATH_ONLY.replace('default = "alpha"', 'default = "gamma"')
    fails(tmp_path, text, r"facets\.project\.default: 'gamma' is not one of")


@pytest.mark.parametrize("raw", ['["alpha"]', '{ id = "alpha" }', "1", "true"])
def test_default_must_be_a_string(tmp_path, raw):
    text = PATH_ONLY.replace('default = "alpha"', f"default = {raw}")
    fails(tmp_path, text, r"facets\.project\.default: must be a string naming one of the values")


def test_model_section_is_required_only_when_a_facet_needs_the_model(tmp_path):
    t = load_taxonomy(write(tmp_path, PATH_ONLY))
    assert t.model is None
    assert not t.facet("project").needs_model
    needs = PATH_ONLY.replace('default = "alpha"\n', "")
    fails(tmp_path, needs, r"\[model\] is required: facets project are answered")


def test_override_at_makes_a_defaulted_facet_need_the_model(tmp_path):
    text = PATH_ONLY.replace('default = "alpha"', 'default = "alpha"\noverride_at = 0.9')
    fails(tmp_path, text, r"\[model\] is required: facets project are answered")
    t = load_taxonomy(write(tmp_path, text + MINIMAL.split("[facets.type]")[0]))
    assert t.facet("project").needs_model


@pytest.mark.parametrize("bad", ["0", "1.5", "-0.1"])
def test_override_at_is_in_zero_exclusive_to_one(tmp_path, bad):
    text = PATH_ONLY.replace('default = "alpha"', f'default = "alpha"\noverride_at = {bad}')
    fails(
        tmp_path,
        text + MINIMAL.split("[facets.type]")[0],
        r"facets\.project\.override_at: must be in",
    )


def test_model_is_a_path_or_a_fully_pinned_repo(tmp_path):
    both = MINIMAL.replace('path = "/models/plumb.gguf"', 'path = "/m.gguf"\nrepo = "o/r"')
    fails(tmp_path, both, r"model: give either path or repo/file/revision/sha256, not both")
    no_sha = MINIMAL.replace(
        'path = "/models/plumb.gguf"', 'repo = "o/r"\nfile = "f.gguf"\nrevision = "abc"'
    )
    fails(tmp_path, no_sha, r"model\.sha256: a pinned download needs")
    neither = MINIMAL.replace('path = "/models/plumb.gguf"\n', "")
    fails(tmp_path, neither, r"model: give either path or repo/file/revision/sha256$")
    pinned = MINIMAL.replace(
        'path = "/models/plumb.gguf"',
        'repo = "o/r"\nfile = "f.gguf"\nrevision = "abc"\nsha256 = "def"',
    )
    m = load_taxonomy(write(tmp_path, pinned)).model
    assert (m.repo, m.file, m.revision, m.sha256, m.path) == ("o/r", "f.gguf", "abc", "def", None)


def test_model_path_is_expanded(tmp_path):
    text = MINIMAL.replace("/models/plumb.gguf", "~/plumb.gguf")
    assert load_taxonomy(write(tmp_path, text)).model.path == Path("~/plumb.gguf").expanduser()


def test_a_model_path_whose_home_folder_cannot_be_found_is_named(tmp_path):
    text = MINIMAL.replace("/models/plumb.gguf", "~nosuchuser_zz/plumb.gguf")
    fails(tmp_path, text, r"model\.path: cannot expand '~nosuchuser_zz/plumb\.gguf'")


def test_an_unforeseen_validation_failure_is_a_taxonomy_error_naming_the_file(
    tmp_path, monkeypatch
):
    """Any exception escaping validation would take down every tool that reads the file."""

    def gap(raw):
        raise ValueError("a case validation does not cover")

    monkeypatch.setattr(taxonomy, "_parse_model", gap)
    fails(tmp_path, MINIMAL, r"ValueError: a case validation does not cover$")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", "0"),
        ("temperature", "-1"),
        ("uncertain_below", "0"),
        ("uncertain_below", "1"),
    ],
)
def test_model_numbers_are_range_checked(tmp_path, field, value):
    text = MINIMAL.replace("temperature = 2.07", "temperature = 2.07\n" + f"{field} = {value}")
    if field == "temperature":
        text = MINIMAL.replace("2.07", value)
    fails(tmp_path, text, rf"model\.{field}: must be")


def _facet_with(n: int) -> str:
    rows = "\n".join(f'values.v{i} = {{ description = "Value {i}." }}' for i in range(n))
    return MINIMAL.split("[facets.type]")[0] + "[facets.type]\n" + rows + "\n"


def test_a_model_facet_takes_at_most_16_values(tmp_path):
    assert len(load_taxonomy(write(tmp_path, _facet_with(16))).facet("type").values) == 16
    fails(tmp_path, _facet_with(17), r"facets\.type\.values: .*at most 16 values")


def test_a_path_only_facet_may_exceed_16_values(tmp_path):
    rows = "\n".join(f'values.v{i} = {{ description = "V {i}." }}' for i in range(17))
    text = '[facets.p]\ndefault = "v0"\n' + rows + "\n"
    assert len(load_taxonomy(write(tmp_path, text)).facet("p").values) == 17


def test_a_facet_needs_at_least_one_value(tmp_path):
    fails(tmp_path, '[facets.p]\ndefault = "x"\n', r"facets\.p\.values: at least one value")


def test_descriptions_are_non_empty(tmp_path):
    fails(
        tmp_path,
        MINIMAL.replace("Mail. Project correspondence.", "  "),
        r"facets\.type\.values\.mail\.description: must be a non-empty",
    )
    missing = MINIMAL.replace('{ description = "Mail. Project correspondence." }', "{ paths = [] }")
    fails(tmp_path, missing, r"facets\.type\.values\.mail\.description: must be a non-empty")


def test_ids_are_lowercase_slugs(tmp_path):
    fails(
        tmp_path,
        MINIMAL.replace("values.mail", "values.Mail"),
        r"facets\.type\.values\.Mail: 'Mail' is not a lowercase slug",
    )
    fails(
        tmp_path,
        MINIMAL.replace("facets.type", "facets.Type"),
        r"facets\.Type: 'Type' is not a lowercase slug",
    )
    fails(
        tmp_path,
        MINIMAL.replace("values.mail", 'values."a=b"'),
        r"facets\.type\.values\.a=b: 'a=b' is not a lowercase slug",
    )
    ok = MINIMAL.replace("facets.type", "facets.client-data")
    assert load_taxonomy(write(tmp_path, ok)).facets[0].name == "client-data"


def test_paths_are_relative_and_normalised(tmp_path):
    base = PATH_ONLY.replace('paths = ["beta"]', 'paths = ["{p}"]')
    fails(
        tmp_path,
        base.replace("{p}", "/abs"),
        r"facets\.project\.values\.beta\.paths: .*non-empty relative path",
    )
    fails(
        tmp_path,
        base.replace("{p}", ""),
        r"facets\.project\.values\.beta\.paths: .*non-empty relative path",
    )
    t = load_taxonomy(write(tmp_path, base.replace("{p}", "a/b/")))
    assert t.facet("project").values[1].paths == ("a/b",)
    nfd = unicodedata.normalize("NFD", "й")
    assert nfd != "й"
    t = load_taxonomy(write(tmp_path, base.replace("{p}", f"dir/{nfd}")))
    assert t.facet("project").values[1].paths == ("dir/й",)


def test_within_is_validated_and_normalised(tmp_path):
    text = PATH_ONLY.replace('default = "alpha"', 'default = "alpha"\nwithin = ["x/y/"]')
    assert load_taxonomy(write(tmp_path, text)).facet("project").within == ("x/y",)
    bad = PATH_ONLY.replace('default = "alpha"', 'default = "alpha"\nwithin = ["/x"]')
    fails(tmp_path, bad, r"facets\.project\.within: '/x' must be a non-empty relative path")


def test_invalid_toml_is_a_taxonomy_error(tmp_path):
    fails(tmp_path, "[model\n", r"(Invalid|Unclosed|Expected)")


def test_a_file_that_is_not_utf8_is_a_taxonomy_error(tmp_path):
    """An editor on a Russian-locale Windows machine saves cp1251 by default."""
    text = MINIMAL.replace("Minutes. Date, participants, decisions.", "Протокол совещания.")
    fails(tmp_path, text, r"the file is not UTF-8", encoding="cp1251")


def test_hash_ignores_formatting_but_not_content(tmp_path):
    base = load_taxonomy(write(tmp_path, MINIMAL)).hash
    assert load_taxonomy(write(tmp_path, REORDERED)).hash == base
    changed = MINIMAL.replace("Mail. Project", "Mail. Client")
    assert load_taxonomy(write(tmp_path, changed)).hash != base
    assert load_taxonomy(write(tmp_path, MINIMAL.replace("2.07", "2.0"))).hash != base


def test_taxonomy_path_is_under_the_first_root(tmp_path):
    assert (
        taxonomy_path((tmp_path / "a", tmp_path / "b"))
        == tmp_path / "a" / ".minirag" / "classify.toml"
    )
