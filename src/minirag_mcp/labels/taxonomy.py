import hashlib
import json
import re
import tomllib
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TAXONOMY_FILE = Path(".minirag") / "classify.toml"
DEFAULT_QUESTION = "What kind of document is this?"
DEFAULT_FILE_NAME_LABEL = "File name"
MAX_MODEL_VALUES = 16

# Ids end up inside stored keys that use `|`, `=`, `~` and `'` as separators.
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class TaxonomyError(Exception):
    pass


@dataclass(frozen=True)
class Value:
    id: str
    description: str
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Facet:
    name: str
    values: tuple[Value, ...]
    question: str = DEFAULT_QUESTION
    within: tuple[str, ...] = ()
    default: str | None = None
    override_at: float | None = None

    @property
    def needs_model(self) -> bool:
        return self.default is None or self.override_at is not None

    @property
    def value_ids(self) -> tuple[str, ...]:
        return tuple(v.id for v in self.values)


@dataclass(frozen=True)
class ModelSpec:
    temperature: float
    path: Path | None = None
    repo: str | None = None
    file: str | None = None
    revision: str | None = None
    sha256: str | None = None
    uncertain_below: float = 0.7
    file_name_label: str = DEFAULT_FILE_NAME_LABEL


@dataclass(frozen=True)
class Taxonomy:
    facets: tuple[Facet, ...]
    model: ModelSpec | None
    source: Path

    @property
    def hash(self) -> str:
        canonical = {
            "facets": [
                {
                    "name": f.name,
                    "question": f.question,
                    "within": list(f.within),
                    "default": f.default,
                    "override_at": f.override_at,
                    "values": [
                        {"id": v.id, "description": v.description, "paths": list(v.paths)}
                        for v in f.values
                    ],
                }
                for f in self.facets
            ],
            "model": None
            if self.model is None
            else {
                "path": None if self.model.path is None else str(self.model.path),
                "repo": self.model.repo,
                "file": self.model.file,
                "revision": self.model.revision,
                "sha256": self.model.sha256,
                "temperature": self.model.temperature,
                "uncertain_below": self.model.uncertain_below,
                "file_name_label": self.model.file_name_label,
            },
        }
        blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def facet(self, name: str) -> Facet:
        for f in self.facets:
            if f.name == name:
                return f
        raise KeyError(name)


def taxonomy_path(roots: Sequence[Path]) -> Path:
    return roots[0] / TAXONOMY_FILE


def _table(data: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(data, dict):
        raise TaxonomyError(f"{key}: expected a table")
    return data


def _reject_unknown(table: Mapping[str, Any], allowed: set[str], key: str) -> None:
    for k in table:
        if k not in allowed:
            raise TaxonomyError(f"{key}.{k}: unknown key" if key else f"{k}: unknown key")


def _check_id(value: str, key: str) -> None:
    if not _ID.match(value):
        raise TaxonomyError(f"{key}: {value!r} is not a lowercase slug (a-z, 0-9, _ and -)")


def _number(table: Mapping[str, Any], name: str, key: str) -> float:
    v = table[name]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TaxonomyError(f"{key}.{name}: expected a number")
    return float(v)


def _paths(raw: Any, key: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise TaxonomyError(f"{key}: expected a list of paths")
    out = []
    for p in raw:
        if not isinstance(p, str):
            raise TaxonomyError(f"{key}: entries must be strings")
        norm = unicodedata.normalize("NFC", p).rstrip("/")
        if not norm or p.startswith("/"):
            raise TaxonomyError(f"{key}: {p!r} must be a non-empty relative path")
        out.append(norm)
    return tuple(out)


def _parse_value(vid: str, raw: Any, key: str) -> Value:
    _check_id(vid, key)
    table = _table(raw, key)
    _reject_unknown(table, {"description", "paths"}, key)
    desc = table.get("description")
    if not isinstance(desc, str) or not desc.strip():
        raise TaxonomyError(f"{key}.description: must be a non-empty string")
    return Value(vid, desc, _paths(table.get("paths", []), f"{key}.paths"))


def _parse_facet(name: str, raw: Any) -> Facet:
    key = f"facets.{name}"
    _check_id(name, key)
    table = _table(raw, key)
    _reject_unknown(table, {"question", "within", "default", "override_at", "values"}, key)
    raw_values = _table(table.get("values", {}), f"{key}.values")
    values = tuple(_parse_value(vid, v, f"{key}.values.{vid}") for vid, v in raw_values.items())
    if not values:
        raise TaxonomyError(f"{key}.values: at least one value is required")

    question = table.get("question", DEFAULT_QUESTION)
    if not isinstance(question, str) or not question.strip():
        raise TaxonomyError(f"{key}.question: must be a non-empty string")

    default = table.get("default")
    if default is not None and not isinstance(default, str):
        raise TaxonomyError(f"{key}.default: must be a string naming one of the values")
    if default is not None and default not in {v.id for v in values}:
        raise TaxonomyError(f"{key}.default: {default!r} is not one of the facet's values")

    override_at = None
    if "override_at" in table:
        override_at = _number(table, "override_at", key)
        if not 0 < override_at <= 1:
            raise TaxonomyError(f"{key}.override_at: must be in (0, 1]")

    facet = Facet(
        name=name,
        values=values,
        question=question,
        within=_paths(table.get("within", []), f"{key}.within"),
        default=default,
        override_at=override_at,
    )
    if facet.needs_model and len(values) > MAX_MODEL_VALUES:
        raise TaxonomyError(
            f"{key}.values: a facet the model answers takes at most {MAX_MODEL_VALUES} values"
        )
    return facet


_PINNED = ("repo", "file", "revision", "sha256")


def _parse_model(raw: Any) -> ModelSpec:
    key = "model"
    table = _table(raw, key)
    _reject_unknown(
        table,
        {"path", *_PINNED, "temperature", "uncertain_below", "file_name_label"},
        key,
    )
    for name in ("path", *_PINNED, "file_name_label"):
        if name in table and (not isinstance(table[name], str) or not table[name].strip()):
            raise TaxonomyError(f"{key}.{name}: must be a non-empty string")

    has_path = "path" in table
    pinned = [n for n in _PINNED if n in table]
    if has_path and pinned:
        raise TaxonomyError("model: give either path or repo/file/revision/sha256, not both")
    if not has_path:
        if not pinned:
            raise TaxonomyError("model: give either path or repo/file/revision/sha256")
        missing = [n for n in _PINNED if n not in table]
        if missing:
            raise TaxonomyError(f"model.{missing[0]}: a pinned download needs {', '.join(_PINNED)}")

    if "temperature" not in table:
        raise TaxonomyError("model.temperature: required")
    temperature = _number(table, "temperature", key)
    if temperature <= 0:
        raise TaxonomyError("model.temperature: must be greater than 0")

    uncertain_below = 0.7
    if "uncertain_below" in table:
        uncertain_below = _number(table, "uncertain_below", key)
        if not 0 < uncertain_below < 1:
            raise TaxonomyError("model.uncertain_below: must be in (0, 1)")

    path = None
    if has_path:
        try:
            path = Path(table["path"]).expanduser()
        except RuntimeError as e:  # `~user` naming no user, or no home folder at all
            raise TaxonomyError(f"model.path: cannot expand {table['path']!r}: {e}") from e

    return ModelSpec(
        temperature=temperature,
        path=path,
        repo=table.get("repo"),
        file=table.get("file"),
        revision=table.get("revision"),
        sha256=table.get("sha256"),
        uncertain_below=uncertain_below,
        file_name_label=table.get("file_name_label", DEFAULT_FILE_NAME_LABEL),
    )


def load_taxonomy(path: Path) -> Taxonomy:
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise TaxonomyError(f"{path}: {e}") from e
    # Not a TOMLDecodeError: tomllib decodes the bytes before it parses them.
    except UnicodeDecodeError as e:
        raise TaxonomyError(f"{path}: the file is not UTF-8 ({e}); save it as UTF-8") from e
    try:
        _reject_unknown(data, {"model", "facets"}, "")
        raw_facets = _table(data.get("facets", {}), "facets")
        facets = tuple(_parse_facet(n, f) for n, f in raw_facets.items())
        if not facets:
            raise TaxonomyError("facets: at least one facet is required")
        model = _parse_model(data["model"]) if "model" in data else None
        if model is None and any(f.needs_model for f in facets):
            raise TaxonomyError(
                "[model] is required: facets "
                + ", ".join(f.name for f in facets if f.needs_model)
                + " are answered by the model"
            )
    except TaxonomyError as e:
        raise TaxonomyError(f"{path}: {e}") from e
    # Validation has missed a case before, and anything but a TaxonomyError escapes every
    # tool that reads this file, unfiltered queries and `status` included.
    except Exception as e:
        raise TaxonomyError(f"{path}: {type(e).__name__}: {e}") from e
    return Taxonomy(facets=facets, model=model, source=path)
