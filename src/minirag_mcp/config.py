"""Environment / CLI-flag configuration. Flags > env > defaults."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import platformdirs

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MAX_FILE_SIZE = 104_857_600  # 100 MB
DEFAULT_CHUNK_MIN_LENGTH = 50
DEFAULT_HYBRID_WEIGHT = 0.6
GROUPING_MODES = ("similar", "related")


class ConfigError(Exception):
    """Invalid configuration. The server stays up; only `status` keeps working."""


@dataclass(frozen=True)
class Config:
    roots: tuple[Path, ...]
    db_path: Path
    cache_dir: Path
    model_name: str
    max_file_size: int
    chunk_min_length: int
    hybrid_weight: float
    grouping: str | None
    max_distance: float | None
    max_files: int | None


def _resolve(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _roots(env: Mapping[str, str], base_dir_flags: Sequence[str], cwd: Path) -> tuple[Path, ...]:
    if base_dir_flags:
        return tuple(_resolve(p) for p in base_dir_flags)
    raw = env.get("BASE_DIRS")
    if raw is not None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConfigError(f"BASE_DIRS must be a JSON array of non-empty paths: {e}") from e
        if (
            not isinstance(parsed, list)
            or not parsed
            or not all(isinstance(p, str) and p.strip() for p in parsed)
        ):
            raise ConfigError(
                "BASE_DIRS must be a JSON array of one or more non-empty path strings"
            )
        return tuple(_resolve(p) for p in parsed)
    base = env.get("BASE_DIR", "").strip()
    if base:
        return (_resolve(base),)
    return (cwd.resolve(),)


def _int(env: Mapping[str, str], key: str, default: int, lo: int, hi: int | None = None) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError as e:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from e
    if val < lo or (hi is not None and val > hi):
        hi_str = hi if hi is not None else "∞"
        raise ConfigError(f"{key} must be in range [{lo}, {hi_str}], got {val}")
    return val


def _opt_float(env: Mapping[str, str], key: str, lo: float) -> float | None:
    raw = env.get(key)
    if raw is None:
        return None
    try:
        val = float(raw)
    except ValueError as e:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from e
    if val < lo:
        raise ConfigError(f"{key} must be >= {lo}, got {val}")
    return val


def load_config(
    env: Mapping[str, str],
    *,
    base_dir_flags: Sequence[str] = (),
    db_path_flag: str | None = None,
    cache_dir_flag: str | None = None,
    model_name_flag: str | None = None,
    cwd: Path | None = None,
) -> Config:
    cwd = (cwd or Path.cwd()).resolve()
    roots = _roots(env, base_dir_flags, cwd)

    db_raw = db_path_flag or env.get("DB_PATH")
    db_path = _resolve(db_raw) if db_raw else roots[0] / ".minirag" / "lancedb"

    cache_raw = cache_dir_flag or env.get("CACHE_DIR")
    cache_dir = (
        _resolve(cache_raw) if cache_raw else platformdirs.user_cache_path("minirag-mcp") / "models"
    )

    weight_raw = env.get("RAG_HYBRID_WEIGHT")
    if weight_raw is None:
        hybrid_weight = DEFAULT_HYBRID_WEIGHT
    else:
        try:
            hybrid_weight = float(weight_raw)
        except ValueError as e:
            raise ConfigError(f"RAG_HYBRID_WEIGHT must be a number, got {weight_raw!r}") from e
        if not 0.0 <= hybrid_weight <= 1.0:
            raise ConfigError(f"RAG_HYBRID_WEIGHT must be in [0, 1], got {hybrid_weight}")

    grouping = env.get("RAG_GROUPING")
    if grouping is not None and grouping not in GROUPING_MODES:
        raise ConfigError(f"RAG_GROUPING must be one of {GROUPING_MODES}, got {grouping!r}")

    max_files_raw = env.get("RAG_MAX_FILES")
    max_files = _int(env, "RAG_MAX_FILES", 1, 1) if max_files_raw is not None else None

    return Config(
        roots=roots,
        db_path=db_path,
        cache_dir=cache_dir,
        model_name=model_name_flag or env.get("MODEL_NAME") or DEFAULT_MODEL,
        max_file_size=_int(env, "MAX_FILE_SIZE", DEFAULT_MAX_FILE_SIZE, 1),
        chunk_min_length=_int(env, "CHUNK_MIN_LENGTH", DEFAULT_CHUNK_MIN_LENGTH, 1, 10_000),
        hybrid_weight=hybrid_weight,
        grouping=grouping,
        max_distance=_opt_float(env, "RAG_MAX_DISTANCE", 0.0),
        max_files=max_files,
    )
