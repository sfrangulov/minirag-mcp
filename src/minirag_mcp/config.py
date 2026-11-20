"""Environment / CLI-flag configuration. Flags > env > defaults."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import platformdirs

from minirag_mcp.embedder import E5_LARGE, MODEL_WINDOWS, default_budget, model_window
from minirag_mcp.endpoints import Endpoint, EndpointError, read_api_key
from minirag_mcp.ocr import DEFAULT_LANGUAGES, SECOND_READER_BELOW, tesseract_languages

DEFAULT_MODEL = E5_LARGE
DEFAULT_DOCUMENT_WEIGHT = 0.5
DEFAULT_RERANK_TOP = 30
MAX_RERANK_TOP = 200
DEFAULT_MAX_FILE_SIZE = 104_857_600  # 100 MB
DEFAULT_HYBRID_WEIGHT = 0.6
GROUPING_MODES = ("similar", "related")
DEFAULT_FTS_LANGUAGE = "English"
# The stemmers LanceDB's FTS accepts (verified against lancedb 0.36). It refuses any
# other spelling, case included, so a setting is matched against these and replaced.
FTS_LANGUAGES = (
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hungarian",
    "Italian",
    "Norwegian",
    "Portuguese",
    "Romanian",
    "Russian",
    "Spanish",
    "Swedish",
    "Tamil",
    "Turkish",
)
# A mean word confidence in percent, as RAG_OCR_SECOND_READER_BELOW takes it.
DEFAULT_SECOND_READER_BELOW = round(SECOND_READER_BELOW * 100)
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})  # an empty value reads as "not set"


class ConfigError(Exception):
    """Invalid configuration. The server stays up; only `status` keeps working."""


@dataclass(frozen=True)
class Config:
    roots: tuple[Path, ...]
    db_path: Path
    cache_dir: Path
    model_name: str
    max_file_size: int
    token_budget: int
    hybrid_weight: float
    grouping: str | None
    max_distance: float | None
    max_files: int | None
    allow_private_urls: bool
    ocr_lang: str = DEFAULT_LANGUAGES
    ocr_min_chars_per_page: int = 25
    ocr_second_reader_below: int = DEFAULT_SECOND_READER_BELOW  # 0 turns the reader off
    fts_language: str = DEFAULT_FTS_LANGUAGE
    document_weight: float = DEFAULT_DOCUMENT_WEIGHT
    rerank_top: int = DEFAULT_RERANK_TOP
    embed_endpoints: Mapping[str, Endpoint] = field(
        default_factory=dict
    )  # fastembed id -> endpoint
    rerank_endpoint: Endpoint | None = None
    classify_endpoint: Endpoint | None = None
    index_name: str | None = None
    sync_workers: int | None = None
    max_table_rows: int | None = None


def default_db_path(roots: Sequence[Path]) -> Path:
    return roots[0] / ".minirag" / "v2"


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


def _bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUE:
        return True
    if val in _FALSE:
        return False
    raise ConfigError(f"{key} must be one of {sorted(_TRUE | (_FALSE - {''}))}, got {raw!r}")


def _float(env: Mapping[str, str], key: str, default: float, lo: float, hi: float) -> float:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError as e:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from e
    if not lo <= val <= hi:
        raise ConfigError(f"{key} must be in [{lo:g}, {hi:g}], got {val}")
    return val


def _endpoint(
    env: Mapping[str, str], url_key: str, name_key: str, api_key: str | None
) -> Endpoint | None:
    url = env.get(url_key, "").strip()
    name = env.get(name_key, "").strip()
    if not url and not name:
        return None
    if not url or not name:
        raise ConfigError(f"{url_key} and {name_key} must be set together")
    if not url.startswith(("http://", "https://")):
        raise ConfigError(f"{url_key} must start with http:// or https://, got {url!r}")
    return Endpoint(url, name, api_key)


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
    db_path = _resolve(db_raw) if db_raw else default_db_path(roots)

    cache_raw = cache_dir_flag or env.get("CACHE_DIR")
    cache_dir = (
        _resolve(cache_raw) if cache_raw else platformdirs.user_cache_path("minirag-mcp") / "models"
    )

    grouping = env.get("RAG_GROUPING")
    if grouping is not None and grouping not in GROUPING_MODES:
        raise ConfigError(f"RAG_GROUPING must be one of {GROUPING_MODES}, got {grouping!r}")

    max_files_raw = env.get("RAG_MAX_FILES")
    max_files = _int(env, "RAG_MAX_FILES", 1, 1) if max_files_raw is not None else None

    try:
        ocr_lang = tesseract_languages(env.get("RAG_OCR_LANG", DEFAULT_LANGUAGES))
    except ValueError as e:
        raise ConfigError(f"RAG_OCR_LANG: {e}") from e

    fts_raw = env.get("RAG_FTS_LANGUAGE", DEFAULT_FTS_LANGUAGE).strip()
    fts_language = next((lang for lang in FTS_LANGUAGES if lang.lower() == fts_raw.lower()), None)
    if fts_language is None:
        raise ConfigError(
            f"RAG_FTS_LANGUAGE must be one of {', '.join(FTS_LANGUAGES)}, got {fts_raw!r}"
        )

    sync_workers_raw = env.get("RAG_SYNC_WORKERS")
    sync_workers = (
        _int(env, "RAG_SYNC_WORKERS", 1, 1, 256) if sync_workers_raw is not None else None
    )

    max_table_rows_raw = env.get("RAG_MAX_TABLE_ROWS")
    max_table_rows = (
        _int(env, "RAG_MAX_TABLE_ROWS", 1, 1) if max_table_rows_raw is not None else None
    )

    model_name = model_name_flag or env.get("MODEL_NAME") or DEFAULT_MODEL

    key_raw = env.get("ENDPOINT_API_KEY_FILE", "").strip()
    try:
        api_key = read_api_key(Path(key_raw).expanduser()) if key_raw else None
    except EndpointError as e:
        raise ConfigError(f"ENDPOINT_API_KEY_FILE: {e}") from e
    embed_endpoint = _endpoint(env, "EMBED_URL", "EMBED_ENDPOINT_MODEL", api_key)
    if embed_endpoint is not None and model_name not in MODEL_WINDOWS:
        raise ConfigError(
            f"EMBED_URL is set for {model_name!r}, whose window is unknown. Text for the "
            "endpoint is cut where fastembed truncates it in-process, and for this model that "
            f"length is not known. Use one of {', '.join(MODEL_WINDOWS)}, or unset EMBED_URL "
            "to embed in-process."
        )

    return Config(
        roots=roots,
        db_path=db_path,
        cache_dir=cache_dir,
        model_name=model_name,
        max_file_size=_int(env, "MAX_FILE_SIZE", DEFAULT_MAX_FILE_SIZE, 1),
        # Capped at the model's window: past it the encoder does not
        # see the text at all, so a larger budget would only build chunks whose tails
        # are silently discarded. The bound is inclusive, and that is only sound
        # because `Embedder.count_tokens` counts with truncation disabled — a counter
        # that saturated at the ceiling would make `count <= window` true of every text
        # and turn the documented maximum into "no budget at all".
        token_budget=_int(
            env, "CHUNK_TOKEN_BUDGET", default_budget(model_name), 16, model_window(model_name)
        ),
        hybrid_weight=_float(env, "RAG_HYBRID_WEIGHT", DEFAULT_HYBRID_WEIGHT, 0.0, 1.0),
        grouping=grouping,
        max_distance=_opt_float(env, "RAG_MAX_DISTANCE", 0.0),
        max_files=max_files,
        allow_private_urls=_bool(env, "ALLOW_PRIVATE_URLS"),
        ocr_lang=ocr_lang,
        ocr_min_chars_per_page=_int(env, "RAG_OCR_MIN_CHARS_PER_PAGE", 25, 0),
        ocr_second_reader_below=_int(
            env, "RAG_OCR_SECOND_READER_BELOW", DEFAULT_SECOND_READER_BELOW, 0, 100
        ),
        fts_language=fts_language,
        document_weight=_float(env, "RAG_DOCUMENT_WEIGHT", DEFAULT_DOCUMENT_WEIGHT, 0.0, 1.0),
        rerank_top=_int(env, "RAG_RERANK_TOP", DEFAULT_RERANK_TOP, 1, MAX_RERANK_TOP),
        embed_endpoints={model_name: embed_endpoint} if embed_endpoint else {},
        rerank_endpoint=_endpoint(env, "RERANK_URL", "RERANK_MODEL", api_key),
        classify_endpoint=_endpoint(env, "CLASSIFY_URL", "CLASSIFY_MODEL", api_key),
        sync_workers=sync_workers,
        max_table_rows=max_table_rows,
    )
