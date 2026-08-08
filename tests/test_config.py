import dataclasses
from pathlib import Path

import pytest

from minirag_mcp.config import DEFAULT_MODEL, ConfigError, load_config


def test_defaults_from_cwd(tmp_path):
    cfg = load_config({}, cwd=tmp_path)
    assert cfg.roots == (tmp_path.resolve(),)
    assert cfg.db_path == tmp_path.resolve() / ".minirag" / "lancedb"
    assert cfg.model_name == DEFAULT_MODEL
    assert cfg.max_file_size == 104857600
    assert cfg.chunk_min_length == 50
    assert cfg.hybrid_weight == 0.6
    assert cfg.grouping is None and cfg.max_distance is None and cfg.max_files is None
    assert cfg.allow_private_urls is False  # the SSRF guard is on unless turned off
    assert "minirag-mcp" in str(cfg.cache_dir)  # platformdirs cache, not cwd-relative


def test_base_dir_env(tmp_path):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=Path("/"))
    assert cfg.roots == (tmp_path.resolve(),)
    assert cfg.db_path == tmp_path.resolve() / ".minirag" / "lancedb"


def test_base_dirs_json_overrides_base_dir(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    env = {"BASE_DIR": "/ignored", "BASE_DIRS": f'["{a}", "{b}"]'}
    cfg = load_config(env, cwd=tmp_path)
    assert cfg.roots == (a.resolve(), b.resolve())
    assert cfg.db_path == a.resolve() / ".minirag" / "lancedb"  # first root hosts the index


@pytest.mark.parametrize("bad", ["/a:/b", "[]", '["ok", ""]', "not json", '"str"'])
def test_invalid_base_dirs_is_hard_error(bad, tmp_path):
    with pytest.raises(ConfigError):
        load_config({"BASE_DIRS": bad}, cwd=tmp_path)


def test_flags_beat_env(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    cfg = load_config(
        {"BASE_DIRS": f'["{a}"]', "DB_PATH": "/env/db", "MODEL_NAME": "env-model"},
        base_dir_flags=[str(b)],
        db_path_flag=str(tmp_path / "flagdb"),
        model_name_flag="flag-model",
        cwd=tmp_path,
    )
    assert cfg.roots == (b.resolve(),)
    assert cfg.db_path == (tmp_path / "flagdb").resolve()
    assert cfg.model_name == "flag-model"


@pytest.mark.parametrize(
    "env",
    [
        {"MAX_FILE_SIZE": "abc"},
        {"MAX_FILE_SIZE": "0"},
        {"CHUNK_MIN_LENGTH": "0"},
        {"CHUNK_MIN_LENGTH": "10001"},
        {"RAG_HYBRID_WEIGHT": "1.5"},
        {"RAG_HYBRID_WEIGHT": "-0.1"},
        {"RAG_GROUPING": "bogus"},
        {"RAG_MAX_DISTANCE": "-1"},
        {"RAG_MAX_FILES": "0"},
        {"ALLOW_PRIVATE_URLS": "maybe"},
        {"ALLOW_PRIVATE_URLS": "2"},
    ],
)
def test_invalid_numeric_env(env, tmp_path):
    with pytest.raises(ConfigError):
        load_config(env, cwd=tmp_path)


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True)]
    + [("0", False), ("false", False), ("no", False), ("off", False), ("", False)],
)
def test_allow_private_urls_env(raw, expected, tmp_path):
    assert load_config({"ALLOW_PRIVATE_URLS": raw}, cwd=tmp_path).allow_private_urls is expected


def test_search_tuning_env(tmp_path):
    env = {
        "RAG_HYBRID_WEIGHT": "0.8",
        "RAG_GROUPING": "related",
        "RAG_MAX_DISTANCE": "0.5",
        "RAG_MAX_FILES": "2",
    }
    cfg = load_config(env, cwd=tmp_path)
    assert (cfg.hybrid_weight, cfg.grouping, cfg.max_distance, cfg.max_files) == (
        0.8,
        "related",
        0.5,
        2,
    )


def test_config_is_frozen(tmp_path):
    cfg = load_config({}, cwd=tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.model_name = "x"  # type: ignore[misc]
