import dataclasses
from pathlib import Path

import pytest

from minirag_mcp.config import DEFAULT_MODEL, ConfigError, load_config
from minirag_mcp.embedder import E5_LARGE, MINILM, default_budget
from minirag_mcp.endpoints import Endpoint


def test_defaults_from_cwd(tmp_path):
    cfg = load_config({}, cwd=tmp_path)
    assert cfg.roots == (tmp_path.resolve(),)
    assert cfg.db_path == tmp_path.resolve() / ".minirag" / "v2"
    assert cfg.model_name == DEFAULT_MODEL == "intfloat/multilingual-e5-large"
    assert cfg.max_file_size == 104857600
    assert cfg.token_budget == default_budget(E5_LARGE)
    assert (cfg.hybrid_weight, cfg.document_weight, cfg.rerank_top) == (0.6, 0.5, 30)
    assert cfg.embed_endpoints == {}
    assert cfg.rerank_endpoint is cfg.classify_endpoint is cfg.index_name is None
    assert cfg.grouping is None and cfg.max_distance is None and cfg.max_files is None
    assert cfg.allow_private_urls is False  # the SSRF guard is on unless turned off
    assert "minirag-mcp" in str(cfg.cache_dir)  # platformdirs cache, not cwd-relative


def test_e5_default_budget_is_the_measured_one(tmp_path):
    assert load_config({}, cwd=tmp_path).token_budget == 250  # g7q.1 notes, 2026-11


def test_base_dir_env(tmp_path):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=Path("/"))
    assert cfg.roots == (tmp_path.resolve(),)
    assert cfg.db_path == tmp_path.resolve() / ".minirag" / "v2"


def test_base_dirs_json_overrides_base_dir(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    env = {"BASE_DIR": "/ignored", "BASE_DIRS": f'["{a}", "{b}"]'}
    cfg = load_config(env, cwd=tmp_path)
    assert cfg.roots == (a.resolve(), b.resolve())
    assert cfg.db_path == a.resolve() / ".minirag" / "v2"  # first root hosts the index


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
    "env,var",
    [
        ({"MAX_FILE_SIZE": "abc"}, "MAX_FILE_SIZE"),
        ({"MAX_FILE_SIZE": "0"}, "MAX_FILE_SIZE"),
        ({"CHUNK_TOKEN_BUDGET": "0"}, "CHUNK_TOKEN_BUDGET"),
        ({"CHUNK_TOKEN_BUDGET": "513"}, "CHUNK_TOKEN_BUDGET"),
        ({"RAG_DOCUMENT_WEIGHT": "1.5"}, "RAG_DOCUMENT_WEIGHT"),
        ({"RAG_DOCUMENT_WEIGHT": "-0.1"}, "RAG_DOCUMENT_WEIGHT"),
        ({"RAG_DOCUMENT_WEIGHT": "x"}, "RAG_DOCUMENT_WEIGHT"),
        ({"RAG_RERANK_TOP": "0"}, "RAG_RERANK_TOP"),
        ({"RAG_RERANK_TOP": "201"}, "RAG_RERANK_TOP"),
        ({"EMBED_URL": "http://127.0.0.1:8080/v1/embeddings"}, "EMBED_URL"),
        ({"EMBED_ENDPOINT_MODEL": "e5"}, "EMBED_ENDPOINT_MODEL"),
        ({"RERANK_URL": "http://127.0.0.1:8080/v1/rerank"}, "RERANK_URL"),
        ({"CLASSIFY_MODEL": "plumb-4b"}, "CLASSIFY_MODEL"),
        ({"EMBED_URL": "ftp://h/v1/embeddings", "EMBED_ENDPOINT_MODEL": "e5"}, "EMBED_URL"),
        ({"ENDPOINT_API_KEY_FILE": "/nonexistent/key"}, "ENDPOINT_API_KEY_FILE"),
        ({"RAG_HYBRID_WEIGHT": "1.5"}, "RAG_HYBRID_WEIGHT"),
        ({"RAG_HYBRID_WEIGHT": "-0.1"}, "RAG_HYBRID_WEIGHT"),
        ({"RAG_GROUPING": "bogus"}, "RAG_GROUPING"),
        ({"RAG_MAX_DISTANCE": "-1"}, "RAG_MAX_DISTANCE"),
        ({"RAG_MAX_FILES": "0"}, "RAG_MAX_FILES"),
        ({"ALLOW_PRIVATE_URLS": "maybe"}, "ALLOW_PRIVATE_URLS"),
        ({"ALLOW_PRIVATE_URLS": "2"}, "ALLOW_PRIVATE_URLS"),
    ],
)
def test_invalid_numeric_env(env, var, tmp_path):
    with pytest.raises(ConfigError, match=var):
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


def test_ocr_defaults(tmp_path):
    cfg = load_config({}, cwd=tmp_path)
    assert cfg.ocr_lang == "eslav"
    assert cfg.ocr_min_chars_per_page == 25


def test_ocr_lang_override(tmp_path):
    cfg = load_config({"RAG_OCR_LANG": "cyrillic"}, cwd=tmp_path)
    assert cfg.ocr_lang == "cyrillic"


def test_ocr_min_chars_zero_allowed_and_negative_rejected(tmp_path):
    cfg = load_config({"RAG_OCR_MIN_CHARS_PER_PAGE": "0"}, cwd=tmp_path)
    assert cfg.ocr_min_chars_per_page == 0
    with pytest.raises(ConfigError):
        load_config({"RAG_OCR_MIN_CHARS_PER_PAGE": "-1"}, cwd=tmp_path)


def test_ocr_lang_blank_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config({"RAG_OCR_LANG": "  "}, cwd=tmp_path)


def test_the_budget_ceiling_is_the_models_window(tmp_path):
    assert load_config({"CHUNK_TOKEN_BUDGET": "512"}, cwd=tmp_path).token_budget == 512
    with pytest.raises(ConfigError, match="512"):
        load_config({"CHUNK_TOKEN_BUDGET": "513"}, cwd=tmp_path)
    with pytest.raises(ConfigError, match="128"):
        load_config({"MODEL_NAME": MINILM, "CHUNK_TOKEN_BUDGET": "129"}, cwd=tmp_path)
    assert load_config({"MODEL_NAME": "some/unknown"}, cwd=tmp_path).token_budget == 110
    with pytest.raises(ConfigError):
        load_config({"MODEL_NAME": "some/unknown", "CHUNK_TOKEN_BUDGET": "129"}, cwd=tmp_path)


def test_endpoints_come_from_the_environment(tmp_path):
    (tmp_path / "key").write_text("\n  secret  \n")
    env = {
        "EMBED_URL": "http://127.0.0.1:8080/v1/embeddings",
        "EMBED_ENDPOINT_MODEL": "multilingual-e5-large",
        "RERANK_URL": "http://127.0.0.1:8080/v1/rerank",
        "RERANK_MODEL": "bge-reranker-v2-m3",
        "CLASSIFY_URL": "http://127.0.0.1:8080",
        "CLASSIFY_MODEL": "plumb-4b",
        "ENDPOINT_API_KEY_FILE": str(tmp_path / "key"),
    }
    cfg = load_config(env, cwd=tmp_path)
    assert cfg.embed_endpoints == {
        E5_LARGE: Endpoint("http://127.0.0.1:8080/v1/embeddings", "multilingual-e5-large", "secret")
    }
    assert cfg.rerank_endpoint == Endpoint(
        "http://127.0.0.1:8080/v1/rerank", "bge-reranker-v2-m3", "secret"
    )
    assert cfg.classify_endpoint == Endpoint("http://127.0.0.1:8080", "plumb-4b", "secret")
    assert set(load_config({**env, "MODEL_NAME": MINILM}, cwd=tmp_path).embed_endpoints) == {MINILM}


def test_an_embed_endpoint_needs_a_model_of_known_window(tmp_path):
    """The endpoint gets text cut to the window where fastembed truncates in-process; for a
    model without a known window the two backends would embed different text."""
    env = {
        "EMBED_URL": "http://127.0.0.1:8080/v1/embeddings",
        "EMBED_ENDPOINT_MODEL": "m",
        "MODEL_NAME": "BAAI/bge-small-en-v1.5",
    }
    with pytest.raises(ConfigError, match=r"(?s)BAAI/bge-small-en-v1\.5.*window.*unknown"):
        load_config(env, cwd=tmp_path)
    with pytest.raises(ConfigError, match="window"):
        load_config(env, model_name_flag="x/unknown", cwd=tmp_path)
    assert set(load_config({**env, "MODEL_NAME": MINILM}, cwd=tmp_path).embed_endpoints) == {MINILM}
    in_process = {"MODEL_NAME": "BAAI/bge-small-en-v1.5"}
    assert load_config(in_process, cwd=tmp_path).embed_endpoints == {}


def test_retrieval_tuning_env(tmp_path):
    cfg = load_config({"RAG_DOCUMENT_WEIGHT": "0", "RAG_RERANK_TOP": "50"}, cwd=tmp_path)
    assert (cfg.document_weight, cfg.rerank_top) == (0.0, 50)


def test_the_model_flag_keys_the_embed_endpoint(tmp_path):
    env = {"EMBED_URL": "http://127.0.0.1:8080/v1/embeddings", "EMBED_ENDPOINT_MODEL": "m"}
    cfg = load_config(env, model_name_flag=MINILM, cwd=tmp_path)
    assert set(cfg.embed_endpoints) == {MINILM}


def test_the_api_key_never_appears_in_a_repr(tmp_path):
    (tmp_path / "key").write_text("secret\n")
    env = {
        "EMBED_URL": "http://127.0.0.1:8080/v1/embeddings",
        "EMBED_ENDPOINT_MODEL": "m",
        "ENDPOINT_API_KEY_FILE": str(tmp_path / "key"),
    }
    cfg = load_config(env, cwd=tmp_path)
    assert next(iter(cfg.embed_endpoints.values())).api_key == "secret"
    assert "secret" not in repr(cfg)
    assert "secret" not in repr(Endpoint("http://h", "n", api_key="secret"))
