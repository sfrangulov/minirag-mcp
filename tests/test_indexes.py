"""indexes.toml: one `Config` per `[index.NAME]`, every value checked by `load_config`."""

from __future__ import annotations

import sys

import pytest

from minirag_mcp import indexes
from minirag_mcp.config import ConfigError, load_config
from minirag_mcp.embedder import E5_LARGE, MINILM
from minirag_mcp.endpoints import Endpoint
from minirag_mcp.handle import configured_scheme, make_embedder
from minirag_mcp.indexes import (
    INDEX_KEYS,
    SERVER_ENV_KEYS,
    default_config_path,
    load_server_config,
)

EMBED_URL = "http://127.0.0.1:8080/v1/embeddings"
RERANK_URL = "http://127.0.0.1:8080/v1/rerank"
CLASSIFY_URL = "http://127.0.0.1:8080"
ONE_INDEX = '[index.a]\nroots = ["~/a"]\n'


def write(tmp_path, text: str):
    path = tmp_path / "config" / "indexes.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load(tmp_path, text: str, **env: str):
    return load_server_config(write(tmp_path, text), {"HOME": str(tmp_path), **env})


def refused(tmp_path, text: str, **env: str) -> str:
    with pytest.raises(ConfigError) as excinfo:
        load(tmp_path, text, **env)
    return str(excinfo.value)


def test_each_index_table_becomes_a_config(indexes_file):
    sc = load_server_config(indexes_file.path, indexes_file.env)
    assert list(sc.indexes) == ["alpha", "beta", "gamma"]
    alpha = sc.indexes["alpha"]
    assert alpha.roots == (indexes_file.roots["alpha"].resolve(),)
    assert alpha.db_path == indexes_file.roots["alpha"].resolve() / ".minirag" / "v2"
    assert alpha.index_name == "alpha" and sc.descriptions["alpha"] == "Alpha documents"
    assert sc.indexes["beta"].fts_language == "Russian"
    assert sc.indexes["gamma"].model_name == MINILM
    assert (sc.host, sc.port) == ("127.0.0.1", 8765)
    assert sc.token_file == indexes_file.path.parent / "token"


class RecordingEnv(dict):
    """An environment that records every name read from it."""

    def __init__(self, values: dict[str, str], seen: set[str]) -> None:
        super().__init__(values)
        self.seen = seen

    def get(self, key, default=None):
        self.seen.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.seen.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.seen.add(key)
        return super().__contains__(key)


def test_every_index_setting_maps_to_what_load_config_reads(tmp_path):
    seen: set[str] = set()
    endpoints = {"EMBED_URL": EMBED_URL, "RERANK_URL": RERANK_URL, "CLASSIFY_URL": CLASSIFY_URL}
    for env in ({}, endpoints):
        try:
            load_config(RecordingEnv(env, seen), cwd=tmp_path)
        except ConfigError:
            pass
    assert seen - {"BASE_DIR", "BASE_DIRS"} == set(INDEX_KEYS.values()) | SERVER_ENV_KEYS
    assert not set(INDEX_KEYS.values()) & SERVER_ENV_KEYS


# Each key with a value its default is not, and the Config field it sets.
SETTINGS = {
    "model": (f'"{MINILM}"', "model_name", MINILM),
    "fts_language": ('"russian"', "fts_language", "Russian"),
    "token_budget": ("200", "token_budget", 200),
    "hybrid_weight": ("0.75", "hybrid_weight", 0.75),
    "document_weight": ("0", "document_weight", 0.0),
    "rerank_top": ("50", "rerank_top", 50),
    "grouping": ('"related"', "grouping", "related"),
    "max_distance": ("0.5", "max_distance", 0.5),
    "max_files": ("3", "max_files", 3),
    "max_file_size": ("1000", "max_file_size", 1000),
    "allow_private_urls": ("true", "allow_private_urls", True),
    "ocr_lang": ('"eng"', "ocr_lang", "eng"),
    "ocr_min_chars_per_page": ("40", "ocr_min_chars_per_page", 40),
    "ocr_second_reader_below": ("0", "ocr_second_reader_below", 0),
    "sync_workers": ("4", "sync_workers", 4),
    "max_table_rows": ("500", "max_table_rows", 500),
    "exclude": ('["drafts/", "*.tmp.md"]', "exclude", ("drafts/", "*.tmp.md")),
}


def test_every_index_setting_has_a_case_below():
    assert set(SETTINGS) | {"db"} == set(INDEX_KEYS)


@pytest.mark.parametrize("key", sorted(SETTINGS))
def test_index_settings_reach_the_config(tmp_path, key):
    value, field, expected = SETTINGS[key]
    default = getattr(load(tmp_path, ONE_INDEX).indexes["a"], field)
    cfg = load(tmp_path, f"{ONE_INDEX}{key} = {value}\n").indexes["a"]
    assert getattr(cfg, field) == expected != default


def test_an_explicit_db_is_used_wherever_the_roots_are(tmp_path):
    cloud = tmp_path / "Cloud" / "Shared"
    text = (
        '[index.a]\nroots = ["~/Cloud/Shared"]\ndb = "~/indexes/shared"\n'
        'exclude = ["drafts/"]\nmax_table_rows = 500\n'
    )
    cfg = load(tmp_path, text).indexes["a"]
    assert cfg.roots == (cloud.resolve(),)
    assert cfg.db_path == (tmp_path / "indexes" / "shared").resolve()
    assert (cfg.exclude, cfg.max_table_rows) == (("drafts/",), 500)
    absolute = load(tmp_path, f'[index.a]\nroots = ["~/a"]\ndb = "{tmp_path / "x"}"\n')
    assert absolute.indexes["a"].db_path == (tmp_path / "x").resolve()


def test_the_process_environment_does_not_reach_a_file_index(tmp_path, monkeypatch):
    leaked = {
        "RAG_HYBRID_WEIGHT": "0.9",
        "DB_PATH": str(tmp_path / "elsewhere"),
        "RAG_EXCLUDE": "drafts/",
        "EMBED_URL": EMBED_URL,
        "EMBED_ENDPOINT_MODEL": "e5",
    }
    for name, value in leaked.items():
        monkeypatch.setenv(name, value)
    cfg = load(tmp_path, ONE_INDEX, **leaked).indexes["a"]
    assert cfg.hybrid_weight == 0.6
    assert cfg.db_path == (tmp_path / "a").resolve() / ".minirag" / "v2"
    assert cfg.exclude == () and cfg.embed_endpoints == {}


def test_a_bad_value_names_the_index_and_its_key(tmp_path):
    path = write(tmp_path, '[index.alpha]\nroots = ["~/a"]\nhybrid_weight = 1.5\n')
    with pytest.raises(ConfigError) as excinfo:
        load_server_config(path, {"HOME": str(tmp_path)})
    msg = str(excinfo.value)
    assert str(path) in msg and "[index.alpha]" in msg and "hybrid_weight" in msg
    assert "RAG_HYBRID_WEIGHT" not in msg


@pytest.mark.parametrize(
    "line, key",
    [
        ("hybrid_weight = [0.5]", "hybrid_weight"),
        ("grouping = { a = 1 }", "grouping"),
        ("description = 3", "description"),
        ('roots = "~/a"', "roots"),
        ("roots = []", "roots"),
        ('roots = ["~/a", ""]', "roots"),
        ('exclude = "drafts/"', "exclude"),
        ("exclude = [1]", "exclude"),
    ],
)
def test_a_value_of_the_wrong_type_names_its_key(tmp_path, line, key):
    msg = refused(
        tmp_path, f"[index.alpha]\n{line}\n" + ("" if "roots" in line else 'roots = ["~/a"]\n')
    )
    assert "[index.alpha]" in msg and key in msg


def test_a_refused_value_is_quoted_as_the_file_spells_it(tmp_path):
    msg = refused(tmp_path, f"{ONE_INDEX}max_files = true\n")
    assert "[index.a] max_files" in msg and "'true'" in msg


def test_an_index_needs_its_roots(tmp_path):
    msg = refused(tmp_path, '[index.alpha]\ndescription = "x"\n')
    assert "[index.alpha]" in msg and "roots" in msg


def test_a_syntax_error_names_the_line(tmp_path):
    path = write(tmp_path, "[server]\nport 8765\n")
    with pytest.raises(ConfigError) as excinfo:
        load_server_config(path, {"HOME": str(tmp_path)})
    assert str(path) in str(excinfo.value) and "line 2" in str(excinfo.value)


@pytest.mark.parametrize(
    "text, key, table",
    [
        ('[index.alpha]\nroots = ["~/a"]\ntokn_budget = 200\n', "tokn_budget", "[index.alpha]"),
        ('[server]\nprot = 1\n[index.a]\nroots = ["~/a"]\n', "prot", "[server]"),
        ('[indexes.a]\nroots = ["~/a"]\n', "indexes", "top level"),
        (
            '[rerank]\nurl = "http://127.0.0.1:1/v1/rerank"\nname = "r"\nmodle = "x"\n'
            '[index.a]\nroots = ["~/a"]\n',
            "modle",
            "[rerank]",
        ),
        (
            f'[embedding."{E5_LARGE}"]\nurl = "{EMBED_URL}"\nname = "e5"\nkey = "x"\n{ONE_INDEX}',
            "key",
            f'[embedding."{E5_LARGE}"]',
        ),
    ],
)
def test_unknown_keys_are_refused(tmp_path, text, key, table):
    msg = refused(tmp_path, text)
    assert repr(key) in msg and table in msg


@pytest.mark.parametrize("name", ["Client", "a_b", "-a", "a b"])
def test_index_names_are_lowercase_letters_digits_and_dashes(tmp_path, name):
    msg = refused(tmp_path, f'[index."{name}"]\nroots = ["~/a"]\n')
    assert repr(name) in msg and "lowercase" in msg


def test_index_names_may_hold_digits_and_dashes(tmp_path):
    assert list(load(tmp_path, '[index.client-2]\nroots = ["~/a"]\n').indexes) == ["client-2"]


def test_two_indexes_sharing_an_index_directory_are_refused(tmp_path):
    msg = refused(
        tmp_path,
        '[index.a]\nroots = ["~/docs", "~/a2"]\n[index.b]\nroots = ["~/docs", "~/b2"]\n',
    )
    assert "[index.a]" in msg and "[index.b]" in msg and ".minirag/v2" in msg


def test_two_indexes_whose_roots_differ_only_in_case_are_refused(tmp_path):
    (tmp_path / "docs").mkdir()
    if not (tmp_path / "DOCS").exists():
        pytest.skip("the temporary folder's filesystem tells case apart")
    msg = refused(tmp_path, '[index.a]\nroots = ["~/docs"]\n[index.b]\nroots = ["~/Docs"]\n')
    assert "[index.a]" in msg and "[index.b]" in msg and ".minirag/v2" in msg


def test_an_index_directory_not_made_yet_is_compared_without_case(tmp_path):
    msg = refused(
        tmp_path,
        '[index.a]\nroots = ["~/a"]\ndb = "~/idx/Work"\n'
        '[index.b]\nroots = ["~/b"]\ndb = "~/idx/work"\n',
    )
    assert "[index.a]" in msg and "[index.b]" in msg


def test_two_indexes_with_one_explicit_db_are_refused(tmp_path):
    msg = refused(
        tmp_path,
        '[index.a]\nroots = ["~/a"]\ndb = "~/idx"\n[index.b]\nroots = ["~/b"]\ndb = "~/idx/"\n',
    )
    assert "[index.a]" in msg and "[index.b]" in msg and str(tmp_path / "idx") in msg


@pytest.mark.parametrize("line", ['roots = ["docs"]', 'roots = ["~/d"]\ndb = "./idx"'])
def test_a_relative_path_is_refused(tmp_path, line):
    msg = refused(tmp_path, f"[index.a]\n{line}\n")
    assert "[index.a]" in msg and "absolute or start with ~" in msg


@pytest.mark.parametrize("key", ["token_file", "api_key_file", "cache_dir"])
def test_a_relative_server_path_is_refused(tmp_path, key):
    msg = refused(tmp_path, f'[server]\n{key} = "rel/x"\n{ONE_INDEX}')
    assert f"[server] {key}" in msg and "absolute or start with ~" in msg


def test_home_relative_paths_expand_against_the_given_home(tmp_path):
    home = tmp_path / "home"
    (home / "key").parent.mkdir(parents=True)
    (home / "key").write_text("secret\n")
    text = (
        '[server]\ntoken_file = "~/tok"\ncache_dir = "~/cache"\napi_key_file = "~/key"\n'
        f'[rerank]\nurl = "{RERANK_URL}"\nname = "bge"\n'
        '[index.a]\nroots = ["~/docs", "~"]\ndb = "~/idx"\n'
    )
    sc = load_server_config(write(tmp_path, text), {"HOME": str(home)})
    cfg = sc.indexes["a"]
    assert cfg.roots == ((home / "docs").resolve(), home.resolve())
    assert cfg.db_path == (home / "idx").resolve()
    assert cfg.cache_dir == (home / "cache").resolve()
    assert sc.token_file == home / "tok"
    assert cfg.rerank_endpoint.api_key == "secret"


def test_without_home_in_the_environment_the_users_home_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    sc = load_server_config(write(tmp_path, ONE_INDEX), {})
    assert sc.indexes["a"].roots == ((tmp_path / "h" / "a").resolve(),)


def test_endpoint_tables_reach_the_indexes_that_use_them(tmp_path):
    key = tmp_path / "key"
    key.write_text("secret\n")
    text = (
        f'[server]\napi_key_file = "{key}"\n'
        f'[embedding."{E5_LARGE}"]\nurl = "{EMBED_URL}"\nname = "multilingual-e5-large"\n'
        f'[rerank]\nurl = "{RERANK_URL}"\nname = "bge"\n'
        f'[classify]\nurl = "{CLASSIFY_URL}"\nname = "plumb"\n'
        '[index.e5]\nroots = ["~/e5"]\n'
        f'[index.mini]\nroots = ["~/mini"]\nmodel = "{MINILM}"\n'
    )
    sc = load(tmp_path, text)
    e5, mini = sc.indexes["e5"], sc.indexes["mini"]
    assert (
        e5.embed_endpoints
        == mini.embed_endpoints
        == {E5_LARGE: Endpoint(EMBED_URL, "multilingual-e5-large", api_key="secret")}
    )
    assert make_embedder(e5, configured_scheme(e5)).backend == "endpoint"
    assert make_embedder(mini, configured_scheme(mini)).backend == "in-process"
    assert (
        e5.rerank_endpoint == mini.rerank_endpoint == Endpoint(RERANK_URL, "bge", api_key="secret")
    )
    assert e5.classify_endpoint == Endpoint(CLASSIFY_URL, "plumb", api_key="secret")


def test_without_endpoint_tables_an_index_embeds_in_process(tmp_path):
    cfg = load(tmp_path, ONE_INDEX).indexes["a"]
    assert cfg.embed_endpoints == {} and cfg.rerank_endpoint is cfg.classify_endpoint is None


def test_an_embedding_url_that_is_not_http_is_refused(tmp_path):
    text = f'[embedding."{E5_LARGE}"]\nurl = "ftp://127.0.0.1/v1"\nname = "e5"\n{ONE_INDEX}'
    msg = refused(tmp_path, text)
    assert f'[embedding."{E5_LARGE}"] url' in msg and "http://" in msg
    assert "EMBED_URL" not in msg and "[index." not in msg


def test_an_embedding_table_for_a_model_of_unknown_window_is_refused(tmp_path):
    model = "BAAI/bge-small-en-v1.5"
    msg = refused(tmp_path, f'[embedding."{model}"]\nurl = "{EMBED_URL}"\nname = "b"\n{ONE_INDEX}')
    assert f'[embedding."{model}"] url' in msg and "window" in msg
    assert "EMBED_URL" not in msg


def test_a_bad_rerank_url_names_its_table_not_an_index(tmp_path):
    msg = refused(tmp_path, f'[rerank]\nurl = "ftp://x"\nname = "bge"\n{ONE_INDEX}')
    assert "[rerank] url" in msg and "RERANK_URL" not in msg and "[index." not in msg


@pytest.mark.parametrize(
    "table",
    ["[rerank]", "[classify]", f'[embedding."{E5_LARGE}"]'],
    ids=["rerank", "classify", "embedding"],
)
@pytest.mark.parametrize(
    "body", ['url = "http://127.0.0.1:1"\n', 'url = "http://127.0.0.1:1"\nname = " "\n']
)
def test_an_endpoint_without_its_name_is_refused(tmp_path, table, body):
    msg = refused(tmp_path, f"{table}\n{body}{ONE_INDEX}")
    assert table in msg and "name" in msg


@pytest.mark.parametrize("model", ["", "  "])
def test_an_embedding_table_without_a_model_id_is_refused(tmp_path, model):
    msg = refused(tmp_path, f'[embedding."{model}"]\nurl = "{EMBED_URL}"\nname = "e5"\n{ONE_INDEX}')
    assert f'[embedding."{model}"]' in msg and "model id" in msg


def test_an_embedding_entry_that_is_not_a_table_is_refused(tmp_path):
    msg = refused(tmp_path, f'[embedding]\nurl = "{EMBED_URL}"\n{ONE_INDEX}')
    assert '[embedding."url"]' in msg


def test_a_missing_api_key_file_is_refused(tmp_path):
    missing = tmp_path / "nokey"
    msg = refused(tmp_path, f'[server]\napi_key_file = "{missing}"\n{ONE_INDEX}')
    assert str(missing) in msg and "[server] api_key_file" in msg
    assert "ENDPOINT_API_KEY_FILE" not in msg


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com", "::ffff:127.0.0.1"])
def test_a_non_loopback_host_is_refused(tmp_path, host):
    msg = refused(tmp_path, f'[server]\nhost = "{host}"\n{ONE_INDEX}')
    assert "loopback" in msg and repr(host) in msg


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_accepted(tmp_path, host):
    assert load(tmp_path, f'[server]\nhost = "{host}"\nport = 9000\n{ONE_INDEX}').host == host


@pytest.mark.parametrize(
    "host", ["127.0.0.1:9000", "[::1]", "[::1]:8765", " localhost ", "0.0.0.0:80"]
)
def test_a_host_that_cannot_be_bound_as_written_is_refused(tmp_path, host):
    msg = refused(tmp_path, f'[server]\nhost = "{host}"\nallow_remote = true\n{ONE_INDEX}')
    assert repr(host) in msg and "[server] port" in msg


def test_allow_remote_lifts_the_host_check(tmp_path):
    sc = load(tmp_path, f'[server]\nhost = "0.0.0.0"\nallow_remote = true\n{ONE_INDEX}')
    assert (sc.host, sc.port) == ("0.0.0.0", 8765)


@pytest.mark.parametrize(
    "line, key",
    [
        ("port = 0", "port"),
        ("port = 65536", "port"),
        ('port = "8765"', "port"),
        ("port = true", "port"),
        ("host = 1", "host"),
        ('host = ""', "host"),
        ('allow_remote = "yes"', "allow_remote"),
        ("token_file = 1", "token_file"),
    ],
)
def test_a_bad_server_value_names_its_key(tmp_path, line, key):
    msg = refused(tmp_path, f"[server]\n{line}\n{ONE_INDEX}")
    assert f"[server] {key}" in msg


def test_a_token_file_can_be_named(tmp_path):
    sc = load(tmp_path, f'[server]\ntoken_file = "{tmp_path / "t"}"\n{ONE_INDEX}')
    assert sc.token_file == tmp_path / "t"


@pytest.mark.parametrize("text", ["", "[server]\nport = 8765\n", "[index]\n"])
def test_a_file_without_indexes_is_refused(tmp_path, text):
    path = write(tmp_path, text)
    with pytest.raises(ConfigError, match=r"\[index\.") as excinfo:
        load_server_config(path, {"HOME": str(tmp_path)})
    assert str(path) in str(excinfo.value)


def test_a_missing_file_is_refused_naming_it(tmp_path):
    path = tmp_path / "nowhere" / "indexes.toml"
    with pytest.raises(ConfigError) as excinfo:
        load_server_config(path, {"HOME": str(tmp_path)})
    assert str(path) in str(excinfo.value)


def test_a_file_that_is_not_utf8_is_refused_naming_it(tmp_path):
    path = write(tmp_path, "")
    path.write_bytes(b'[index.a]\nroots = ["~/\xff"]\n')
    with pytest.raises(ConfigError) as excinfo:
        load_server_config(path, {"HOME": str(tmp_path)})
    assert str(path) in str(excinfo.value)


def test_an_exclude_pattern_holding_a_semicolon_is_refused(tmp_path):
    path = write(tmp_path, '[index.a]\nroots = ["~/a"]\nexclude = ["drafts/", "a;b"]\n')
    with pytest.raises(ConfigError) as excinfo:
        load_server_config(path, {"HOME": str(tmp_path)})
    msg = str(excinfo.value)
    assert str(path) in msg and "[index.a] exclude" in msg and "'a;b'" in msg
    assert "RAG_EXCLUDE" not in msg


def test_a_refused_exclude_pattern_names_the_key_not_the_variable(tmp_path):
    msg = refused(tmp_path, '[index.a]\nroots = ["~/a"]\nexclude = ["#notes"]\n')
    assert "[index.a] exclude" in msg and "'#notes'" in msg and "RAG_EXCLUDE" not in msg


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_minirag_config_wins_on_every_platform(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_config_path({"MINIRAG_CONFIG": "~/x.toml"}) == tmp_path / "x.toml"


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_macos_and_linux_use_dot_config_not_application_support(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_config_path({}) == tmp_path / ".config" / "minirag" / "indexes.toml"
    assert (
        default_config_path({"XDG_CONFIG_HOME": str(tmp_path / "x")})
        == tmp_path / "x" / "minirag" / "indexes.toml"
    )
    assert default_config_path({"MINIRAG_CONFIG": "  "}) == default_config_path({})


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
@pytest.mark.parametrize("value", ["indexes.toml", "config/indexes.toml", "~other/x.toml"])
def test_a_relative_minirag_config_is_refused_naming_the_variable(monkeypatch, platform, value):
    """Relative to the working directory, which is / under launchd: the token file and
    every reload would follow wherever the process was started."""
    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(ConfigError) as refused:
        default_config_path({"MINIRAG_CONFIG": value})
    assert str(refused.value) == f"MINIRAG_CONFIG must be an absolute path, got {value!r}"


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_a_relative_xdg_config_home_is_ignored_as_the_xdg_spec_says(
    monkeypatch, tmp_path, platform
):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert (
        default_config_path({"XDG_CONFIG_HOME": "relative/config"})
        == tmp_path / ".config" / "minirag" / "indexes.toml"
    )


def test_windows_uses_platformdirs_without_an_author_folder(monkeypatch, tmp_path):
    from platformdirs import windows

    local = tmp_path / "AppData" / "Local"
    monkeypatch.setattr(sys, "platform", "win32")
    # platformdirs' own Windows rules, with only the folder lookup the OS answers stood in for.
    monkeypatch.setattr(windows, "get_win_folder", lambda csidl: str(local))
    monkeypatch.setattr(
        indexes.platformdirs,
        "user_config_path",
        lambda *args, **kwargs: windows.Windows(*args, **kwargs).user_config_path,
    )
    assert (
        default_config_path({"XDG_CONFIG_HOME": str(tmp_path / "x")})
        == local / "minirag" / "indexes.toml"
    )


def test_the_default_path_reads_the_process_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIRAG_CONFIG", str(tmp_path / "y.toml"))
    assert default_config_path() == tmp_path / "y.toml"
