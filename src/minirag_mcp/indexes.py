"""`indexes.toml`: the named indexes one server serves, and the server's own settings.

Each `[index.NAME]` becomes a `Config` through `load_config`, the code that reads the
environment: the file's keys are translated into the variables it reads, so one piece of
code checks every value, and its errors are translated back into the file's spelling. The
process's own environment never reaches an index; of `env`, only HOME is read.
"""

from __future__ import annotations

import dataclasses
import os
import re
import sys
import tomllib
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from minirag_mcp.config import Config, ConfigError, load_config
from minirag_mcp.endpoints import Endpoint
from minirag_mcp.security import LOOPBACK_NAMES, host_name, is_loopback_name

CONFIG_ENV = "MINIRAG_CONFIG"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# `[index.NAME]` key -> the variable `load_config` reads. A variable it gains goes in under
# its name, lowercased, with RAG_ dropped; a test fails until it does.
INDEX_KEYS: dict[str, str] = {
    "model": "MODEL_NAME",
    "db": "DB_PATH",
    "fts_language": "RAG_FTS_LANGUAGE",
    "token_budget": "CHUNK_TOKEN_BUDGET",
    "hybrid_weight": "RAG_HYBRID_WEIGHT",
    "document_weight": "RAG_DOCUMENT_WEIGHT",
    "rerank_top": "RAG_RERANK_TOP",
    "grouping": "RAG_GROUPING",
    "max_distance": "RAG_MAX_DISTANCE",
    "max_files": "RAG_MAX_FILES",
    "max_file_size": "MAX_FILE_SIZE",
    "allow_private_urls": "ALLOW_PRIVATE_URLS",
    "ocr_lang": "RAG_OCR_LANG",
    "ocr_min_chars_per_page": "RAG_OCR_MIN_CHARS_PER_PAGE",
    "ocr_second_reader_below": "RAG_OCR_SECOND_READER_BELOW",
    "sync_workers": "RAG_SYNC_WORKERS",
    "max_table_rows": "RAG_MAX_TABLE_ROWS",
    "exclude": "RAG_EXCLUDE",
}
# Set from `[server]` and the endpoint tables, never per index.
SERVER_ENV_KEYS = frozenset(
    {
        "CACHE_DIR",
        "ENDPOINT_API_KEY_FILE",
        "RERANK_URL",
        "RERANK_MODEL",
        "CLASSIFY_URL",
        "CLASSIFY_MODEL",
        "EMBED_URL",
        "EMBED_ENDPOINT_MODEL",
    }
)

_TOP_KEYS = ("server", "embedding", "rerank", "classify", "index")
_SERVER_KEYS = ("host", "port", "token_file", "api_key_file", "cache_dir", "allow_remote")
_SERVER_PATHS = {"cache_dir": "CACHE_DIR", "api_key_file": "ENDPOINT_API_KEY_FILE"}
_ENDPOINT_KEYS = ("url", "name")
_ENDPOINT_TABLES = {
    "rerank": ("RERANK_URL", "RERANK_MODEL"),
    "classify": ("CLASSIFY_URL", "CLASSIFY_MODEL"),
}
_INDEX_OWN_KEYS = ("roots", "description")
_INDEX_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
_MAX_PORT = 65535
# How a message from `load_config` names each variable, in the file's spelling.
_SPELLING = {
    **{var: key for key, var in INDEX_KEYS.items()},
    "CACHE_DIR": "[server] cache_dir",
    "ENDPOINT_API_KEY_FILE": "[server] api_key_file",
    "RERANK_URL": "[rerank] url",
    "RERANK_MODEL": "[rerank] name",
    "CLASSIFY_URL": "[classify] url",
    "CLASSIFY_MODEL": "[classify] name",
}


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    token_file: Path
    indexes: dict[str, Config]
    descriptions: dict[str, str]


def _home(env: Mapping[str, str]) -> Path:
    return Path(env["HOME"]) if env.get("HOME") else Path.home()


def _expanded(value: str, env: Mapping[str, str]) -> Path:
    if value == "~" or value.startswith("~/"):
        return _home(env) / value[2:]
    return Path(value)


def default_config_path(env: Mapping[str, str] | None = None) -> Path:
    """MINIRAG_CONFIG when set, else where this platform keeps a user's indexes.toml.

    A relative MINIRAG_CONFIG is a ConfigError: the token file next to it, and every
    reload, would follow the working directory, which is / under launchd. A relative
    XDG_CONFIG_HOME is ignored, as the XDG spec says."""
    env = os.environ if env is None else env
    named = env.get(CONFIG_ENV, "").strip()
    if named:
        path = _expanded(named, env)
        if not path.is_absolute():
            raise ConfigError(f"{CONFIG_ENV} must be an absolute path, got {named!r}")
        return path
    if sys.platform == "win32":
        # appauthor=False: by default it nests a folder named after the author, minirag again.
        return platformdirs.user_config_path("minirag", appauthor=False) / "indexes.toml"
    # Not platformdirs: on macOS it answers ~/Library/Application Support.
    base = Path(env.get("XDG_CONFIG_HOME", "").strip())
    if not base.is_absolute():
        base = _home(env) / ".config"
    return base / "minirag" / "indexes.toml"


def load_server_config(path: Path, env: Mapping[str, str]) -> ServerConfig:
    """The server and its indexes as `path` configures them. A ConfigError names `path`,
    and the table and key at fault."""
    data = _read(path)
    _known_keys(path, "the top level", data, _TOP_KEYS)
    server = _table(path, "[server]", data.get("server", {}))
    _known_keys(path, "[server]", server, _SERVER_KEYS)
    host, port = _address(path, server)
    if "token_file" in server:
        token_file = _path(path, "[server] token_file", server["token_file"], env)
    else:
        token_file = path.parent / "token"

    shared: dict[str, str] = {}
    for key, var in _SERVER_PATHS.items():
        if key in server:
            shared[var] = str(_path(path, f"[server] {key}", server[key], env))
    for table, (url_var, name_var) in _ENDPOINT_TABLES.items():
        if table in data:
            shared[url_var], shared[name_var] = _endpoint(path, f"[{table}]", data[table])
    # Checked once here, so that an error in them is not reported as one of the first index.
    _load(path, "", shared, [str(path.parent)], _SPELLING)
    embeddings = _embeddings(path, data.get("embedding", {}), shared)

    configs: dict[str, Config] = {}
    descriptions: dict[str, str] = {}
    for name, table in _table(path, "[index]", data.get("index", {})).items():
        configs[name], descriptions[name] = _index(path, name, table, shared, embeddings, env)
    if not configs:
        raise ConfigError(
            f"{path}: no index is configured; add an [index.NAME] table with its roots"
        )
    _refuse_shared_db(path, configs)
    return ServerConfig(host, port, token_file, configs, descriptions)


def _read(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    except UnicodeDecodeError as e:
        raise ConfigError(f"{path}: not UTF-8 text: {e}") from e
    except OSError as e:
        raise ConfigError(f"{path}: cannot read it: {e.strerror or e}") from e


def _table(path: Path, where: str, value: object) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: {where} must be a table, got {value!r}")
    return value


def _known_keys(path: Path, where: str, table: dict, known: Sequence[str]) -> None:
    for key in table:
        if key not in known:
            raise ConfigError(
                f"{path}: unknown key {key!r} in {where}; the keys there are {', '.join(known)}"
            )


def _path(path: Path, where: str, value: object, env: Mapping[str, str]) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: {where} must be a path, got {value!r}")
    expanded = _expanded(value, env)
    if not expanded.is_absolute():
        raise ConfigError(
            f"{path}: {where} must be absolute or start with ~, got {value!r}: a relative path "
            "would depend on the folder the server is started in"
        )
    return expanded


def _address(path: Path, server: dict) -> tuple[str, int]:
    host = server.get("host", DEFAULT_HOST)
    if not isinstance(host, str) or not host.strip():
        raise ConfigError(f"{path}: [server] host must be a host name, got {host!r}")
    if host_name(host) != host.lower():
        raise ConfigError(
            f"{path}: [server] host {host!r} cannot be listened on as written: give the bare "
            "name or address, without brackets, spaces or a port; the port goes in [server] port"
        )
    port = server.get("port", DEFAULT_PORT)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= _MAX_PORT:
        raise ConfigError(
            f"{path}: [server] port must be an integer from 1 to {_MAX_PORT}, got {port!r}"
        )
    allow_remote = server.get("allow_remote", False)
    if not isinstance(allow_remote, bool):
        raise ConfigError(
            f"{path}: [server] allow_remote must be true or false, got {allow_remote!r}"
        )
    if not allow_remote and not is_loopback_name(host):
        raise ConfigError(
            f"{path}: [server] host {host!r} is not a loopback address; the server listens "
            f"only on this machine: {', '.join(LOOPBACK_NAMES)}"
        )
    return host, port


def _endpoint(path: Path, where: str, value: object) -> tuple[str, str]:
    table = _table(path, where, value)
    _known_keys(path, where, table, _ENDPOINT_KEYS)
    for key in _ENDPOINT_KEYS:
        if not isinstance(table.get(key), str) or not table[key].strip():
            raise ConfigError(
                f"{path}: {where} {key} is required, a non-empty string; got {table.get(key)!r}"
            )
    return table["url"], table["name"]


def _respelled(message: str, spelling: Mapping[str, str]) -> str:
    names = "|".join(re.escape(var) for var in sorted(spelling, key=len, reverse=True))
    return re.sub(rf"\b({names})\b", lambda m: spelling[m.group(1)], message)


def _load(
    path: Path, where: str, env: dict[str, str], roots: list[str], spelling: Mapping[str, str]
) -> Config:
    try:
        return load_config(env, base_dir_flags=roots, cwd=path.parent)
    except ConfigError as e:
        raise ConfigError(f"{path}: {where}{_respelled(str(e), spelling)}") from e


def _embeddings(path: Path, value: object, shared: dict[str, str]) -> dict[str, Endpoint]:
    """Each `[embedding."<id>"]` table as the endpoint `load_config` builds from EMBED_URL
    for that model, so it is held to the same rules, the model's window among them."""
    endpoints = {}
    for model, table in _table(path, "[embedding]", value).items():
        where = f'[embedding."{model}"]'
        # load_config reads an empty MODEL_NAME as unset and would key the default model.
        if not model.strip():
            raise ConfigError(
                f"{path}: {where} names no model: write the fastembed model id between the quotes"
            )
        url, name = _endpoint(path, where, table)
        env = {"MODEL_NAME": model, "EMBED_URL": url, "EMBED_ENDPOINT_MODEL": name}
        if "ENDPOINT_API_KEY_FILE" in shared:
            env["ENDPOINT_API_KEY_FILE"] = shared["ENDPOINT_API_KEY_FILE"]
        spelling = {
            **_SPELLING,
            "EMBED_URL": f"{where} url",
            "EMBED_ENDPOINT_MODEL": f"{where} name",
        }
        endpoints[model] = _load(path, "", env, [str(path.parent)], spelling).embed_endpoints[model]
    return endpoints


def _index(
    path: Path,
    name: str,
    value: object,
    shared: dict[str, str],
    embeddings: dict[str, Endpoint],
    env: Mapping[str, str],
) -> tuple[Config, str]:
    if not _INDEX_NAME.fullmatch(name):
        raise ConfigError(
            f"{path}: index name {name!r} must be lowercase letters, digits and '-', starting "
            "with a letter or digit"
        )
    where = f"[index.{name}]"
    table = _table(path, where, value)
    _known_keys(path, where, table, (*_INDEX_OWN_KEYS, *INDEX_KEYS))
    roots = table.get("roots")
    if (
        not isinstance(roots, list)
        or not roots
        or not all(isinstance(root, str) and root.strip() for root in roots)
    ):
        raise ConfigError(
            f"{path}: {where} roots must be a non-empty list of folder paths, got {roots!r}"
        )
    description = table.get("description", "")
    if not isinstance(description, str):
        raise ConfigError(f"{path}: {where} description must be a string, got {description!r}")

    index_env = dict(shared)
    for key, setting in table.items():
        if key not in _INDEX_OWN_KEYS:
            index_env[INDEX_KEYS[key]] = _setting(path, where, key, setting, env)
    folders = [str(_path(path, f"{where} roots", root, env)) for root in roots]
    config = _load(path, f"{where} ", index_env, folders, _SPELLING)
    config = dataclasses.replace(
        config,
        index_name=name,
        embed_endpoints=dict(embeddings),
        exclude_source=f"{path} {where} exclude",
    )
    return config, description


def _setting(path: Path, where: str, key: str, value: object, env: Mapping[str, str]) -> str:
    if key == "db":
        return str(_path(path, f"{where} db", value, env))
    if key == "exclude":
        return _exclude(path, where, value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    raise ConfigError(
        f"{path}: {where} {key} must be a string, a number or true/false, got {value!r}"
    )


def _exclude(path: Path, where: str, value: object) -> str:
    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
        raise ConfigError(
            f"{path}: {where} exclude must be a list of gitignore patterns, got {value!r}"
        )
    for pattern in value:
        # The patterns reach `load_config` joined by ';'.
        if ";" in pattern:
            raise ConfigError(
                f"{path}: {where} exclude: {pattern!r} contains ';', which no exclude pattern "
                "can hold; match it with ? instead"
            )
    return ";".join(value)


def _directory_key(folder: Path) -> tuple:
    """One key for one folder however it is spelled. The part that exists is the folder
    itself, so a case-insensitive filesystem's `Docs` and `docs` are one. The part not made
    yet is compared without case or Unicode composition, as such a filesystem will make it;
    on a case-sensitive one that refuses two folders that would have been distinct, which is
    loud, where two indexes writing one directory is not."""
    tail: list[str] = []
    for existing in (folder, *folder.parents):
        try:
            st = os.stat(existing)
        except OSError:
            tail.append(unicodedata.normalize("NFC", existing.name).casefold())
            continue
        return (st.st_dev, st.st_ino, *reversed(tail))
    return (str(folder).casefold(),)


def _refuse_shared_db(path: Path, configs: dict[str, Config]) -> None:
    owners: dict[tuple, str] = {}
    for name, config in configs.items():
        other = owners.setdefault(_directory_key(config.db_path), name)
        if other != name:
            raise ConfigError(
                f"{path}: [index.{other}] and [index.{name}] would both write the index at "
                f"{config.db_path}, and two indexes cannot share one; give one of them its own "
                "db, or another first root"
            )
