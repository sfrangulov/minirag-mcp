# minirag-mcp v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Local-first RAG MCP server in Python — 10 MCP tools + CLI over one LanceDB index, per spec `docs/superpowers/specs/2026-08-07-minirag-mcp-v1-design.md`.

**Architecture:** Everything converts to Markdown via markitdown, gets chunked (structural split + semantic merge), embedded with fastembed, and stored in LanceDB (vectors + BM25 FTS). One entry point: no args → MCP server on stdio (fastmcp), subcommand → CLI (cyclopts). Search is hybrid: explicit query vector + text with `LinearCombinationReranker`.

**Tech Stack:** Python ≥3.11, uv, fastmcp 3.x, lancedb ≥0.36, fastembed ≥0.8, markitdown[pdf,docx,pptx,xlsx] ≥0.1.7, cyclopts ≥4, platformdirs ≥4, pytest + pytest-asyncio, ruff.

## Global Constraints

- All code, docs, identifiers, commit messages: **English**. (Spec: "All documentation, tool descriptions, and code are in English.")
- Default embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim).
- Default `DB_PATH`: `<first root>/.minirag/lancedb`. Default `CACHE_DIR`: `platformdirs.user_cache_path("minirag-mcp") / "models"`.
- Env vars: `BASE_DIR`, `BASE_DIRS` (JSON array, hard error if invalid), `DB_PATH`, `CACHE_DIR`, `MODEL_NAME`, `MAX_FILE_SIZE` (default 104857600), `CHUNK_MIN_LENGTH` (default 50, range 1–10000), `RAG_HYBRID_WEIGHT` (default 0.6, range 0–1), `RAG_GROUPING` (`similar`|`related`), `RAG_MAX_DISTANCE`, `RAG_MAX_FILES`.
- `ingest_url`: **http/https only** — `file:`/`data:` URIs must be rejected (root-boundary bypass).
- MCP file paths must be absolute and inside a configured root; symlink escapes rejected.
- Scanner whitelist: `.md .markdown .txt .pdf .docx .pptx .xlsx .html .htm .csv .epub .ipynb`; skip hidden entries and `node_modules`, `__pycache__`, `.venv`, `venv`.
- `status` tool must work even when config is invalid.
- Tests never touch the network; the real model is only used in tests marked `slow` (excluded by default via addopts).
- Verified API facts (live-probed 2026-08-07, lancedb 0.36 / fastmcp 3.4.6 / markitdown 0.1.7 / fastembed 0.8 / cyclopts 4.22):
  - FTS index: `from lancedb.index import FTS; tbl.create_index("text", config=FTS())` (`create_fts_index` is deprecated). Rows added after index creation ARE found by FTS.
  - Hybrid works without a registered embedding function: `tbl.search(query_type="hybrid").vector(vec).text(q).limit(n).rerank(r).to_list()` → rows carry `_relevance_score`.
  - `LinearCombinationReranker(weight=w)`: **w is the weight of the VECTOR score** ⇒ use `weight = 1.0 - hybrid_weight`.
  - Vector search rows carry `_distance`; FTS rows carry `_score`; `.where(clause, prefilter=True)` works.
  - markitdown: `.convert(path)`/`.convert_url(url)`/`.convert_stream(BytesIO, file_extension=".html")` → result has `.markdown` and `.title` (`.title` is often `None` — fallback needed). Exceptions: `MarkItDownException`, `UnsupportedFormatException`, `FileConversionException`, `MissingDependencyException`.
  - fastembed: `TextEmbedding.list_supported_models()` entries are dicts with `dim`; `TextEmbedding(model_name=..., cache_dir=...)`; has `.embed(list)` and `.query_embed(str)`.
  - cyclopts: `@app.default` handles no-subcommand; `list[str]` params give repeatable flags; underscores become dashes; `Annotated[bool, Parameter(name="--json")]` renames; tests call `app(tokens, result_action="return_value")`; default result_action prints and sys.exits.
  - fastmcp: `@mcp.tool`, `raise ToolError("msg")`, in-memory test client `async with Client(mcp) as c: (await c.call_tool(name, args)).data`.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/minirag_mcp/__init__.py`, `tests/test_scaffold.py`

**Interfaces:**
- Produces: importable package `minirag_mcp` with `__version__`; `uv run pytest` green; entry point `minirag-mcp = "minirag_mcp.__main__:main"` declared (module lands in Task 13).

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "minirag-mcp"
version = "0.1.0"
description = "Local-first RAG MCP server: hybrid search over your documents, fully local"
readme = "README.md"
requires-python = ">=3.11"
license = "MIT"
dependencies = [
    "fastmcp>=3.4",
    "lancedb>=0.36",
    "fastembed>=0.8",
    "markitdown[pdf,docx,pptx,xlsx]>=0.1.7",
    "cyclopts>=4.0",
    "platformdirs>=4.0",
]

[project.scripts]
minirag-mcp = "minirag_mcp.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/minirag_mcp"]

[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=1.0", "ruff>=0.6"]

[tool.pytest.ini_options]
addopts = "-m 'not slow'"
markers = ["slow: needs the real embedding model (downloads ~220 MB)"]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 2: Create package and smoke test**

`src/minirag_mcp/__init__.py`:

```python
__version__ = "0.1.0"
```

`tests/test_scaffold.py`:

```python
import minirag_mcp


def test_package_importable():
    assert minirag_mcp.__version__ == "0.1.0"
```

- [ ] **Step 3: Install and run**

Run: `uv sync && uv run pytest -q`
Expected: 1 passed. (First `uv sync` resolves and installs all deps — takes a minute.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src tests uv.lock
git commit -m "feat: project scaffold (uv, pyproject, package skeleton)"
```

---

### Task 2: config.py

**Files:**
- Create: `src/minirag_mcp/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
  - `class ConfigError(Exception)`
  - `@dataclass(frozen=True) Config(roots: tuple[Path, ...], db_path: Path, cache_dir: Path, model_name: str, max_file_size: int, chunk_min_length: int, hybrid_weight: float, grouping: str | None, max_distance: float | None, max_files: int | None)`
  - `load_config(env: Mapping[str, str], *, base_dir_flags: Sequence[str] = (), db_path_flag: str | None = None, cache_dir_flag: str | None = None, model_name_flag: str | None = None, cwd: Path | None = None) -> Config`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
from pathlib import Path

import pytest

from minirag_mcp.config import DEFAULT_MODEL, Config, ConfigError, load_config


def test_defaults_from_cwd(tmp_path):
    cfg = load_config({}, cwd=tmp_path)
    assert cfg.roots == (tmp_path.resolve(),)
    assert cfg.db_path == tmp_path.resolve() / ".minirag" / "lancedb"
    assert cfg.model_name == DEFAULT_MODEL
    assert cfg.max_file_size == 104857600
    assert cfg.chunk_min_length == 50
    assert cfg.hybrid_weight == 0.6
    assert cfg.grouping is None and cfg.max_distance is None and cfg.max_files is None
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
    ],
)
def test_invalid_numeric_env(env, tmp_path):
    with pytest.raises(ConfigError):
        load_config(env, cwd=tmp_path)


def test_search_tuning_env(tmp_path):
    env = {
        "RAG_HYBRID_WEIGHT": "0.8",
        "RAG_GROUPING": "related",
        "RAG_MAX_DISTANCE": "0.5",
        "RAG_MAX_FILES": "2",
    }
    cfg = load_config(env, cwd=tmp_path)
    assert (cfg.hybrid_weight, cfg.grouping, cfg.max_distance, cfg.max_files) == (0.8, "related", 0.5, 2)


def test_config_is_frozen(tmp_path):
    cfg = load_config({}, cwd=tmp_path)
    with pytest.raises(Exception):
        cfg.model_name = "x"  # type: ignore[misc]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'minirag_mcp.config'`

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/config.py
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
            raise ConfigError("BASE_DIRS must be a JSON array of one or more non-empty path strings")
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
        raise ConfigError(f"{key} must be in range [{lo}, {hi if hi is not None else '∞'}], got {val}")
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
    cache_dir = _resolve(cache_raw) if cache_raw else platformdirs.user_cache_path("minirag-mcp") / "models"

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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/config.py tests/test_config.py
git commit -m "feat: config loading with flags > env > defaults precedence"
```

---

### Task 3: security.py

**Files:**
- Create: `src/minirag_mcp/security.py`, `tests/test_security.py`

**Interfaces:**
- Produces:
  - `class SecurityError(Exception)`
  - `resolve_in_roots(path_str: str, roots: Sequence[Path], *, require_absolute: bool = True) -> Path` — expanduser+resolve (follows symlinks), raises `SecurityError` if outside every root or not absolute when required
  - `check_url_scheme(url: str) -> None` — raises `SecurityError` unless scheme is http/https

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_security.py
import pytest

from minirag_mcp.security import SecurityError, check_url_scheme, resolve_in_roots


def test_inside_root_ok(tmp_path):
    f = tmp_path / "docs" / "a.md"
    f.parent.mkdir()
    f.write_text("x")
    assert resolve_in_roots(str(f), [tmp_path]) == f.resolve()


def test_root_itself_ok(tmp_path):
    assert resolve_in_roots(str(tmp_path), [tmp_path]) == tmp_path.resolve()


def test_outside_root_rejected(tmp_path):
    other = tmp_path / "in"
    other.mkdir()
    with pytest.raises(SecurityError, match="outside"):
        resolve_in_roots("/etc/passwd", [other])


def test_dotdot_traversal_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(SecurityError, match="outside"):
        resolve_in_roots(str(root / ".." / "escape.md"), [root])


def test_relative_path_rejected_when_absolute_required(tmp_path):
    with pytest.raises(SecurityError, match="absolute"):
        resolve_in_roots("relative/a.md", [tmp_path])


def test_relative_ok_when_allowed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.md").write_text("x")
    assert resolve_in_roots("a.md", [tmp_path], require_absolute=False) == (tmp_path / "a.md").resolve()


def test_symlink_escape_rejected(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.md"
    secret.write_text("s")
    link = root / "link.md"
    link.symlink_to(secret)
    with pytest.raises(SecurityError, match="outside"):
        resolve_in_roots(str(link), [root])


def test_multiple_roots_second_matches(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    f = b / "doc.md"
    f.write_text("x")
    assert resolve_in_roots(str(f), [a, b]) == f.resolve()


@pytest.mark.parametrize("url", ["http://x.io/p", "https://x.io/p"])
def test_url_ok(url):
    check_url_scheme(url)  # no raise


@pytest.mark.parametrize("url", ["file:///etc/passwd", "data:text/html,hi", "ftp://x.io", "x.io/nope"])
def test_url_bad_scheme(url):
    with pytest.raises(SecurityError, match="http"):
        check_url_scheme(url)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_security.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/security.py
"""Document-root containment and URL scheme checks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse


class SecurityError(Exception):
    pass


def resolve_in_roots(
    path_str: str, roots: Sequence[Path], *, require_absolute: bool = True
) -> Path:
    p = Path(path_str).expanduser()
    if require_absolute and not p.is_absolute():
        raise SecurityError(f"Path must be absolute: {path_str}")
    real = p.resolve()  # follows symlinks; the check below sees the true target
    for root in roots:
        real_root = root.resolve()
        if real == real_root or real_root in real.parents:
            return real
    raise SecurityError(f"Path outside configured document roots: {path_str}")


def check_url_scheme(url: str) -> None:
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise SecurityError(
            f"Only http/https URLs are allowed, got scheme {scheme or '(none)'!r}"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_security.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/security.py tests/test_security.py
git commit -m "feat: root containment and URL scheme security checks"
```

---

### Task 4: chunker/structural.py

**Files:**
- Create: `src/minirag_mcp/chunker/__init__.py` (empty), `src/minirag_mcp/chunker/structural.py`, `tests/test_chunker_structural.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Block(text: str, is_code: bool)`
  - `split_markdown(markdown: str, *, max_chars: int = 1500) -> list[Block]` — units are paragraphs; a heading line glues to the following paragraph; fenced code blocks (``` or ~~~) are atomic `is_code=True` blocks even when longer than `max_chars`; over-long text paragraphs are split at whitespace boundaries to fit `max_chars`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunker_structural.py
from minirag_mcp.chunker.structural import Block, split_markdown


def test_empty_doc():
    assert split_markdown("") == []
    assert split_markdown("   \n\n  ") == []


def test_paragraphs_are_blocks():
    blocks = split_markdown("Para one.\n\nPara two.")
    assert [b.text for b in blocks] == ["Para one.", "Para two."]
    assert all(not b.is_code for b in blocks)


def test_heading_glues_to_next_paragraph():
    blocks = split_markdown("# Title\n\nIntro text.\n\n## Section\n\nBody.")
    assert blocks[0].text == "# Title\n\nIntro text."
    assert blocks[1].text == "## Section\n\nBody."


def test_trailing_heading_is_own_block():
    blocks = split_markdown("Body.\n\n# Dangling")
    assert blocks[-1].text == "# Dangling"


def test_code_fence_is_atomic_with_blank_lines():
    md = "Intro.\n\n```python\nx = 1\n\n\ny = 2\n```\n\nOutro."
    blocks = split_markdown(md)
    assert [b.is_code for b in blocks] == [False, True, False]
    assert "x = 1\n\n\ny = 2" in blocks[1].text
    assert blocks[1].text.startswith("```python") and blocks[1].text.endswith("```")


def test_tilde_fence():
    blocks = split_markdown("~~~\ncode here\n~~~")
    assert blocks == [Block(text="~~~\ncode here\n~~~", is_code=True)]


def test_unterminated_fence_runs_to_eof():
    blocks = split_markdown("```\nno closing fence\nstill code")
    assert len(blocks) == 1 and blocks[0].is_code


def test_long_code_fence_never_split():
    md = "```\n" + "\n".join(f"line {i}" for i in range(500)) + "\n```"
    blocks = split_markdown(md, max_chars=100)
    assert len(blocks) == 1 and blocks[0].is_code


def test_long_paragraph_split_at_whitespace():
    md = " ".join(f"word{i}" for i in range(400))
    blocks = split_markdown(md, max_chars=200)
    assert len(blocks) > 1
    assert all(len(b.text) <= 200 for b in blocks)
    joined = " ".join(b.text for b in blocks)
    assert joined.split() == md.split()  # no words lost
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_chunker_structural.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/chunker/structural.py
"""Stage 1 chunking: split Markdown into structural blocks.

Units are paragraphs. Headings glue to the paragraph that follows them so a
heading never dangles alone. Fenced code blocks are atomic and never split,
regardless of size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"^(```+|~~~+)")
_HEADING_RE = re.compile(r"^#{1,6}\s")


@dataclass(frozen=True)
class Block:
    text: str
    is_code: bool


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    words = text.split(" ")
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            pieces.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def split_markdown(markdown: str, *, max_chars: int = 1500) -> list[Block]:
    lines = markdown.split("\n")
    blocks: list[Block] = []
    para: list[str] = []          # accumulating non-code lines
    pending_heading: str | None = None

    def flush_para() -> None:
        nonlocal pending_heading
        text = "\n".join(para).strip()
        para.clear()
        if not text:
            return
        if pending_heading is not None:
            text = f"{pending_heading}\n\n{text}"
            pending_heading = None
        for piece in _split_long_paragraph(text, max_chars):
            blocks.append(Block(text=piece, is_code=False))

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE_RE.match(line.lstrip())
        if fence:
            flush_para()
            marker = fence.group(1)[0] * 3
            code = [line]
            i += 1
            while i < len(lines):
                code.append(lines[i])
                if lines[i].lstrip().startswith(marker):
                    break
                i += 1
            if pending_heading is not None:
                code.insert(0, "")
                code.insert(0, pending_heading)
                pending_heading = None
            blocks.append(Block(text="\n".join(code).strip(), is_code=True))
        elif _HEADING_RE.match(line):
            flush_para()
            if pending_heading is not None:
                blocks.append(Block(text=pending_heading, is_code=False))
            pending_heading = line.strip()
        elif line.strip() == "":
            flush_para()
        else:
            para.append(line)
        i += 1

    flush_para()
    if pending_heading is not None:
        blocks.append(Block(text=pending_heading, is_code=False))
    return blocks
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_chunker_structural.py -q`
Expected: all pass. If `test_heading_glues_to_next_paragraph` fails on blank-line handling, remember: blank line after a heading must NOT flush the pending heading — only `flush_para` consumes it.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/chunker tests/test_chunker_structural.py
git commit -m "feat: structural markdown chunker with atomic code fences"
```

---

### Task 5: chunker/semantic.py

**Files:**
- Create: `src/minirag_mcp/chunker/semantic.py`, `tests/test_chunker_semantic.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `Block` from `minirag_mcp.chunker.structural`.
- Produces:
  - `cosine(a: Sequence[float], b: Sequence[float]) -> float` (0.0 for zero vectors)
  - `merge_blocks(blocks: list[Block], embed: Callable[[Sequence[str]], list[list[float]]], *, max_chars: int = 1500, min_length: int = 50, similarity_threshold: float = 0.60) -> list[str]` — adjacent similar text blocks merge while the result fits `max_chars`; code blocks skip the similarity test and merge into the previous chunk when they fit; a final pass merges chunks shorter than `min_length` into a neighbor.
  - `tests/conftest.py` provides `FakeEmbedder` (dim=8, deterministic sha256-based vectors) used by later tasks: methods `embed_documents(texts) -> list[list[float]]`, `embed_query(text) -> list[float]`, attr `dim = 8`, `model_name = "fake"`.

- [ ] **Step 1: Write conftest with FakeEmbedder**

```python
# tests/conftest.py
"""Shared fixtures: deterministic fake embedder (no model download)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import pytest


class FakeEmbedder:
    """Deterministic 8-dim embeddings from sha256 — same text, same vector."""

    dim = 8
    model_name = "fake"

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [int.from_bytes(h[i : i + 4], "big") / 2**32 - 0.5 for i in range(0, 32, 4)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_chunker_semantic.py
from minirag_mcp.chunker.semantic import cosine, merge_blocks
from minirag_mcp.chunker.structural import Block


class StubEmbedder:
    """Maps exact text -> vector; unknown text -> orthogonal-ish default."""

    def __init__(self, table):
        self.table = table

    def __call__(self, texts):
        return [self.table[t] for t in texts]


def T(text):
    return Block(text=text, is_code=False)


def C(text):
    return Block(text=text, is_code=True)


def test_cosine():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([0, 0], [1, 0]) == 0.0  # zero vector guard


def test_empty():
    assert merge_blocks([], StubEmbedder({})) == []


def test_similar_neighbors_merge():
    embed = StubEmbedder({"a": [1.0, 0.0], "b": [0.99, 0.1], "c": [-1.0, 0.0]})
    out = merge_blocks([T("a"), T("b"), T("c")], embed, min_length=1)
    assert out == ["a\n\nb", "c"]


def test_max_chars_stops_merge():
    embed = StubEmbedder({"x" * 60: [1.0, 0.0], "y" * 60: [1.0, 0.0]})
    out = merge_blocks([T("x" * 60), T("y" * 60)], embed, max_chars=100, min_length=1)
    assert len(out) == 2  # merged would be 122 chars > 100


def test_code_block_merges_without_similarity():
    embed = StubEmbedder({"intro text": [1.0, 0.0], "```\ncode\n```": [-1.0, 0.0]})
    out = merge_blocks([T("intro text"), C("```\ncode\n```")], embed, min_length=1)
    assert out == ["intro text\n\n```\ncode\n```"]  # dissimilar but code attaches


def test_short_chunk_folds_into_neighbor():
    embed = StubEmbedder({"tiny": [1.0, 0.0], "long enough paragraph": [-1.0, 0.0]})
    out = merge_blocks([T("tiny"), T("long enough paragraph")], embed, min_length=10)
    assert out == ["tiny\n\nlong enough paragraph"]


def test_dissimilar_stay_separate():
    embed = StubEmbedder({"first topic here": [1.0, 0.0], "second topic here": [0.0, 1.0]})
    out = merge_blocks([T("first topic here"), T("second topic here")], embed, min_length=1)
    assert len(out) == 2
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_chunker_semantic.py -q`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement**

```python
# src/minirag_mcp/chunker/semantic.py
"""Stage 2 chunking: merge adjacent structural blocks by embedding similarity."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from minirag_mcp.chunker.structural import Block

SEPARATOR = "\n\n"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def merge_blocks(
    blocks: list[Block],
    embed: Callable[[Sequence[str]], list[list[float]]],
    *,
    max_chars: int = 1500,
    min_length: int = 50,
    similarity_threshold: float = 0.60,
) -> list[str]:
    if not blocks:
        return []
    vectors = embed([b.text for b in blocks])

    chunks: list[str] = [blocks[0].text]
    prev_vec = vectors[0]
    for block, vec in zip(blocks[1:], vectors[1:]):
        candidate = chunks[-1] + SEPARATOR + block.text
        fits = len(candidate) <= max_chars
        similar = block.is_code or cosine(prev_vec, vec) >= similarity_threshold
        if fits and similar:
            chunks[-1] = candidate
        else:
            chunks.append(block.text)
        prev_vec = vec

    # Fold under-length chunks into a neighbor (previous if any, else next).
    folded: list[str] = []
    for chunk in chunks:
        if folded and len(folded[-1]) < min_length:
            folded[-1] = folded[-1] + SEPARATOR + chunk
        else:
            folded.append(chunk)
    if len(folded) >= 2 and len(folded[-1]) < min_length:
        tail = folded.pop()
        folded[-1] = folded[-1] + SEPARATOR + tail
    return folded
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_chunker_semantic.py -q`
Expected: all pass. Note `test_short_chunk_folds_into_neighbor` exercises the "previous chunk is short" branch: `tiny` is < 10 chars so the next chunk folds INTO it.

- [ ] **Step 6: Commit**

```bash
git add src/minirag_mcp/chunker/semantic.py tests/test_chunker_semantic.py tests/conftest.py
git commit -m "feat: semantic merge chunker with code-block affinity"
```

---

### Task 6: embedder.py

**Files:**
- Create: `src/minirag_mcp/embedder.py`, `tests/test_embedder.py`, `tests/integration/test_real_model.py`

**Interfaces:**
- Produces:
  - `class Embedder` with `__init__(model_name: str, cache_dir: Path)`, lazy model load on first embed; `embed_documents(texts: Sequence[str]) -> list[list[float]]`; `embed_query(text: str) -> list[float]`; property `dim: int` (registry lookup, **no** model download); attr `model_name: str`.
  - `class UnknownModelError(Exception)` raised by `dim` for models absent from the fastembed registry.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embedder.py
from pathlib import Path

import pytest

from minirag_mcp.config import DEFAULT_MODEL
from minirag_mcp.embedder import Embedder, UnknownModelError


def test_dim_from_registry_without_download(tmp_path):
    emb = Embedder(DEFAULT_MODEL, cache_dir=tmp_path)
    assert emb.dim == 384
    assert not any(tmp_path.iterdir())  # nothing downloaded


def test_unknown_model_dim_raises(tmp_path):
    emb = Embedder("no-such/model", cache_dir=tmp_path)
    with pytest.raises(UnknownModelError):
        _ = emb.dim


def test_lazy_no_model_instantiation_on_init(tmp_path, monkeypatch):
    import minirag_mcp.embedder as mod

    def boom(*a, **k):
        raise AssertionError("TextEmbedding must not be constructed in __init__")

    monkeypatch.setattr(mod, "TextEmbedding", boom)
    Embedder(DEFAULT_MODEL, cache_dir=tmp_path)  # must not raise
```

```python
# tests/integration/test_real_model.py
"""Slow tests: real model download (~220 MB). Run with: uv run pytest -m slow"""

import pytest

from minirag_mcp.config import DEFAULT_MODEL
from minirag_mcp.embedder import Embedder

pytestmark = pytest.mark.slow


def test_real_embeddings_shape_and_similarity(tmp_path):
    emb = Embedder(DEFAULT_MODEL, cache_dir=tmp_path / "models")
    vecs = emb.embed_documents(["Аутентификация через токен", "Token-based authentication", "Рецепт борща"])
    assert len(vecs) == 3 and all(len(v) == 384 for v in vecs)
    from minirag_mcp.chunker.semantic import cosine

    ru_en = cosine(vecs[0], vecs[1])
    ru_off = cosine(vecs[0], vecs[2])
    assert ru_en > ru_off  # multilingual model: RU/EN same topic closer than unrelated RU
    q = emb.embed_query("аутентификация")
    assert len(q) == 384
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_embedder.py -q`
Expected: FAIL — module not found. Also run `uv run pytest -q` and confirm the slow test is NOT collected (addopts excludes it).

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/embedder.py
"""fastembed wrapper: lazy model load, registry-based dim lookup."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from fastembed import TextEmbedding


class UnknownModelError(Exception):
    pass


class Embedder:
    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model: TextEmbedding | None = None
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            for entry in TextEmbedding.list_supported_models():
                if entry["model"] == self.model_name:
                    self._dim = int(entry["dim"])
                    break
            else:
                raise UnknownModelError(
                    f"Model {self.model_name!r} is not in the fastembed registry. "
                    "See TextEmbedding.list_supported_models() for valid names."
                )
        return self._dim

    def _load(self) -> TextEmbedding:
        if self._model is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(x) for x in v] for v in self._load().embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [[float(x) for x in v] for v in self._load().query_embed([text])][0]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_embedder.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/embedder.py tests/test_embedder.py tests/integration
git commit -m "feat: lazy fastembed wrapper with registry dim lookup"
```

---

### Task 7: store.py — schema and CRUD

**Files:**
- Create: `src/minirag_mcp/store.py`, `tests/test_store.py`

**Interfaces:**
- Produces (search lands in Task 8; this task delivers everything else):
  - `@dataclass(frozen=True) ChunkRecord(id: str, source: str, source_type: str, title: str, chunk_index: int, text: str, vector: list[float], file_hash: str, mtime: float, ingested_at: str)`
  - `@dataclass(frozen=True) SearchResult(text: str, source: str, title: str, chunk_index: int, score: float, distance: float | None)`
  - `@dataclass(frozen=True) SourceInfo(source: str, source_type: str, title: str, chunk_count: int, file_hash: str, mtime: float)`
  - `class Store` with `__init__(db_path: Path, dim: int)` (creates dir/table/FTS index idempotently), `replace_source(source, records: Sequence[ChunkRecord]) -> None`, `delete_source(source) -> int`, `neighbors(source, chunk_index, before=1, after=1) -> list[SearchResult]` (score=0.0, distance=None), `all_chunks(source) -> list[SearchResult]` (every chunk of the source in `chunk_index` order), `list_sources(scopes: tuple[str, ...] = ()) -> list[SourceInfo]` (sorted by source), `get_source(source) -> SourceInfo | None`, `chunk_count() -> int`, `source_count() -> int`
  - Module helpers: `_sql_str(s: str) -> str` (escape `'` → `''`), `_scope_clause(scopes) -> str | None` (`source LIKE 'p%' OR ...`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store.py
from pathlib import Path

import pytest

from minirag_mcp.store import ChunkRecord, Store


def rec(source, i, text, vec=None, source_type="file", title="T"):
    return ChunkRecord(
        id=f"{source}#{i}", source=source, source_type=source_type, title=title,
        chunk_index=i, text=text, vector=vec or [0.1 * (i + 1)] * 8,
        file_hash="h", mtime=1.0, ingested_at="2026-08-07T00:00:00+00:00",
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "db", dim=8)


def test_empty_store_counts(store):
    assert store.chunk_count() == 0
    assert store.source_count() == 0
    assert store.list_sources() == []
    assert store.get_source("/nope.md") is None


def test_replace_and_get(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "alpha"), rec("/a.md", 1, "beta")])
    assert store.chunk_count() == 2
    info = store.get_source("/a.md")
    assert info.chunk_count == 2 and info.file_hash == "h" and info.source_type == "file"


def test_replace_is_atomic_swap(store):
    store.replace_source("/a.md", [rec("/a.md", i, f"t{i}") for i in range(3)])
    store.replace_source("/a.md", [rec("/a.md", 0, "new")])
    assert store.chunk_count() == 1
    assert store.get_source("/a.md").chunk_count == 1


def test_delete_source_returns_count(store):
    store.replace_source("/a.md", [rec("/a.md", 0, "x")])
    assert store.delete_source("/a.md") == 1
    assert store.delete_source("/a.md") == 0
    assert store.chunk_count() == 0


def test_sql_injection_safe_source_names(store):
    evil = "/o'brien's notes.md"
    store.replace_source(evil, [rec(evil, 0, "x")])
    assert store.get_source(evil) is not None
    assert store.delete_source(evil) == 1


def test_neighbors_window_and_order(store):
    store.replace_source("/a.md", [rec("/a.md", i, f"chunk {i}") for i in range(30)])
    got = store.neighbors("/a.md", 15, before=2, after=2)
    assert [g.chunk_index for g in got] == [13, 14, 15, 16, 17]  # >10 rows must survive default limits
    edge = store.neighbors("/a.md", 0, before=3, after=1)
    assert [g.chunk_index for g in edge] == [0, 1]


def test_all_chunks_full_order(store):
    store.replace_source("/a.md", [rec("/a.md", i, f"chunk {i}") for i in range(25)])
    got = store.all_chunks("/a.md")
    assert [g.chunk_index for g in got] == list(range(25))  # all rows, ordered
    assert store.all_chunks("/missing.md") == []


def test_list_sources_scopes_and_many_rows(store):
    for n in range(15):
        src = f"/docs/api/f{n:02d}.md"
        store.replace_source(src, [rec(src, 0, "x")])
    store.replace_source("/other/z.md", [rec("/other/z.md", 0, "x")])
    store.replace_source("https://x.io/p", [rec("https://x.io/p", 0, "x", source_type="url")])
    assert store.source_count() == 17
    scoped = store.list_sources(scopes=("/docs/api",))
    assert len(scoped) == 15 and all(s.source.startswith("/docs/api") for s in scoped)
    everything = store.list_sources()
    assert len(everything) == 17
    assert everything == sorted(everything, key=lambda s: s.source)


def test_persistence_across_instances(tmp_path):
    s1 = Store(tmp_path / "db", dim=8)
    s1.replace_source("/a.md", [rec("/a.md", 0, "x")])
    s2 = Store(tmp_path / "db", dim=8)  # reopen, no create conflict
    assert s2.chunk_count() == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_store.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/store.py
"""LanceDB persistence: one `chunks` table + BM25 FTS index."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.index import FTS

TABLE = "chunks"
_LIST_LIMIT = 2**31 - 1  # LanceDB scalar queries default to limit 10 — always set explicitly


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    source: str
    source_type: str  # "file" | "data" | "url"
    title: str
    chunk_index: int
    text: str
    vector: list[float]
    file_hash: str
    mtime: float
    ingested_at: str


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    title: str
    chunk_index: int
    score: float
    distance: float | None


@dataclass(frozen=True)
class SourceInfo:
    source: str
    source_type: str
    title: str
    chunk_count: int
    file_hash: str
    mtime: float


def _sql_str(s: str) -> str:
    return s.replace("'", "''")


def _scope_clause(scopes: tuple[str, ...]) -> str | None:
    if not scopes:
        return None
    return " OR ".join(f"source LIKE '{_sql_str(p)}%'" for p in scopes)


_META_COLS = ["source", "source_type", "title", "chunk_index", "file_hash", "mtime"]


class Store:
    def __init__(self, db_path: Path, dim: int):
        db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(db_path))
        self.dim = dim
        if TABLE in [t for t in self._db.table_names()]:
            self._table = self._db.open_table(TABLE)
        else:
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("source_type", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("chunk_index", pa.int32()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    pa.field("file_hash", pa.string()),
                    pa.field("mtime", pa.float64()),
                    pa.field("ingested_at", pa.string()),
                ]
            )
            self._table = self._db.create_table(TABLE, schema=schema)
            self._table.create_index("text", config=FTS())

    def replace_source(self, source: str, records: Sequence[ChunkRecord]) -> None:
        self._table.delete(f"source = '{_sql_str(source)}'")
        if records:
            self._table.add([asdict(r) for r in records])

    def delete_source(self, source: str) -> int:
        clause = f"source = '{_sql_str(source)}'"
        before = len(self._table.search().where(clause).select(["id"]).limit(_LIST_LIMIT).to_list())
        if before:
            self._table.delete(clause)
        return before

    def neighbors(self, source: str, chunk_index: int, before: int = 1, after: int = 1) -> list[SearchResult]:
        lo, hi = max(0, chunk_index - before), chunk_index + after
        clause = f"source = '{_sql_str(source)}' AND chunk_index >= {lo} AND chunk_index <= {hi}"
        rows = self._table.search().where(clause).limit(_LIST_LIMIT).to_list()
        rows.sort(key=lambda r: r["chunk_index"])
        return [
            SearchResult(
                text=r["text"], source=r["source"], title=r["title"],
                chunk_index=r["chunk_index"], score=0.0, distance=None,
            )
            for r in rows
        ]

    def all_chunks(self, source: str) -> list[SearchResult]:
        clause = f"source = '{_sql_str(source)}'"
        rows = self._table.search().where(clause).limit(_LIST_LIMIT).to_list()
        rows.sort(key=lambda r: r["chunk_index"])
        return [
            SearchResult(
                text=r["text"], source=r["source"], title=r["title"],
                chunk_index=r["chunk_index"], score=0.0, distance=None,
            )
            for r in rows
        ]

    def _iter_meta(self, scopes: tuple[str, ...] = ()) -> list[dict]:
        q = self._table.search().select(_META_COLS)
        clause = _scope_clause(scopes)
        if clause:
            q = q.where(clause)
        return q.limit(_LIST_LIMIT).to_list()

    def list_sources(self, scopes: tuple[str, ...] = ()) -> list[SourceInfo]:
        by_source: dict[str, dict] = {}
        counts: dict[str, int] = {}
        for row in self._iter_meta(scopes):
            by_source.setdefault(row["source"], row)
            counts[row["source"]] = counts.get(row["source"], 0) + 1
        return [
            SourceInfo(
                source=src, source_type=row["source_type"], title=row["title"],
                chunk_count=counts[src], file_hash=row["file_hash"], mtime=row["mtime"],
            )
            for src, row in sorted(by_source.items())
        ]

    def get_source(self, source: str) -> SourceInfo | None:
        rows = (
            self._table.search()
            .where(f"source = '{_sql_str(source)}'")
            .select(_META_COLS)
            .limit(_LIST_LIMIT)
            .to_list()
        )
        if not rows:
            return None
        r = rows[0]
        return SourceInfo(
            source=r["source"], source_type=r["source_type"], title=r["title"],
            chunk_count=len(rows), file_hash=r["file_hash"], mtime=r["mtime"],
        )

    def chunk_count(self) -> int:
        return self._table.count_rows()

    def source_count(self) -> int:
        return len({r["source"] for r in self._iter_meta()})
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_store.py -q`
Expected: all pass. If `.select()` on a scalar (no-vector) search errors in this lancedb version, drop `.select(...)` and accept full rows — correctness first, note it in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/store.py tests/test_store.py
git commit -m "feat: LanceDB store with CRUD, neighbors, source listing"
```

---

### Task 8: store.py — hybrid search, filters, grouping

**Files:**
- Modify: `src/minirag_mcp/store.py` (append search methods + module functions)
- Test: `tests/test_store_search.py`

**Interfaces:**
- Consumes: `Store`, `ChunkRecord`, `SearchResult` from Task 7.
- Produces:
  - `relevance_cutoff(distances: Sequence[float], mode: str) -> int` — index to cut the ordered result list at; `len(distances)` when no cut. Gap heuristic: boundaries where `gap > mean(gaps) + std(gaps)`; `similar` cuts at the 1st boundary, `related` at the 2nd; fewer than 3 items → no cut.
  - `Store.search(query_text: str, query_vector: list[float], *, top_k: int = 8, hybrid_weight: float = 0.6, scopes: tuple[str, ...] = (), max_distance: float | None = None, grouping: str | None = None, max_files: int | None = None) -> list[SearchResult]`
  - Semantics: over-fetch `max(top_k * 4, 50)`; `hybrid_weight <= 0` → pure vector search ordering; else hybrid query `.vector(...).text(...)` + `LinearCombinationReranker(weight=1.0 - hybrid_weight)`; distances attached from a parallel vector-only query by `id`; `max_distance` drops rows whose known distance exceeds it (unknown distance = keep); grouping applies only when every kept row has a distance; `max_files` keeps rows from the first N distinct sources; finally truncate to `top_k`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store_search.py
import pytest

from minirag_mcp.store import ChunkRecord, Store, relevance_cutoff


def rec(source, i, text, vec):
    return ChunkRecord(
        id=f"{source}#{i}", source=source, source_type="file", title="T",
        chunk_index=i, text=text, vector=vec, file_hash="h", mtime=1.0,
        ingested_at="2026-08-07T00:00:00+00:00",
    )


def V(x):  # 8-dim vector pointing "x of the way" between axis 0 and axis 1
    return [1.0 - x, x] + [0.0] * 6


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db", dim=8)
    s.replace_source("/auth.md", [
        rec("/auth.md", 0, "OAuth2 token authentication flow", V(0.0)),
        rec("/auth.md", 1, "The API returns ERR_CONNECTION_REFUSED when down", V(0.1)),
    ])
    s.replace_source("/cook.md", [rec("/cook.md", 0, "Borscht recipe with beets", V(1.0))])
    return s


def test_vector_only_ordering(store):
    got = store.search("anything", V(0.05), top_k=3, hybrid_weight=0.0)
    assert got[0].source == "/auth.md"
    assert got[0].distance is not None
    assert [g.source for g in got].count("/cook.md") == 1


def test_keyword_boost_lifts_exact_term(store):
    # Query vector points AT the recipe; only the keyword boost can lift the auth chunk.
    got = store.search("ERR_CONNECTION_REFUSED", V(1.0), top_k=3, hybrid_weight=1.0)
    assert got[0].text.startswith("The API returns ERR_CONNECTION_REFUSED")


def test_scope_filter(store):
    got = store.search("token", V(0.0), top_k=5, hybrid_weight=0.5, scopes=("/auth",))
    assert got and all(g.source == "/auth.md" for g in got)


def test_max_distance_filter(store):
    all_rows = store.search("x", V(0.0), top_k=5, hybrid_weight=0.0)
    far = max(r.distance for r in all_rows)
    got = store.search("x", V(0.0), top_k=5, hybrid_weight=0.0, max_distance=far - 1e-6)
    assert len(got) < len(all_rows)


def test_max_files_limits_distinct_sources(store):
    got = store.search("x", V(0.05), top_k=5, hybrid_weight=0.0, max_files=1)
    assert len({g.source for g in got}) == 1


def test_top_k_truncates(store):
    got = store.search("x", V(0.0), top_k=1, hybrid_weight=0.0)
    assert len(got) == 1


def test_relevance_cutoff():
    assert relevance_cutoff([], "similar") == 0
    assert relevance_cutoff([0.1, 0.11], "similar") == 2  # <3 items: no cut
    d = [0.10, 0.11, 0.12, 0.55, 0.56]  # one big gap after index 2
    assert relevance_cutoff(d, "similar") == 3
    assert relevance_cutoff(d, "related") == 5  # only one boundary => keep all
    d2 = [0.10, 0.11, 0.40, 0.41, 0.80, 0.81]  # two big gaps
    assert relevance_cutoff(d2, "similar") == 2
    assert relevance_cutoff(d2, "related") == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_store_search.py -q`
Expected: FAIL — `relevance_cutoff` not importable.

- [ ] **Step 3: Implement (append to store.py)**

```python
# append to src/minirag_mcp/store.py
import statistics

from lancedb.rerankers import LinearCombinationReranker


def relevance_cutoff(distances: Sequence[float], mode: str) -> int:
    n = len(distances)
    if n < 3:
        return n
    gaps = [distances[i + 1] - distances[i] for i in range(n - 1)]
    mean = statistics.fmean(gaps)
    std = statistics.pstdev(gaps)
    threshold = mean + std
    boundaries = [i + 1 for i, g in enumerate(gaps) if g > threshold and g > 0]
    if not boundaries:
        return n
    if mode == "similar":
        return boundaries[0]
    return boundaries[1] if len(boundaries) >= 2 else n


# --- methods on Store (add inside the class) ---

    def search(
        self,
        query_text: str,
        query_vector: list[float],
        *,
        top_k: int = 8,
        hybrid_weight: float = 0.6,
        scopes: tuple[str, ...] = (),
        max_distance: float | None = None,
        grouping: str | None = None,
        max_files: int | None = None,
    ) -> list[SearchResult]:
        fetch = max(top_k * 4, 50)
        clause = _scope_clause(scopes)

        vq = self._table.search(query_vector).limit(fetch)
        if clause:
            vq = vq.where(clause, prefilter=True)
        vrows = vq.to_list()
        dist_by_id = {r["id"]: r["_distance"] for r in vrows}

        if hybrid_weight <= 0.0:
            ordered = vrows
        else:
            hq = self._table.search(query_type="hybrid").vector(query_vector).text(query_text).limit(fetch)
            if clause:
                hq = hq.where(clause, prefilter=True)
            ordered = hq.rerank(LinearCombinationReranker(weight=1.0 - hybrid_weight)).to_list()

        results: list[SearchResult] = []
        for r in ordered:
            distance = dist_by_id.get(r["id"], r.get("_distance"))
            if max_distance is not None and distance is not None and distance > max_distance:
                continue
            score = r.get("_relevance_score")
            if score is None:
                score = 1.0 / (1.0 + distance) if distance is not None else 0.0
            results.append(
                SearchResult(
                    text=r["text"], source=r["source"], title=r["title"],
                    chunk_index=r["chunk_index"], score=float(score), distance=distance,
                )
            )

        if grouping in ("similar", "related") and results and all(
            r.distance is not None for r in results
        ):
            cut = relevance_cutoff([r.distance for r in results], grouping)
            results = results[:cut]

        if max_files is not None:
            keep: list[SearchResult] = []
            seen: list[str] = []
            for r in results:
                if r.source not in seen:
                    if len(seen) >= max_files:
                        continue
                    seen.append(r.source)
                keep.append(r)
            results = keep

        return results[:top_k]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_store_search.py tests/test_store.py -q`
Expected: all pass. Known trap: `test_keyword_boost_lifts_exact_term` needs the FTS index to cover rows added after index creation — probe-verified to work; if it still flakes, call `self._table.optimize()` after `add` in `replace_source`.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/store.py tests/test_store_search.py
git commit -m "feat: hybrid search with keyword boost, distance/grouping/file filters"
```

---

### Task 9: ingest/parser.py

**Files:**
- Create: `src/minirag_mcp/ingest/__init__.py` (empty), `src/minirag_mcp/ingest/parser.py`, `tests/test_parser.py`

**Interfaces:**
- Produces:
  - `SUPPORTED_EXTENSIONS: frozenset[str]` = `{".md", ".markdown", ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".csv", ".epub", ".ipynb"}`
  - `class ParserError(Exception)`
  - `@dataclass(frozen=True) ParsedDoc(markdown: str, title: str)`
  - `extract_title(markdown: str, explicit: str | None, fallback: str) -> str` — explicit > first `# H1` > fallback
  - `parse_file(path: Path) -> ParsedDoc`, `parse_html(html: str, title: str | None = None) -> ParsedDoc`, `parse_url(url: str) -> ParsedDoc` (no scheme check here — that is `security.check_url_scheme`, called by the pipeline). markitdown exceptions are wrapped in `ParserError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parser.py
import pytest

from minirag_mcp.ingest.parser import (
    SUPPORTED_EXTENSIONS,
    ParsedDoc,
    ParserError,
    extract_title,
    parse_file,
    parse_html,
)


def test_supported_extensions_frozen():
    assert ".md" in SUPPORTED_EXTENSIONS and ".ipynb" in SUPPORTED_EXTENSIONS
    assert ".py" not in SUPPORTED_EXTENSIONS


def test_extract_title_precedence():
    md = "# Real Title\n\nBody"
    assert extract_title(md, "Explicit", "fallback") == "Explicit"
    assert extract_title(md, None, "fallback") == "Real Title"
    assert extract_title("no heading here", None, "fallback") == "fallback"


def test_parse_markdown_file(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# My Doc\n\nHello **world**.\n", encoding="utf-8")
    doc = parse_file(f)
    assert isinstance(doc, ParsedDoc)
    assert doc.title == "My Doc"  # markitdown returns title=None for md; H1 fallback
    assert "Hello" in doc.markdown


def test_parse_txt_file_title_falls_back_to_stem(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("just plain text without headings", encoding="utf-8")
    doc = parse_file(f)
    assert doc.title == "notes"


def test_parse_html_string_gets_title_tag():
    doc = parse_html("<html><head><title>Page T</title></head><body><p>Body text</p></body></html>")
    assert doc.title == "Page T"
    assert "Body text" in doc.markdown


def test_parse_html_explicit_title_wins():
    doc = parse_html("<html><body><p>x</p></body></html>", title="Given")
    assert doc.title == "Given"


def test_parse_file_failure_wrapped(tmp_path):
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"not a real pdf")
    with pytest.raises(ParserError):
        parse_file(f)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_parser.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/ingest/parser.py
"""markitdown wrapper: files, HTML strings, URLs -> Markdown + title."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from markitdown import MarkItDown, MarkItDownException

SUPPORTED_EXTENSIONS = frozenset(
    {".md", ".markdown", ".txt", ".pdf", ".docx", ".pptx", ".xlsx",
     ".html", ".htm", ".csv", ".epub", ".ipynb"}
)

_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)

_converter: MarkItDown | None = None


class ParserError(Exception):
    pass


@dataclass(frozen=True)
class ParsedDoc:
    markdown: str
    title: str


def _md() -> MarkItDown:
    global _converter
    if _converter is None:
        _converter = MarkItDown(enable_plugins=False)
    return _converter


def extract_title(markdown: str, explicit: str | None, fallback: str) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    m = _H1_RE.search(markdown)
    if m:
        return m.group(1)
    return fallback


def parse_file(path: Path) -> ParsedDoc:
    try:
        result = _md().convert(str(path))
    except (MarkItDownException, Exception) as e:  # markitdown may raise lib-specific errors too
        raise ParserError(f"Failed to convert {path}: {e}") from e
    markdown = result.markdown or ""
    return ParsedDoc(markdown=markdown, title=extract_title(markdown, result.title, path.stem))


def parse_html(html: str, title: str | None = None) -> ParsedDoc:
    try:
        result = _md().convert_stream(io.BytesIO(html.encode("utf-8")), file_extension=".html")
    except (MarkItDownException, Exception) as e:
        raise ParserError(f"Failed to convert HTML: {e}") from e
    markdown = result.markdown or ""
    return ParsedDoc(
        markdown=markdown, title=extract_title(markdown, title or result.title, "Untitled")
    )


def parse_url(url: str) -> ParsedDoc:
    try:
        result = _md().convert_url(url)
    except (MarkItDownException, Exception) as e:
        raise ParserError(f"Failed to fetch/convert {url}: {e}") from e
    markdown = result.markdown or ""
    return ParsedDoc(markdown=markdown, title=extract_title(markdown, result.title, url))
```

Note: the broad `except (MarkItDownException, Exception)` is deliberate — pdfminer & friends raise their own exception types; anything failing conversion becomes `ParserError`. Keep the tuple (documents intent) even though `Exception` subsumes it.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_parser.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/ingest tests/test_parser.py
git commit -m "feat: markitdown parser wrapper with title extraction"
```

---

### Task 10: ingest/pipeline.py

**Files:**
- Create: `src/minirag_mcp/ingest/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Store`/`ChunkRecord` (Task 7), `Embedder`-shaped object (Task 6 / `FakeEmbedder`), `split_markdown` (Task 4), `merge_blocks` (Task 5), parser functions (Task 9), `check_url_scheme` (Task 3), `Config` (Task 2).
- Produces:
  - `MAX_CHUNK_CHARS = 1500`
  - `class UnsupportedFormatError(Exception)`, `class FileTooLargeError(Exception)`, `class EmptyDocumentError(Exception)`
  - `@dataclass(frozen=True) IngestResult(source: str, chunk_count: int, title: str)`
  - `class Pipeline` with `__init__(store, embedder, config)` and:
    - `ingest_file(path: Path) -> IngestResult` — path is already root-validated by the caller; checks extension + size; `source = str(path)`, `source_type="file"`, sha256 + mtime recorded
    - `ingest_data(data: str, source: str, fmt: str = "text", title: str | None = None) -> IngestResult` — fmt in `{"text","markdown","html"}` else `UnsupportedFormatError`; `source_type="data"`
    - `ingest_url(url: str, source: str | None = None, title: str | None = None) -> IngestResult` — scheme check, `source_type="url"`, source defaults to url
  - `file_sha256(path: Path) -> str` helper (also used by scanner in Task 11).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
import pytest

from minirag_mcp.config import load_config
from minirag_mcp.ingest.pipeline import (
    EmptyDocumentError,
    FileTooLargeError,
    Pipeline,
    UnsupportedFormatError,
    file_sha256,
)
from minirag_mcp.security import SecurityError
from minirag_mcp.store import Store


@pytest.fixture
def pipe(tmp_path, fake_embedder):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    store = Store(tmp_path / ".minirag" / "lancedb", dim=fake_embedder.dim)
    return Pipeline(store, fake_embedder, cfg), store, tmp_path


def test_ingest_file_roundtrip(pipe):
    p, store, root = pipe
    f = root / "doc.md"
    f.write_text("# Title\n\n" + "Sentence about topic. " * 30, encoding="utf-8")
    res = p.ingest_file(f)
    assert res.source == str(f) and res.title == "Title" and res.chunk_count >= 1
    info = store.get_source(str(f))
    assert info.source_type == "file"
    assert info.file_hash == file_sha256(f)
    assert info.mtime == f.stat().st_mtime


def test_reingest_replaces(pipe):
    p, store, root = pipe
    f = root / "doc.md"
    f.write_text("# A\n\n" + "words " * 200, encoding="utf-8")
    p.ingest_file(f)
    first = store.get_source(str(f)).chunk_count
    f.write_text("# A\n\nshort now", encoding="utf-8")
    res = p.ingest_file(f)
    assert res.chunk_count <= first
    assert store.get_source(str(f)).chunk_count == res.chunk_count


def test_unsupported_extension(pipe):
    p, _, root = pipe
    f = root / "script.py"
    f.write_text("print('hi')")
    with pytest.raises(UnsupportedFormatError):
        p.ingest_file(f)


def test_file_too_large(pipe, tmp_path, fake_embedder):
    from minirag_mcp.config import load_config

    cfg = load_config({"BASE_DIR": str(tmp_path), "MAX_FILE_SIZE": "10"}, cwd=tmp_path)
    store = Store(tmp_path / "db2", dim=fake_embedder.dim)
    p = Pipeline(store, fake_embedder, cfg)
    f = tmp_path / "big.md"
    f.write_text("more than ten bytes of content")
    with pytest.raises(FileTooLargeError):
        p.ingest_file(f)


def test_empty_document_rejected(pipe):
    p, _, root = pipe
    f = root / "empty.md"
    f.write_text("   \n\n  ")
    with pytest.raises(EmptyDocumentError):
        p.ingest_file(f)


def test_ingest_data_text_and_markdown(pipe):
    p, store, _ = pipe
    res = p.ingest_data("Plain text body long enough to keep.", source="note-1", fmt="text", title="Note")
    assert res.title == "Note" and store.get_source("note-1").source_type == "data"
    res2 = p.ingest_data("# MD Title\n\nBody here.", source="note-2", fmt="markdown")
    assert res2.title == "MD Title"


def test_ingest_data_html(pipe):
    p, store, _ = pipe
    res = p.ingest_data(
        "<html><head><title>H</title></head><body><p>Hypertext body.</p></body></html>",
        source="page-1", fmt="html",
    )
    assert res.title == "H"
    assert "Hypertext body" in store.neighbors("page-1", 0, 0, 0)[0].text


def test_ingest_data_bad_format(pipe):
    p, _, _ = pipe
    with pytest.raises(UnsupportedFormatError):
        p.ingest_data("x", source="s", fmt="pdf")


def test_ingest_url_scheme_rejected(pipe):
    p, _, _ = pipe
    with pytest.raises(SecurityError):
        p.ingest_url("file:///etc/passwd")


def test_ingest_url_mocked(pipe, monkeypatch):
    import minirag_mcp.ingest.pipeline as mod
    from minirag_mcp.ingest.parser import ParsedDoc

    p, store, _ = pipe
    monkeypatch.setattr(
        mod, "parse_url", lambda url: ParsedDoc(markdown="# Remote\n\nFetched body.", title="Remote")
    )
    res = p.ingest_url("https://example.com/docs")
    assert res.source == "https://example.com/docs"
    assert store.get_source(res.source).source_type == "url"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/ingest/pipeline.py
"""Ingestion pipeline: parse -> chunk -> embed -> store (replace by source)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from minirag_mcp.chunker.semantic import merge_blocks
from minirag_mcp.chunker.structural import split_markdown
from minirag_mcp.config import Config
from minirag_mcp.ingest.parser import (
    SUPPORTED_EXTENSIONS,
    parse_file,
    parse_html,
)
from minirag_mcp.ingest import parser as _parser
from minirag_mcp.security import check_url_scheme
from minirag_mcp.store import ChunkRecord, Store

MAX_CHUNK_CHARS = 1500
DATA_FORMATS = ("text", "markdown", "html")


class UnsupportedFormatError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


class EmptyDocumentError(Exception):
    pass


@dataclass(frozen=True)
class IngestResult:
    source: str
    chunk_count: int
    title: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# module-level alias so tests can monkeypatch minirag_mcp.ingest.pipeline.parse_url
def parse_url(url: str):
    return _parser.parse_url(url)


class Pipeline:
    def __init__(self, store: Store, embedder, config: Config):
        self.store = store
        self.embedder = embedder
        self.config = config

    def _chunk_and_store(
        self, markdown: str, *, source: str, source_type: str, title: str,
        file_hash: str = "", mtime: float = 0.0,
    ) -> IngestResult:
        blocks = split_markdown(markdown, max_chars=MAX_CHUNK_CHARS)
        texts = merge_blocks(
            blocks, self.embedder.embed_documents,
            max_chars=MAX_CHUNK_CHARS, min_length=self.config.chunk_min_length,
        )
        if not texts:
            raise EmptyDocumentError(f"No text content extracted from {source}")
        vectors = self.embedder.embed_documents(texts)
        now = datetime.now(timezone.utc).isoformat()
        records = [
            ChunkRecord(
                id=f"{source}#{i}", source=source, source_type=source_type, title=title,
                chunk_index=i, text=text, vector=vec, file_hash=file_hash, mtime=mtime,
                ingested_at=now,
            )
            for i, (text, vec) in enumerate(zip(texts, vectors))
        ]
        self.store.replace_source(source, records)
        return IngestResult(source=source, chunk_count=len(records), title=title)

    def ingest_file(self, path: Path) -> IngestResult:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file extension {path.suffix!r}; supported: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS))
            )
        size = path.stat().st_size
        if size > self.config.max_file_size:
            raise FileTooLargeError(
                f"{path} is {size} bytes; MAX_FILE_SIZE is {self.config.max_file_size}"
            )
        doc = parse_file(path)
        return self._chunk_and_store(
            doc.markdown, source=str(path), source_type="file", title=doc.title,
            file_hash=file_sha256(path), mtime=path.stat().st_mtime,
        )

    def ingest_data(
        self, data: str, source: str, fmt: str = "text", title: str | None = None
    ) -> IngestResult:
        if fmt not in DATA_FORMATS:
            raise UnsupportedFormatError(f"format must be one of {DATA_FORMATS}, got {fmt!r}")
        if fmt == "html":
            doc = parse_html(data, title=title)
            markdown, final_title = doc.markdown, doc.title
        else:
            markdown = data
            from minirag_mcp.ingest.parser import extract_title

            final_title = extract_title(markdown, title, source)
        return self._chunk_and_store(markdown, source=source, source_type="data", title=final_title)

    def ingest_url(
        self, url: str, source: str | None = None, title: str | None = None
    ) -> IngestResult:
        check_url_scheme(url)
        doc = parse_url(url)
        final_title = title.strip() if title and title.strip() else doc.title
        return self._chunk_and_store(
            doc.markdown, source=source or url, source_type="url", title=final_title
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/ingest/pipeline.py tests/test_pipeline.py
git commit -m "feat: ingest pipeline for files, data, and URLs"
```

---

### Task 11: ingest/scanner.py

**Files:**
- Create: `src/minirag_mcp/ingest/scanner.py`, `tests/test_scanner.py`

**Interfaces:**
- Consumes: `SUPPORTED_EXTENSIONS` (Task 9), `file_sha256` (Task 10), `SourceInfo` (Task 7).
- Produces:
  - `SKIP_DIRS = frozenset({"node_modules", "__pycache__", ".venv", "venv"})`
  - `@dataclass(frozen=True) ScanEntry(path: Path, size: int, mtime: float)`
  - `scan_roots(roots: Sequence[Path]) -> list[ScanEntry]` — recursive, whitelist extensions, prunes hidden dirs (dot-prefixed) and `SKIP_DIRS`, skips hidden files, sorted by path
  - `@dataclass(frozen=True) SyncDiff(to_ingest: list[Path], to_delete: list[str], unchanged: list[Path], oversized: list[Path])`
  - `compute_diff(entries: list[ScanEntry], indexed: list[SourceInfo], *, max_file_size: int, scope: Path | None = None) -> SyncDiff` — mtime fast-path (equal mtime ⇒ unchanged, no hashing), else sha256 compare; `to_delete` = indexed `file`-type sources under scope that are not on disk; `data`/`url` sources never appear.
  - `@dataclass(frozen=True) FileState(source: str, source_type: str, title: str, state: str, chunk_count: int)` — `state` ∈ `ingested | not_ingested | stale`
  - `compute_states(entries: list[ScanEntry], indexed: list[SourceInfo]) -> list[FileState]` — one entry per file on disk (`ingested` when hash or mtime matches, `stale` when both differ, `not_ingested` when absent from the index; `title=""`/`chunk_count=0` for not-ingested), then every indexed `data`/`url` source appended as `ingested`, sorted by source.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scanner.py
from pathlib import Path

from minirag_mcp.ingest.pipeline import file_sha256
from minirag_mcp.ingest.scanner import ScanEntry, compute_diff, scan_roots
from minirag_mcp.store import SourceInfo


def info(source, *, source_type="file", file_hash="", mtime=0.0):
    return SourceInfo(source=source, source_type=source_type, title="T",
                      chunk_count=1, file_hash=file_hash, mtime=mtime)


def make_tree(root: Path):
    (root / "sub" / "deep").mkdir(parents=True)
    (root / "a.md").write_text("a")
    (root / "sub" / "b.txt").write_text("b")
    (root / "sub" / "deep" / "c.pdf").write_bytes(b"%PDF-fake")
    (root / "skip.py").write_text("nope")            # not whitelisted
    (root / ".hidden.md").write_text("hidden file")   # hidden file
    (root / ".git").mkdir()
    (root / ".git" / "x.md").write_text("in hidden dir")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "y.md").write_text("in skip dir")


def test_scan_recursive_whitelist_and_skips(tmp_path):
    make_tree(tmp_path)
    entries = scan_roots([tmp_path])
    names = [e.path.relative_to(tmp_path).as_posix() for e in entries]
    assert names == ["a.md", "sub/b.txt", "sub/deep/c.pdf"]
    assert all(isinstance(e, ScanEntry) and e.size > 0 for e in entries)


def test_diff_new_changed_unchanged_deleted(tmp_path):
    make_tree(tmp_path)
    entries = scan_roots([tmp_path])
    a = tmp_path / "a.md"
    b = tmp_path / "sub" / "b.txt"
    indexed = [
        info(str(a), file_hash=file_sha256(a), mtime=a.stat().st_mtime),  # unchanged (mtime match)
        info(str(b), file_hash="stale-hash", mtime=-1.0),                  # changed (hash differs)
        info(str(tmp_path / "gone.md")),                                    # deleted from disk
        info("https://x.io/p", source_type="url"),                          # never deleted by sync
        info("note-1", source_type="data"),                                 # never deleted by sync
    ]
    diff = compute_diff(entries, indexed, max_file_size=10**9)
    assert diff.unchanged == [a]
    assert b in diff.to_ingest and (tmp_path / "sub" / "deep" / "c.pdf") in diff.to_ingest
    assert diff.to_delete == [str(tmp_path / "gone.md")]


def test_diff_mtime_differs_but_hash_same_is_unchanged(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("stable content")
    entries = [ScanEntry(path=f, size=f.stat().st_size, mtime=999.0)]
    indexed = [info(str(f), file_hash=file_sha256(f), mtime=1.0)]
    diff = compute_diff(entries, indexed, max_file_size=10**9)
    assert diff.unchanged == [f] and diff.to_ingest == []


def test_diff_oversized(tmp_path):
    f = tmp_path / "big.md"
    f.write_text("0123456789ABCDEF")
    entries = scan_roots([tmp_path])
    diff = compute_diff(entries, [], max_file_size=4)
    assert diff.oversized == [f] and diff.to_ingest == []


def test_diff_scope_limits_both_sides(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    fin = tmp_path / "in" / "f.md"
    fin.write_text("x")
    entries = scan_roots([tmp_path])
    indexed = [info(str(tmp_path / "out" / "gone.md"))]
    diff = compute_diff(entries, indexed, max_file_size=10**9, scope=tmp_path / "in")
    assert diff.to_ingest == [fin]
    assert diff.to_delete == []  # out/gone.md is outside scope: not touched


def test_compute_states(tmp_path):
    from minirag_mcp.ingest.scanner import compute_states

    a = tmp_path / "a.md"
    a.write_text("ingested content")
    b = tmp_path / "b.md"
    b.write_text("changed on disk")
    c = tmp_path / "c.md"
    c.write_text("never ingested")
    entries = scan_roots([tmp_path])
    indexed = [
        info(str(a), file_hash=file_sha256(a), mtime=a.stat().st_mtime),
        info(str(b), file_hash="old-hash", mtime=-1.0),
        info("note-1", source_type="data"),
        info("https://x.io/p", source_type="url"),
    ]
    states = {s.source: s.state for s in compute_states(entries, indexed)}
    assert states[str(a)] == "ingested"
    assert states[str(b)] == "stale"
    assert states[str(c)] == "not_ingested"
    assert states["note-1"] == "ingested" and states["https://x.io/p"] == "ingested"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_scanner.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/ingest/scanner.py
"""Recursive document-root scanning and sync diff computation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from minirag_mcp.ingest.parser import SUPPORTED_EXTENSIONS
from minirag_mcp.ingest.pipeline import file_sha256
from minirag_mcp.store import SourceInfo

SKIP_DIRS = frozenset({"node_modules", "__pycache__", ".venv", "venv"})


@dataclass(frozen=True)
class ScanEntry:
    path: Path
    size: int
    mtime: float


@dataclass(frozen=True)
class SyncDiff:
    to_ingest: list[Path]
    to_delete: list[str]
    unchanged: list[Path]
    oversized: list[Path]


def scan_roots(roots: Sequence[Path]) -> list[ScanEntry]:
    entries: list[ScanEntry] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS
            )
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                p = Path(dirpath) / name
                if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                st = p.stat()
                entries.append(ScanEntry(path=p, size=st.st_size, mtime=st.st_mtime))
    entries.sort(key=lambda e: e.path)
    return entries


def _in_scope(path_str: str, scope: Path | None) -> bool:
    if scope is None:
        return True
    p = Path(path_str)
    return p == scope or scope in p.parents


def compute_diff(
    entries: list[ScanEntry],
    indexed: list[SourceInfo],
    *,
    max_file_size: int,
    scope: Path | None = None,
) -> SyncDiff:
    indexed_files = {
        s.source: s for s in indexed if s.source_type == "file" and _in_scope(s.source, scope)
    }
    to_ingest: list[Path] = []
    unchanged: list[Path] = []
    oversized: list[Path] = []
    seen: set[str] = set()

    for e in entries:
        if not _in_scope(str(e.path), scope):
            continue
        seen.add(str(e.path))
        if e.size > max_file_size:
            oversized.append(e.path)
            continue
        prior = indexed_files.get(str(e.path))
        if prior is None:
            to_ingest.append(e.path)
        elif prior.mtime == e.mtime or prior.file_hash == file_sha256(e.path):
            unchanged.append(e.path)
        else:
            to_ingest.append(e.path)

    to_delete = [src for src in indexed_files if src not in seen]
    return SyncDiff(
        to_ingest=to_ingest, to_delete=sorted(to_delete),
        unchanged=unchanged, oversized=oversized,
    )


@dataclass(frozen=True)
class FileState:
    source: str
    source_type: str
    title: str
    state: str  # "ingested" | "not_ingested" | "stale"
    chunk_count: int


def compute_states(entries: list[ScanEntry], indexed: list[SourceInfo]) -> list[FileState]:
    by_source = {s.source: s for s in indexed}
    states: list[FileState] = []
    for e in entries:
        prior = by_source.get(str(e.path))
        if prior is None:
            states.append(FileState(str(e.path), "file", "", "not_ingested", 0))
        elif prior.mtime == e.mtime or prior.file_hash == file_sha256(e.path):
            states.append(FileState(str(e.path), "file", prior.title, "ingested", prior.chunk_count))
        else:
            states.append(FileState(str(e.path), "file", prior.title, "stale", prior.chunk_count))
    for s in indexed:
        if s.source_type in ("data", "url"):
            states.append(FileState(s.source, s.source_type, s.title, "ingested", s.chunk_count))
    states.sort(key=lambda s: s.source)
    return states
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_scanner.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/ingest/scanner.py tests/test_scanner.py
git commit -m "feat: recursive scanner with sha256 sync diff"
```

---

### Task 12: sync.py

**Files:**
- Create: `src/minirag_mcp/sync.py`, `tests/test_sync.py`

**Interfaces:**
- Consumes: `Pipeline` (Task 10), `Store` (Task 7), `scan_roots`/`compute_diff` (Task 11), `Config` (Task 2).
- Produces:
  - `class SyncBusyError(Exception)`
  - `@dataclass SyncJob(job_id: str, state: str, started_at: str, finished_at: str | None, counts: dict[str, int], errors: list[dict], error: str | None)` — counts keys: `scanned, ingested, skipped, deleted, failed`; states `pending|running|succeeded|failed`; `to_dict() -> dict` with camelCase keys `jobId, state, startedAt, finishedAt, counts, errors, error`
  - `run_sync(pipeline: Pipeline, store: Store, roots: Sequence[Path], max_file_size: int, scope: Path | None = None, on_event: Callable[[str], None] | None = None) -> tuple[dict[str, int], list[dict]]` — synchronous engine shared by the MCP thread and the CLI; per-file failures land in the errors list, never abort
  - `class SyncManager` with `__init__(pipeline, store, config)`, `start(scope: Path | None = None) -> str` (raises `SyncBusyError` when a job is running; only the latest job is retained), `status(job_id: str) -> SyncJob` (raises `KeyError` for unknown/discarded ids), `wait(timeout: float = 30.0) -> None` (test helper — joins the worker thread)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sync.py
import pytest

from minirag_mcp.config import load_config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.store import Store
from minirag_mcp.sync import SyncBusyError, SyncManager, run_sync


@pytest.fixture
def env(tmp_path, fake_embedder):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    store = Store(tmp_path / ".minirag" / "lancedb", dim=fake_embedder.dim)
    pipeline = Pipeline(store, fake_embedder, cfg)
    return cfg, store, pipeline, tmp_path


def seed(root):
    (root / "sub").mkdir(exist_ok=True)
    (root / "a.md").write_text("# A\n\nAlpha body text for the corpus.")
    (root / "sub" / "b.md").write_text("# B\n\nBeta body text for the corpus.")


def test_run_sync_ingests_and_deletes(env):
    cfg, store, pipeline, root = env
    seed(root)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 2 and counts["scanned"] == 2 and errors == []
    (root / "a.md").unlink()
    counts, _ = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["deleted"] == 1 and counts["skipped"] == 1
    assert store.get_source(str(root / "a.md")) is None


def test_run_sync_collects_per_file_errors(env, monkeypatch):
    cfg, store, pipeline, root = env
    seed(root)

    real = pipeline.ingest_file

    def flaky(path):
        if path.name == "b.md":
            raise RuntimeError("boom")
        return real(path)

    monkeypatch.setattr(pipeline, "ingest_file", flaky)
    counts, errors = run_sync(pipeline, store, cfg.roots, cfg.max_file_size)
    assert counts["ingested"] == 1 and counts["failed"] == 1
    assert errors and "boom" in errors[0]["error"]


def test_manager_lifecycle(env):
    cfg, store, pipeline, root = env
    seed(root)
    mgr = SyncManager(pipeline, store, cfg)
    job_id = mgr.start()
    mgr.wait()
    job = mgr.status(job_id)
    assert job.state == "succeeded"
    assert job.counts["ingested"] == 2
    d = job.to_dict()
    assert d["jobId"] == job_id and d["startedAt"] and d["finishedAt"]


def test_manager_rejects_concurrent_and_forgets_old(env):
    import threading

    cfg, store, pipeline, root = env
    seed(root)
    gate = threading.Event()
    real = pipeline.ingest_file

    def slow(path):
        gate.wait(5)
        return real(path)

    pipeline.ingest_file = slow
    mgr = SyncManager(pipeline, store, cfg)
    first = mgr.start()
    with pytest.raises(SyncBusyError):
        mgr.start()
    gate.set()
    mgr.wait()
    second = mgr.start()
    mgr.wait()
    with pytest.raises(KeyError):
        mgr.status(first)  # only the latest job is retained
    assert mgr.status(second).state == "succeeded"


def test_unknown_job_id(env):
    cfg, store, pipeline, _ = env
    mgr = SyncManager(pipeline, store, cfg)
    with pytest.raises(KeyError):
        mgr.status("nope")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sync.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/sync.py
"""Sync engine + single-job background manager."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from minirag_mcp.config import Config
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.ingest.scanner import compute_diff, scan_roots
from minirag_mcp.store import Store


class SyncBusyError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SyncJob:
    job_id: str
    state: str = "pending"  # pending | running | succeeded | failed
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    counts: dict[str, int] = field(
        default_factory=lambda: {"scanned": 0, "ingested": 0, "skipped": 0, "deleted": 0, "failed": 0}
    )
    errors: list[dict] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "jobId": self.job_id,
            "state": self.state,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "counts": dict(self.counts),
            "errors": list(self.errors),
            "error": self.error,
        }


def run_sync(
    pipeline: Pipeline,
    store: Store,
    roots: Sequence[Path],
    max_file_size: int,
    scope: Path | None = None,
    on_event: Callable[[str], None] | None = None,
) -> tuple[dict[str, int], list[dict]]:
    def emit(msg: str) -> None:
        if on_event:
            on_event(msg)

    entries = scan_roots(list(roots))
    diff = compute_diff(entries, store.list_sources(), max_file_size=max_file_size, scope=scope)
    counts = {
        "scanned": len(diff.to_ingest) + len(diff.unchanged) + len(diff.oversized),
        "ingested": 0,
        "skipped": len(diff.unchanged),
        "deleted": 0,
        "failed": 0,
    }
    errors: list[dict] = []

    for path in diff.oversized:
        counts["failed"] += 1
        errors.append({"source": str(path), "error": f"exceeds MAX_FILE_SIZE ({max_file_size})"})

    for path in diff.to_ingest:
        try:
            pipeline.ingest_file(path)
            counts["ingested"] += 1
            emit(f"ingested {path}")
        except Exception as e:  # per-file failures never abort the job
            counts["failed"] += 1
            errors.append({"source": str(path), "error": str(e)})
            emit(f"failed {path}: {e}")

    for source in diff.to_delete:
        store.delete_source(source)
        counts["deleted"] += 1
        emit(f"deleted {source}")

    return counts, errors


class SyncManager:
    def __init__(self, pipeline: Pipeline, store: Store, config: Config):
        self._pipeline = pipeline
        self._store = store
        self._config = config
        self._lock = threading.Lock()
        self._job: SyncJob | None = None
        self._thread: threading.Thread | None = None

    def start(self, scope: Path | None = None) -> str:
        with self._lock:
            if self._job is not None and self._job.state in ("pending", "running"):
                raise SyncBusyError("A sync job is already running")
            job = SyncJob(job_id=uuid.uuid4().hex)
            self._job = job  # newer job replaces any finished record

        def work() -> None:
            job.state = "running"
            try:
                counts, errors = run_sync(
                    self._pipeline, self._store, self._config.roots,
                    self._config.max_file_size, scope=scope,
                )
                job.counts, job.errors = counts, errors
                job.state = "succeeded"
            except Exception as e:  # catastrophic failure (scan error, DB down, ...)
                job.error = str(e)
                job.state = "failed"
            finally:
                job.finished_at = _now()

        self._thread = threading.Thread(target=work, name="minirag-sync", daemon=True)
        self._thread.start()
        return job.job_id

    def status(self, job_id: str) -> SyncJob:
        job = self._job
        if job is None or job.job_id != job_id:
            raise KeyError(f"Unknown sync job {job_id!r} (only the latest job is retained)")
        return job

    def wait(self, timeout: float = 30.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_sync.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/minirag_mcp/sync.py tests/test_sync.py
git commit -m "feat: sync engine with single background job manager"
```

---

### Task 13: server.py + __main__.py — 11 MCP tools

**Files:**
- Create: `src/minirag_mcp/server.py`, `src/minirag_mcp/__main__.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `create_app(config: Config | None, *, config_error: str | None = None, embedder=None) -> FastMCP` — builds the FastMCP app; `embedder=None` means a real `Embedder(config.model_name, config.cache_dir)`; heavy objects (store/embedder/pipeline/sync manager) are built lazily on first tool call so the server starts instantly and `status` works without touching the model
  - `run_server() -> None` — loads config from `os.environ` (capturing `ConfigError` into degraded mode) and calls `mcp.run()` (stdio)
  - `main() -> None` in `__main__.py`: `sys.argv[1:]` empty → `run_server()`, else CLI app (Task 14 wires it; until then `main` imports lazily so this task stays testable)
  - Tool names and parameter names (parity + extras): `sync_start(path=None)`, `sync_status(jobId)`, `ingest_file(filePath)`, `ingest_data(data, source, format="text", title=None)`, `ingest_url(url, source=None, title=None)`, `query_documents(query, topK=8, scope=None)` → `{results, sources}` where `sources` = distinct matched sources in rank order `{source, title, hits}`, `read_chunk_neighbors(chunkIndex, filePath=None, source=None, before=1, after=1)`, `read_file(filePath=None, source=None)` → `{source, sourceType, title, chunkCount, text}` (chunks joined with a blank line), `list_files(scope=None)` → files on disk with `state` `ingested|not_ingested|stale` plus data/url sources (uses `scan_roots` + `compute_states`), `delete_file(filePath=None, source=None)`, `status()`
  - All domain exceptions (`SecurityError`, `ConfigError`, pipeline errors, `ParserError`, `SyncBusyError`, `KeyError`, `UnknownModelError`, `EmptyDocumentError`, ...) surface as `ToolError` with the original message.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_server.py
import pytest
from fastmcp import Client

from minirag_mcp.config import load_config
from minirag_mcp.server import create_app

# pytest-asyncio runs in auto mode (see pyproject) — bare `async def` tests are collected as-is.


@pytest.fixture
def app(tmp_path, fake_embedder):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "auth.md").write_text(
        "# Auth Guide\n\nOAuth2 token authentication flow explained at length here."
    )
    (root / "sub").mkdir()
    (root / "sub" / "err.md").write_text(
        "# Errors\n\nThe API returns ERR_CONNECTION_REFUSED when the backend is down."
    )
    cfg = load_config({"BASE_DIR": str(root)}, cwd=root)
    return create_app(cfg, embedder=fake_embedder), root


async def test_all_eleven_tools_listed(app):
    mcp, _ = app
    async with Client(mcp) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == {
        "sync_start", "sync_status", "ingest_file", "ingest_data", "ingest_url",
        "query_documents", "read_chunk_neighbors", "read_file", "list_files",
        "delete_file", "status",
    }


async def test_sync_then_query_then_neighbors(app):
    mcp, root = app
    async with Client(mcp) as c:
        job = (await c.call_tool("sync_start", {})).data
        for _ in range(100):
            st = (await c.call_tool("sync_status", {"jobId": job["jobId"]})).data
            if st["state"] in ("succeeded", "failed"):
                break
        assert st["state"] == "succeeded" and st["counts"]["ingested"] == 2

        res = (await c.call_tool("query_documents", {"query": "ERR_CONNECTION_REFUSED"})).data
        assert res["results"], "expected at least one search result"
        top = res["results"][0]
        assert {"text", "source", "title", "chunkIndex", "score"} <= set(top)
        assert res["sources"], "expected aggregated sources"
        assert {"source", "title", "hits"} <= set(res["sources"][0])
        assert res["sources"][0]["source"] == top["source"]  # rank order preserved

        nb = (await c.call_tool(
            "read_chunk_neighbors",
            {"filePath": top["source"], "chunkIndex": top["chunkIndex"]},
        )).data
        assert nb["chunks"]


async def test_ingest_file_read_list_delete(app):
    mcp, root = app
    f = root / "new.md"
    f.write_text("# New\n\nFresh content body that is long enough to index.")
    async with Client(mcp) as c:
        r = (await c.call_tool("ingest_file", {"filePath": str(f)})).data
        assert r["source"] == str(f) and r["chunkCount"] >= 1

        full = (await c.call_tool("read_file", {"filePath": str(f)})).data
        assert full["source"] == str(f) and full["sourceType"] == "file"
        assert "Fresh content body" in full["text"]
        assert full["chunkCount"] == r["chunkCount"]

        listed = (await c.call_tool("list_files", {})).data
        by_source = {x["source"]: x for x in listed["files"]}
        assert by_source[str(f)]["state"] == "ingested"
        # auth.md/err.md exist on disk but were not ingested in this test's client session
        assert any(x["state"] == "not_ingested" for x in listed["files"])

        d = (await c.call_tool("delete_file", {"filePath": str(f)})).data
        assert d["deletedChunks"] >= 1
        with pytest.raises(Exception, match="not"):
            await c.call_tool("read_file", {"filePath": str(f)})


async def test_ingest_file_outside_root_is_tool_error(app):
    mcp, _ = app
    async with Client(mcp) as c:
        with pytest.raises(Exception, match="outside"):
            await c.call_tool("ingest_file", {"filePath": "/etc/passwd"})


async def test_ingest_data_and_url(app, monkeypatch):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url: ParsedDoc("# R\n\nRemote body.", "R"))
    mcp, _ = app
    async with Client(mcp) as c:
        r = (await c.call_tool(
            "ingest_data",
            {"data": "# Note\n\nSaved note body.", "source": "note-1", "format": "markdown"},
        )).data
        assert r["source"] == "note-1"
        r2 = (await c.call_tool("ingest_url", {"url": "https://example.com/x"})).data
        assert r2["source"] == "https://example.com/x"
        with pytest.raises(Exception, match="http"):
            await c.call_tool("ingest_url", {"url": "file:///etc/passwd"})


async def test_status_reports_counts(app):
    mcp, root = app
    async with Client(mcp) as c:
        st = (await c.call_tool("status", {})).data
        assert st["roots"] == [str(root)]
        assert "chunkCount" in st and "model" in st


async def test_degraded_mode_status_alive_others_fail(tmp_path, fake_embedder):
    mcp = create_app(None, config_error="BASE_DIRS must be a JSON array", embedder=fake_embedder)
    async with Client(mcp) as c:
        st = (await c.call_tool("status", {})).data
        assert "BASE_DIRS" in st["configError"]
        with pytest.raises(Exception, match="BASE_DIRS"):
            await c.call_tool("list_files", {})
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_server.py -q`
Expected: FAIL — module not found. (pytest-asyncio auto mode from pyproject collects the bare async tests; no marks needed.)

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/server.py
"""FastMCP server: 10 tools over the shared core."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from minirag_mcp import __version__
from minirag_mcp.config import Config, ConfigError, load_config
from minirag_mcp.embedder import Embedder
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.ingest.scanner import compute_states, scan_roots
from minirag_mcp.security import resolve_in_roots
from minirag_mcp.store import SearchResult, Store
from minirag_mcp.sync import SyncManager


@dataclass
class _Ctx:
    config: Config
    store: Store
    embedder: object
    pipeline: Pipeline
    sync: SyncManager


def _result_dict(r: SearchResult) -> dict:
    return {
        "text": r.text, "source": r.source, "title": r.title,
        "chunkIndex": r.chunk_index, "score": r.score, "distance": r.distance,
    }


def _scopes(scope: str | list[str] | None) -> tuple[str, ...]:
    if scope is None:
        return ()
    if isinstance(scope, str):
        return (scope,)
    return tuple(scope)


def create_app(
    config: Config | None,
    *,
    config_error: str | None = None,
    embedder: object | None = None,
) -> FastMCP:
    mcp = FastMCP("minirag-mcp")
    holder: dict[str, _Ctx | None] = {"ctx": None}

    def ctx() -> _Ctx:
        if config is None or config_error is not None:
            raise ToolError(f"Configuration error: {config_error or 'no configuration'}")
        if holder["ctx"] is None:
            emb = embedder if embedder is not None else Embedder(config.model_name, config.cache_dir)
            store = Store(config.db_path, dim=emb.dim)
            pipeline = Pipeline(store, emb, config)
            holder["ctx"] = _Ctx(config, store, emb, pipeline, SyncManager(pipeline, store, config))
        return holder["ctx"]

    def guard(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except ToolError:
                raise
            except KeyError as e:
                raise ToolError(str(e.args[0]) if e.args else str(e)) from e
            except Exception as e:
                raise ToolError(str(e)) from e
        return inner

    @mcp.tool
    @guard
    def sync_start(path: str | None = None) -> dict:
        """Reconcile the index with the document roots (or one path inside them).

        Returns a jobId immediately; poll sync_status until state is
        'succeeded' or 'failed'. New and changed files are ingested,
        byte-identical files skipped, vanished files removed from the index.
        """
        c = ctx()
        scope = None
        if path is not None:
            scope = resolve_in_roots(path, c.config.roots)
        return {"jobId": c.sync.start(scope)}

    @mcp.tool
    @guard
    def sync_status(jobId: str) -> dict:
        """Poll a sync job started by sync_start."""
        return ctx().sync.status(jobId).to_dict()

    @mcp.tool
    @guard
    def ingest_file(filePath: str) -> dict:
        """Ingest or re-ingest one file (absolute path inside a document root)."""
        c = ctx()
        path = resolve_in_roots(filePath, c.config.roots)
        r = c.pipeline.ingest_file(path)
        return {"source": r.source, "chunkCount": r.chunk_count, "title": r.title}

    @mcp.tool
    @guard
    def ingest_data(data: str, source: str, format: str = "text", title: str | None = None) -> dict:
        """Ingest text/markdown/html content held by the client under a stable source id."""
        r = ctx().pipeline.ingest_data(data, source=source, fmt=format, title=title)
        return {"source": r.source, "chunkCount": r.chunk_count, "title": r.title}

    @mcp.tool
    @guard
    def ingest_url(url: str, source: str | None = None, title: str | None = None) -> dict:
        """Fetch an http(s) URL, convert it to Markdown, and index it."""
        r = ctx().pipeline.ingest_url(url, source=source, title=title)
        return {"source": r.source, "chunkCount": r.chunk_count, "title": r.title}

    @mcp.tool
    @guard
    def query_documents(query: str, topK: int = 8, scope: str | list[str] | None = None) -> dict:
        """Hybrid search: semantic similarity + keyword boost for exact terms."""
        if not query.strip():
            raise ToolError("query must not be empty")
        c = ctx()
        results = c.store.search(
            query, c.embedder.embed_query(query),
            top_k=topK, hybrid_weight=c.config.hybrid_weight, scopes=_scopes(scope),
            max_distance=c.config.max_distance, grouping=c.config.grouping,
            max_files=c.config.max_files,
        )
        sources: dict[str, dict] = {}
        for r in results:  # rank order: first hit of a source fixes its position
            agg = sources.setdefault(r.source, {"source": r.source, "title": r.title, "hits": 0})
            agg["hits"] += 1
        return {"results": [_result_dict(r) for r in results], "sources": list(sources.values())}

    @mcp.tool
    @guard
    def read_chunk_neighbors(
        chunkIndex: int,
        filePath: str | None = None,
        source: str | None = None,
        before: int = 1,
        after: int = 1,
    ) -> dict:
        """Read chunks surrounding a search result for more context."""
        c = ctx()
        if filePath is not None:
            key = str(resolve_in_roots(filePath, c.config.roots))
        elif source is not None:
            key = source
        else:
            raise ToolError("Provide filePath or source")
        chunks = c.store.neighbors(key, chunkIndex, before=before, after=after)
        return {"chunks": [_result_dict(r) for r in chunks]}

    @mcp.tool
    @guard
    def read_file(filePath: str | None = None, source: str | None = None) -> dict:
        """Read a source's full indexed content (all chunks in order, as Markdown)."""
        c = ctx()
        if filePath is not None:
            key = str(resolve_in_roots(filePath, c.config.roots))
        elif source is not None:
            key = source
        else:
            raise ToolError("Provide filePath or source")
        chunks = c.store.all_chunks(key)
        if not chunks:
            raise ToolError(f"Source not found in index: {key}")
        info = c.store.get_source(key)
        return {
            "source": key,
            "sourceType": info.source_type,
            "title": info.title,
            "chunkCount": len(chunks),
            "text": "\n\n".join(ch.text for ch in chunks),
        }

    @mcp.tool
    @guard
    def list_files(scope: str | list[str] | None = None) -> dict:
        """List files on disk (with ingestion state) and indexed data/url sources."""
        c = ctx()
        scopes = _scopes(scope)
        entries = scan_roots(c.config.roots)
        if scopes:
            entries = [e for e in entries if any(str(e.path).startswith(p) for p in scopes)]
        indexed = c.store.list_sources(scopes=scopes)
        states = compute_states(entries, indexed)
        return {
            "files": [
                {
                    "source": s.source, "sourceType": s.source_type, "title": s.title,
                    "state": s.state, "chunkCount": s.chunk_count,
                }
                for s in states
            ]
        }

    @mcp.tool
    @guard
    def delete_file(filePath: str | None = None, source: str | None = None) -> dict:
        """Delete an indexed file (by absolute path) or a data/url item (by source id)."""
        c = ctx()
        if filePath is not None:
            key = str(resolve_in_roots(filePath, c.config.roots))
        elif source is not None:
            key = source
        else:
            raise ToolError("Provide filePath or source")
        deleted = c.store.delete_source(key)
        if deleted == 0:
            raise ToolError(f"Source not found in index: {key}")
        return {"source": key, "deletedChunks": deleted}

    @mcp.tool
    def status() -> dict:
        """Index and configuration status. Works even when configuration is invalid."""
        if config is None or config_error is not None:
            return {"version": __version__, "configError": config_error or "no configuration"}
        out: dict = {
            "version": __version__,
            "roots": [str(r) for r in config.roots],
            "dbPath": str(config.db_path),
            "model": config.model_name,
            "hybridWeight": config.hybrid_weight,
        }
        try:
            c = ctx()
            out["chunkCount"] = c.store.chunk_count()
            out["sourceCount"] = c.store.source_count()
        except Exception as e:
            out["indexError"] = str(e)
        return out

    return mcp


def run_server() -> None:
    try:
        config: Config | None = load_config(os.environ)
        error = None
    except ConfigError as e:
        config, error = None, str(e)
    create_app(config, config_error=error).run()
```

```python
# src/minirag_mcp/__main__.py
"""Entry point: no args -> MCP server on stdio; subcommand -> CLI."""

import sys


def main() -> None:
    if len(sys.argv) > 1:
        from minirag_mcp.cli import app

        app()
    else:
        from minirag_mcp.server import run_server

        run_server()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_server.py -q`
Expected: all pass. The sync polling loop is a plain loop because the fake-embedder sync completes in milliseconds; if it flakes, insert `await asyncio.sleep(0.05)` in the polling loop.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: everything green.

- [ ] **Step 6: Commit**

```bash
git add src/minirag_mcp/server.py src/minirag_mcp/__main__.py tests/test_server.py
git commit -m "feat: FastMCP server with 10 tools and degraded-config mode"
```

---

### Task 14: cli/

**Files:**
- Create: `src/minirag_mcp/cli/__init__.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: cyclopts `app` with subcommands `ingest`, `ingest-url`, `sync`, `query`, `read-neighbors`, `read`, `list`, `status`, `delete`. `read` prints a source's full indexed Markdown; `list` shows disk files with `ingested|not_ingested|stale` state plus data/url sources (same `compute_states` join as the server). Shared option quartet on every command: `--base-dir` (repeatable), `--db-path`, `--cache-dir`, `--model-name` (flags beat env; note: flags come AFTER the subcommand — documented deviation from the reference's global-flags-first style). `--json` (`Annotated[bool, Parameter(name="--json")]`) switches to JSON output. Human output → stdout; errors → stderr + `SystemExit(1)`. CLI paths may be relative (resolved against cwd, `require_absolute=False`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
import json

import pytest

import minirag_mcp.cli as cli


@pytest.fixture(autouse=True)
def fake_model(monkeypatch, fake_embedder):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)


def run(tokens):
    return cli.app(tokens, result_action="return_value", exit_on_error=False)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha body about tokens and auth.")
    (tmp_path / "sub" / "b.md").write_text("# Beta\n\nERR_CONNECTION_REFUSED appears here.")
    return tmp_path


def test_ingest_directory_recursive_and_query(corpus, capsys):
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    out = capsys.readouterr().out
    assert "2" in out  # 2 files ingested

    run(["query", "ERR_CONNECTION_REFUSED", "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"], "expected results"


def test_ingest_single_file(corpus, capsys):
    run(["ingest", str(corpus / "a.md"), "--base-dir", str(corpus)])
    assert "a.md" in capsys.readouterr().out


def test_list_and_status_and_delete(corpus, capsys):
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["list", "--base-dir", str(corpus), "--json"])
    files = json.loads(capsys.readouterr().out)["files"]
    assert len(files) == 2 and all(f["state"] == "ingested" for f in files)

    run(["status", "--base-dir", str(corpus), "--json"])
    st = json.loads(capsys.readouterr().out)
    assert st["chunkCount"] >= 2

    run(["delete", str(corpus / "a.md"), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["list", "--base-dir", str(corpus), "--json"])
    files = json.loads(capsys.readouterr().out)["files"]
    by_source = {f["source"]: f["state"] for f in files}
    assert by_source[str(corpus / "a.md")] == "not_ingested"  # still on disk, gone from index


def test_read_full_source(corpus, capsys):
    run(["ingest", str(corpus / "a.md"), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["read", str(corpus / "a.md"), "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "Alpha body" in payload["text"] and payload["chunkCount"] >= 1


def test_sync_and_read_neighbors(corpus, capsys):
    run(["sync", "--base-dir", str(corpus)])
    out = capsys.readouterr().out
    assert "ingested" in out

    run([
        "read-neighbors", "--file-path", str(corpus / "a.md"), "--chunk-index", "0",
        "--base-dir", str(corpus), "--json",
    ])
    assert json.loads(capsys.readouterr().out)["chunks"]


def test_ingest_url_mocked(corpus, capsys, monkeypatch):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url: ParsedDoc("# R\n\nRemote body.", "R"))
    run(["ingest-url", "https://example.com/p", "--base-dir", str(corpus)])
    assert "example.com" in capsys.readouterr().out


def test_error_exits_nonzero(corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        run(["delete", str(corpus / "never-ingested.md"), "--base-dir", str(corpus)])
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_env_used_when_no_flags(corpus, capsys, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(corpus))
    run(["ingest", str(corpus)])
    assert "2" in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# src/minirag_mcp/cli/__init__.py
"""CLI over the same core as the MCP server. Flags beat env vars."""

from __future__ import annotations

import json as jsonlib
import os
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter

from minirag_mcp import __version__
from minirag_mcp.config import Config, ConfigError, load_config
from minirag_mcp.embedder import Embedder
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.ingest.scanner import compute_states, scan_roots
from minirag_mcp.security import SecurityError, resolve_in_roots
from minirag_mcp.store import SearchResult, Store
from minirag_mcp.sync import run_sync

app = App(name="minirag-mcp", version=__version__)

JsonFlag = Annotated[bool, Parameter(name="--json", help="Machine-readable JSON output")]


def _make_embedder(cfg: Config) -> Embedder:
    return Embedder(cfg.model_name, cfg.cache_dir)  # patched in tests


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _load(
    base_dir: list[str] | None,
    db_path: str | None,
    cache_dir: str | None,
    model_name: str | None,
) -> tuple[Config, Store, Pipeline]:
    try:
        cfg = load_config(
            os.environ,
            base_dir_flags=base_dir or (),
            db_path_flag=db_path,
            cache_dir_flag=cache_dir,
            model_name_flag=model_name,
        )
        emb = _make_embedder(cfg)
        store = Store(cfg.db_path, dim=emb.dim)
        return cfg, store, Pipeline(store, emb, cfg)
    except ConfigError as e:
        _fail(str(e))
        raise AssertionError  # unreachable


def _emit(payload: dict, as_json: bool, human: str) -> None:
    print(jsonlib.dumps(payload, ensure_ascii=False, indent=2) if as_json else human)


def _result_line(r: SearchResult) -> str:
    return f"[{r.score:.3f}] {r.source}#{r.chunk_index} — {r.title}\n    {r.text[:160]}"


@app.command
def ingest(
    paths: list[str],
    *,
    base_dir: list[str] | None = None,
    db_path: str | None = None,
    cache_dir: str | None = None,
    model_name: str | None = None,
    json: JsonFlag = False,
):
    """Ingest files or directories (recursive)."""
    cfg, store, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    files: list[Path] = []
    try:
        for raw in paths:
            p = resolve_in_roots(raw, cfg.roots, require_absolute=False)
            if p.is_dir():
                files.extend(e.path for e in scan_roots([p]))
            else:
                files.append(p)
        done = [pipeline.ingest_file(f) for f in files]
    except Exception as e:  # SecurityError, pipeline errors, parser errors
        _fail(str(e))
    payload = {
        "ingested": [{"source": r.source, "chunkCount": r.chunk_count} for r in done]
    }
    _emit(payload, json, f"ingested {len(done)} file(s):\n" + "\n".join(f"  {r.source}" for r in done))


@app.command(name="ingest-url")
def ingest_url(
    url: str,
    *,
    source: str | None = None,
    title: str | None = None,
    base_dir: list[str] | None = None,
    db_path: str | None = None,
    cache_dir: str | None = None,
    model_name: str | None = None,
    json: JsonFlag = False,
):
    """Fetch an http(s) URL and index it."""
    _, _, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    try:
        r = pipeline.ingest_url(url, source=source, title=title)
    except Exception as e:
        _fail(str(e))
    _emit(
        {"source": r.source, "chunkCount": r.chunk_count, "title": r.title},
        json, f"ingested {r.source} ({r.chunk_count} chunks)",
    )


@app.command
def sync(
    path: str | None = None,
    *,
    base_dir: list[str] | None = None,
    db_path: str | None = None,
    cache_dir: str | None = None,
    model_name: str | None = None,
    json: JsonFlag = False,
):
    """Reconcile the index with the document roots (synchronous)."""
    cfg, store, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    scope = None
    if path is not None:
        try:
            scope = resolve_in_roots(path, cfg.roots, require_absolute=False)
        except SecurityError as e:
            _fail(str(e))
    counts, errors = run_sync(
        pipeline, store, cfg.roots, cfg.max_file_size, scope=scope,
        on_event=lambda m: print(m, file=sys.stderr),
    )
    human = ", ".join(f"{k}: {v}" for k, v in counts.items())
    _emit({"counts": counts, "errors": errors}, json, human)
    if errors:
        for err in errors:
            print(f"warn: {err['source']}: {err['error']}", file=sys.stderr)


@app.command
def query(
    text: str,
    *,
    scope: list[str] | None = None,
    top_k: int = 8,
    base_dir: list[str] | None = None,
    db_path: str | None = None,
    cache_dir: str | None = None,
    model_name: str | None = None,
    json: JsonFlag = False,
):
    """Hybrid search over the index."""
    cfg, store, pipeline = _load(base_dir, db_path, cache_dir, model_name)
    results = store.search(
        text, pipeline.embedder.embed_query(text),
        top_k=top_k, hybrid_weight=cfg.hybrid_weight, scopes=tuple(scope or ()),
        max_distance=cfg.max_distance, grouping=cfg.grouping, max_files=cfg.max_files,
    )
    payload = {
        "results": [
            {"text": r.text, "source": r.source, "title": r.title,
             "chunkIndex": r.chunk_index, "score": r.score, "distance": r.distance}
            for r in results
        ]
    }
    _emit(payload, json, "\n".join(_result_line(r) for r in results) or "no results")


@app.command(name="read-neighbors")
def read_neighbors(
    *,
    file_path: str | None = None,
    source: str | None = None,
    chunk_index: int,
    before: int = 1,
    after: int = 1,
    base_dir: list[str] | None = None,
    db_path: str | None = None,
    cache_dir: str | None = None,
    model_name: str | None = None,
    json: JsonFlag = False,
):
    """Read chunks around a given chunk index."""
    cfg, store, _ = _load(base_dir, db_path, cache_dir, model_name)
    if file_path is not None:
        try:
            key = str(resolve_in_roots(file_path, cfg.roots, require_absolute=False))
        except SecurityError as e:
            _fail(str(e))
    elif source is not None:
        key = source
    else:
        _fail("provide --file-path or --source")
    chunks = store.neighbors(key, chunk_index, before=before, after=after)
    payload = {"chunks": [{"chunkIndex": r.chunk_index, "text": r.text} for r in chunks]}
    _emit(payload, json, "\n\n".join(f"[{r.chunk_index}] {r.text}" for r in chunks) or "no chunks")


@app.command
def read(
    path: str | None = None,
    *,
    source: str | None = None,
    base_dir: list[str] | None = None,
    db_path: str | None = None,
    cache_dir: str | None = None,
    model_name: str | None = None,
    json: JsonFlag = False,
):
    """Print a source's full indexed content (all chunks, as Markdown)."""
    cfg, store, _ = _load(base_dir, db_path, cache_dir, model_name)
    if path is not None:
        try:
            key = str(resolve_in_roots(path, cfg.roots, require_absolute=False))
        except SecurityError as e:
            _fail(str(e))
    elif source is not None:
        key = source
    else:
        _fail("provide a path or --source")
    chunks = store.all_chunks(key)
    if not chunks:
        _fail(f"source not found in index: {key}")
    text = "\n\n".join(ch.text for ch in chunks)
    payload = {"source": key, "chunkCount": len(chunks), "text": text}
    _emit(payload, json, text)


@app.command(name="list")
def list_cmd(
    *,
    scope: list[str] | None = None,
    base_dir: list[str] | None = None,
    db_path: str | None = None,
    cache_dir: str | None = None,
    model_name: str | None = None,
    json: JsonFlag = False,
):
    """List files on disk with ingestion state, plus indexed data/url sources."""
    cfg, store, _ = _load(base_dir, db_path, cache_dir, model_name)
    scopes = tuple(scope or ())
    entries = scan_roots(cfg.roots)
    if scopes:
        entries = [e for e in entries if any(str(e.path).startswith(p) for p in scopes)]
    states = compute_states(entries, store.list_sources(scopes=scopes))
    payload = {
        "files": [
            {"source": s.source, "sourceType": s.source_type, "title": s.title,
             "state": s.state, "chunkCount": s.chunk_count}
            for s in states
        ]
    }
    human = "\n".join(f"[{s.state}] {s.source} ({s.chunk_count} chunks)" for s in states)
    _emit(payload, json, human or "no files found")


@app.command
def status(
    *,
    base_dir: list[str] | None = None,
    db_path: str | None = None,
    cache_dir: str | None = None,
    model_name: str | None = None,
    json: JsonFlag = False,
):
    """Show index and configuration status."""
    cfg, store, _ = _load(base_dir, db_path, cache_dir, model_name)
    payload = {
        "version": __version__,
        "roots": [str(r) for r in cfg.roots],
        "dbPath": str(cfg.db_path),
        "model": cfg.model_name,
        "chunkCount": store.chunk_count(),
        "sourceCount": store.source_count(),
    }
    human = "\n".join(f"{k}: {v}" for k, v in payload.items())
    _emit(payload, json, human)


@app.command
def delete(
    path: str | None = None,
    *,
    source: str | None = None,
    base_dir: list[str] | None = None,
    db_path: str | None = None,
    cache_dir: str | None = None,
    model_name: str | None = None,
    json: JsonFlag = False,
):
    """Delete an indexed file (by path) or data/url item (by --source)."""
    cfg, store, _ = _load(base_dir, db_path, cache_dir, model_name)
    if path is not None:
        try:
            key = str(resolve_in_roots(path, cfg.roots, require_absolute=False))
        except SecurityError as e:
            _fail(str(e))
    elif source is not None:
        key = source
    else:
        _fail("provide a path or --source")
    deleted = store.delete_source(key)
    if deleted == 0:
        _fail(f"source not found in index: {key}")
    _emit({"source": key, "deletedChunks": deleted}, json, f"deleted {deleted} chunk(s) of {key}")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cli.py -q`
Expected: all pass. (`_fail` raises `SystemExit`, which is a `BaseException` — the `except Exception` blocks never swallow it.)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: everything green.

- [ ] **Step 6: Commit**

```bash
git add src/minirag_mcp/cli tests/test_cli.py
git commit -m "feat: cyclopts CLI over the shared core"
```

---

### Task 15: README + spec sync

**Files:**
- Create: `README.md`
- Modify: `docs/superpowers/specs/2026-08-07-minirag-mcp-v1-design.md` (CLI flags line)

**Interfaces:**
- Produces: English README covering: what it is (local-first RAG MCP server), quick start (`claude mcp add minirag --scope user --env BASE_DIR=/abs/path -- uvx minirag-mcp`, plus Cursor/Codex JSON snippets), the 11 tools table, CLI examples, configuration table (all env vars + defaults, incl. the DB-next-to-docs and global-model-cache defaults), search tuning (`RAG_HYBRID_WEIGHT` etc.), security notes (roots boundary, symlink rejection, http/https-only `ingest_url`, single writer per DB_PATH, network promise: only `ingest_url` and the one-time model download touch the network), troubleshooting (model download, "No results found" → ingest first, changing MODEL_NAME → re-ingest).

- [ ] **Step 1: Write README.md** — follow the reference README's structure (Features / Quick Start / Supported Content / MCP Tools / CLI / Search Tuning / Configuration / Security and Operation / Troubleshooting), adapted to this stack. State explicitly: model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~220 MB, 50+ languages), downloads on first use to the platformdirs cache; index lives at `<first root>/.minirag/lancedb` by default.

- [ ] **Step 2: Update the spec's CLI section** — replace the "Global options before the subcommand" sentence with: "The option quartet `--base-dir/--db-path/--cache-dir/--model-name` is accepted by every subcommand (flags after the subcommand); flags take precedence over env vars."

- [ ] **Step 3: Sanity-check examples** — every command in the README must be copy-pasteable; run `uv run minirag-mcp status --base-dir .` locally and fix drift.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/
git commit -m "docs: README and spec sync"
```

---

### Task 16: Quality gates and build

**Files:**
- Modify: whatever ruff flags.

- [ ] **Step 1: Lint and format**

Run: `uv run ruff check src tests --fix && uv run ruff format src tests`
Expected: clean (or auto-fixed; re-run tests after fixes).

- [ ] **Step 2: Full test suite**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 3: Slow integration test (real model, one-time ~220 MB download)**

Run: `uv run pytest -m slow -q`
Expected: pass. Skip only if offline; note it in the final report if skipped.

- [ ] **Step 4: Build and smoke the wheel**

Run: `uv build && uvx --from dist/minirag_mcp-0.1.0-py3-none-any.whl minirag-mcp status --base-dir .`
Expected: status output with version 0.1.0. (`uvx --from` a wheel exercises the entry point exactly like `uvx minirag-mcp` will after publishing.)

- [ ] **Step 5: Commit any fixes, close beads task**

```bash
git add -A && git commit -m "chore: lint fixes and build verification" || true
bd close minirag-mcp-cuz --reason "v1 implemented per plan"
```

---

## Self-review notes (done at plan time)

- **Spec coverage:** 11 tools (incl. `read_file`, disk-state `list_files`, `sources` aggregation in query responses) → Task 13; CLI 9 commands (incl. `read`) → Task 14; env contract + deviations → Task 2; chunking (structural+semantic, atomic code fences, CHUNK_MIN_LENGTH) → Tasks 4–5; hybrid + RAG_* filters → Task 8; recursive scan/whitelist/skip dirs + sha256 diff → Task 11; sync semantics (single job, data/url untouched) → Tasks 11–12; security (roots, symlink, absolute-only MCP paths, URL schemes) → Tasks 3, 10, 13; degraded status → Task 13; README → Task 15; slow real-model test → Tasks 6, 16.
- **Deliberate deviations locked in:** CLI flags after subcommand (spec updated in Task 15); `chunkIndex`-style camelCase only at the MCP/CLI boundary, snake_case internally.
- **Known risks accepted:** FTS coverage of freshly added rows (probe-verified; `optimize()` fallback documented in Task 8); `.select()` on scalar search (fallback documented in Task 7); pytest-asyncio collection mode (fallback documented in Task 13).
