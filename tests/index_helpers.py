"""An indexes.toml with three indexes over small corpora, for the tests of file mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minirag_mcp.registry import IndexRegistry

CORPORA = {
    "alpha": ("alpha.md", "# Alpha\n\nAlpha body about apricots and audits."),
    "beta": ("beta.md", "# Beta\n\nBeta body about bananas and budgets."),
    "gamma": ("gamma.md", "# Gamma\n\nGamma body about grapes and gardens."),
}
INDEXES_TOML = """\
[index.alpha]
description = "Alpha documents"
roots = ["~/corpora/alpha"]

[index.beta]
description = "Beta documents"
roots = ["~/corpora/beta"]
fts_language = "Russian"

[index.gamma]
description = "Gamma documents"
roots = ["~/corpora/gamma"]
model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
"""


@dataclass(frozen=True)
class IndexesFile:
    path: Path
    env: dict[str, str]
    roots: dict[str, Path]
    # What the file held when it was made; `write` leaves it, so a test can restore it.
    text: str

    def write(self, text: str) -> None:
        self.path.write_text(text, encoding="utf-8")


def make_indexes_file(tmp_path: Path) -> IndexesFile:
    roots = {}
    for name, (filename, body) in CORPORA.items():
        root = tmp_path / "corpora" / name
        root.mkdir(parents=True)
        (root / filename).write_text(body, encoding="utf-8")
        roots[name] = root
    path = tmp_path / "config" / "indexes.toml"
    path.parent.mkdir(parents=True)
    made = IndexesFile(path, {"HOME": str(tmp_path)}, roots, INDEXES_TOML)
    made.write(INDEXES_TOML)
    return made


def file_registry(indexes_file: IndexesFile, embedder_factory, **kw) -> IndexRegistry:
    # Imported here: conftest imports this module, and so does every sync worker the tests
    # spawn that unpickles a conftest stand-in.
    from minirag_mcp.registry import IndexRegistry

    return IndexRegistry.from_file(
        indexes_file.path, indexes_file.env, embedder_factory=embedder_factory, **kw
    )


def run_meta(tokens: list[str], capsys) -> tuple[int, str, str]:
    """The CLI as its entry point runs it, with the options around the command read first:
    the exit code, stdout and stderr."""
    import minirag_mcp.cli as cli  # imported here for the reason file_registry gives

    try:
        cli.app.meta(tokens, result_action="return_value", exit_on_error=False)
        code = 0
    except SystemExit as e:
        code = e.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err
