"""An indexes.toml with three indexes over small corpora, for the tests of file mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
