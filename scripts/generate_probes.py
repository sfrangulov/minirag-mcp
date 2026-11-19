"""Writes src/minirag_mcp/probes.json: PROBE_TEXT as the product embeds a document, per
known model, made by fastembed in-process from the locally cached models.

    HF_HUB_OFFLINE=1 uv run python scripts/generate_probes.py [CACHE_DIR]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import platformdirs

from minirag_mcp.embedder import E5_LARGE, MINILM, Embedder

TARGET = Path(__file__).resolve().parent.parent / "src" / "minirag_mcp" / "probes.json"


def compute(cache_dir: Path) -> dict[str, list[float]]:
    # No endpoint: the vectors are fastembed's own.
    return {
        model: [round(x, 6) for x in Embedder(model, cache_dir).probe_vector()]
        for model in (E5_LARGE, MINILM)
    }


if __name__ == "__main__":
    cache = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else platformdirs.user_cache_path("minirag-mcp") / "models"
    )
    TARGET.write_text(json.dumps(compute(cache), separators=(",", ":")) + "\n", encoding="utf-8")
