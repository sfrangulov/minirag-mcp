"""Run one of the benchmark's evaluators with the Codex judge swapped in.

Runs inside the benchmark's own venv with the benchmark checkout as working directory
(score.py arranges both):

    python judged_eval.py metrics|comparative --model M --cheap-model C -- <evaluator args>
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

MODULES = {
    "metrics": "src.scripts.answer_evaluation.metrics_based_eval",
    "comparative": "src.scripts.answer_evaluation.comparative_eval",
}


def main() -> int:
    sys.path.insert(0, os.getcwd())
    sys.path.insert(1, str(Path(__file__).resolve().parent))
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=MODULES)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cheap-model", required=True)
    ap.add_argument("--stats", type=Path, required=True)
    args = ap.parse_args(argv[:cut])
    rest = argv[cut + 1 :]

    import judge_codex

    evaluator = importlib.import_module(MODULES[args.which])
    swapped = judge_codex.install(args.model, args.cheap_model)
    if not swapped:
        raise SystemExit("found no get_llm to replace; the benchmark's layout changed")
    sys.argv = [MODULES[args.which]] + rest
    try:
        evaluator.main()
    finally:
        stats = dict(judge_codex.STATS, model=args.model, cheap_model=args.cheap_model)
        args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 1 if judge_codex.STATS["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
