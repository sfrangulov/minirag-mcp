"""Run as a process's main script: starts one of minirag's entry points as this process's
`__main__`, the way a client starts it, with the MCP server swapped, in this process only,
for a parallel sync of BASE_DIR. Spawned workers start without the swap, as a real
server's do. The entry point is ENTRY, a script path or, with ENTRY_KIND=module, a module
name. The sync reports to stderr: stdout is the protocol channel."""

import json
import os
import runpy
import sys
from pathlib import Path

# Workers inherit sys.path, and unpickle the embedder from conftest.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import minirag_mcp.server as server  # noqa: E402
from conftest import FakeEmbedder  # noqa: E402
from minirag_mcp import ocr, parallel  # noqa: E402
from minirag_mcp.config import load_config  # noqa: E402
from minirag_mcp.handle import IndexHandle  # noqa: E402


def sync_instead_of_serving() -> None:
    pools: list[int] = []
    prepare_files = parallel.prepare_files

    def spy(make_preparer, paths, *, workers, **kwargs):
        pools.append(workers)
        return prepare_files(make_preparer, paths, workers=workers, **kwargs)

    parallel.prepare_files = spy
    # Not the subject, and fetching Tesseract's data would go online.
    ocr.available = lambda: False
    handle = IndexHandle(load_config(os.environ), embedder_factory=lambda scheme: FakeEmbedder())
    counts, errors, _ = handle.run_sync(None)
    print(json.dumps({"counts": counts, "errors": errors, "pools": pools}), file=sys.stderr)


server.run_server = sync_instead_of_serving

if os.environ.get("ENTRY_KIND") == "module":
    runpy.run_module(os.environ["ENTRY"], run_name="__main__", alter_sys=True)
else:
    runpy.run_path(os.environ["ENTRY"], run_name="__main__")
