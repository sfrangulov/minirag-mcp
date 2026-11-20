"""Run as a process's main script: starts one of minirag's entry points as this process's
`__main__`, the way a client starts it, with the arguments ENTRY_ARGS (a JSON list, none
by default) and the MCP server swapped, in this process only, for a parallel sync: of
BASE_DIR where the environment's server would start, of the index served where
`--index NAME` would serve one, of the first index of indexes.toml where `serve` would
listen. Spawned workers start without the swap, as a real server's do. The entry point is
ENTRY, a script path or, with ENTRY_KIND=module, a module name. The sync reports to
stderr: stdout is the protocol channel."""

import json
import os
import runpy
import sys
from pathlib import Path

# Workers inherit sys.path, and unpickle the embedder from conftest.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402
from fastmcp import FastMCP  # noqa: E402

import minirag_mcp.cli as cli  # noqa: E402
import minirag_mcp.server as server  # noqa: E402
from conftest import FakeEmbedder  # noqa: E402
from minirag_mcp import ocr, parallel  # noqa: E402
from minirag_mcp.config import Config, load_config  # noqa: E402
from minirag_mcp.handle import IndexHandle  # noqa: E402
from minirag_mcp.indexes import load_server_config  # noqa: E402


def sync(config: Config) -> None:
    pools: list[int] = []
    prepare_files = parallel.prepare_files

    def spy(make_preparer, paths, *, workers, **kwargs):
        pools.append(workers)
        return prepare_files(make_preparer, paths, workers=workers, **kwargs)

    parallel.prepare_files = spy
    # Not the subject, and fetching Tesseract's data would go online.
    ocr.available = lambda: False
    handle = IndexHandle(config, embedder_factory=lambda scheme: FakeEmbedder())
    counts, errors, _ = handle.run_sync(None)
    print(json.dumps({"counts": counts, "errors": errors, "pools": pools}), file=sys.stderr)


def sync_instead_of_serving() -> None:
    sync(load_config(os.environ))


def sync_the_served_index_instead(app: FastMCP, *args, **kwargs) -> None:
    # Called by `--index NAME`, while the options around the command are still selected.
    served = load_server_config(cli._config_path(), os.environ)
    sync(served.indexes[cli._SELECTION.get().index])


async def sync_the_first_index_instead(self: uvicorn.Server, *args, **kwargs) -> None:
    # Called by `serve` in place of listening, while the options around the command are
    # still selected.
    served = load_server_config(cli._config_path(), os.environ)
    sync(next(iter(served.indexes.values())))


server.run_server = sync_instead_of_serving
FastMCP.run = sync_the_served_index_instead
uvicorn.Server.serve = sync_the_first_index_instead
sys.argv[1:] = json.loads(os.environ.get("ENTRY_ARGS", "[]"))

if os.environ.get("ENTRY_KIND") == "module":
    runpy.run_module(os.environ["ENTRY"], run_name="__main__", alter_sys=True)
else:
    runpy.run_path(os.environ["ENTRY"], run_name="__main__")
