"""Entry point: no args -> MCP server on stdio for the environment's index; otherwise the
CLI, whose `--index NAME` without a command serves that index of indexes.toml on stdio."""

import sys


def main() -> None:
    if len(sys.argv) > 1:
        from minirag_mcp.cli import app

        app.meta()
    else:
        from minirag_mcp.server import run_server

        run_server()


if __name__ == "__main__":
    main()
