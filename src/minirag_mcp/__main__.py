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
