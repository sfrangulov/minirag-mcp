"""minirag-mcp: local-first RAG over your documents, as an MCP server and a CLI."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("minirag-mcp")
except PackageNotFoundError:
    # Importable without being installed — a source checkout on sys.path, or a
    # vendored copy. There is no metadata to read, and guessing a number here
    # would be worse than admitting it: `status` reports this string to the
    # user, and a plausible-looking version that is not the installed one is
    # exactly the drift this indirection exists to prevent.
    __version__ = "0+unknown"

__all__ = ["__version__"]
