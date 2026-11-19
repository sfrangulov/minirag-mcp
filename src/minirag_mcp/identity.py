"""What an index was built with, how a configuration differs from it, and the swap.

Every v2 index folder holds INDEX_FILE: the embedding and chunking schemes its vectors and
chunks were made with, its generation, and the probe vector its model gave for PROBE_TEXT.
A folder with a LanceDB table and no INDEX_FILE is a minirag 1.x index, which v2 never
writes to. A rebuild builds a complete index in `staging_dir(db)`; `swap_in` moves it into
place and keeps the one it replaces as `previous_dir(db)`.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import shutil
import uuid
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from minirag_mcp import __version__
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.lock import is_held

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX-only; the tests all run on POSIX
    fcntl = None

if TYPE_CHECKING:
    from minirag_mcp.config import Config
    from minirag_mcp.store import IndexOpenError

# `config`, `store` and `embedder` are imported inside the functions that use them, so
# that embedder.py can import the probe constants from here: store imports config, and
# config imports embedder.

INDEX_FILE = "index.json"
FORMAT = 2
PROBE_COSINE = 0.999
# Never changed: every recorded probe is this text's vector, and an index is verified
# by embedding it again.
PROBE_TEXT = (
    "minirag index probe. The quick brown fox jumps over the lazy dog. "
    "Съешь же ещё этих мягких французских булок, да выпей чаю."
)
# The table every 1.x index holds; 1.x is frozen, so this name is too.
_LEGACY_TABLE = "chunks.lance"


@dataclass(frozen=True)
class EmbeddingScheme:
    model: str
    dim: int
    prefixes: str


@dataclass(frozen=True)
class ChunkingScheme:
    scheme_version: int
    token_budget: int


@dataclass(frozen=True)
class IndexIdentity:
    format: int
    minirag: str
    generation: int
    embedding: EmbeddingScheme
    # Stored under "embedding" in the file. Empty until the first write records it.
    probe: tuple[float, ...]
    chunking: ChunkingScheme
    fts_language: str
    document_table: bool

    @classmethod
    def fresh(
        cls, embedding: EmbeddingScheme, chunking: ChunkingScheme, fts_language: str
    ) -> IndexIdentity:
        """What a new index folder is stamped with at its first open."""
        return cls(
            format=FORMAT,
            minirag=__version__,
            generation=1,
            embedding=embedding,
            probe=(),
            chunking=chunking,
            fts_language=fts_language,
            document_table=True,
        )


class IndexState(StrEnum):
    READY = "ready"
    REBUILD_PENDING = "rebuild pending"
    SYNCING = "syncing"
    UNAVAILABLE = "unavailable"


class IdentityError(Exception):
    """INDEX_FILE is unreadable, malformed, or in a format newer than FORMAT."""


class LegacyIndexError(Exception):
    """The folder holds a minirag 1.x index."""


@dataclass(frozen=True)
class RebuildInfo:
    kind: str
    reasons: tuple[str, ...]
    generation: int
    reembedded: int
    restored: int

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "reasons": list(self.reasons),
            "generation": self.generation,
            "reembedded": self.reembedded,
            "restored": self.restored,
        }


def _open_error(db_path: Path, error: OSError) -> IndexOpenError:
    from minirag_mcp.store import IndexOpenError

    return IndexOpenError.at(db_path, error)


def _open_failure(message: str) -> IndexOpenError:
    from minirag_mcp.store import IndexOpenError

    return IndexOpenError(message)


def _to_json(identity: IndexIdentity) -> dict:
    return {
        "format": identity.format,
        "minirag": identity.minirag,
        "generation": identity.generation,
        "embedding": {
            "model": identity.embedding.model,
            "dim": identity.embedding.dim,
            "prefixes": identity.embedding.prefixes,
            "probe": list(identity.probe),
        },
        "chunking": {
            "scheme_version": identity.chunking.scheme_version,
            "token_budget": identity.chunking.token_budget,
        },
        "fts_language": identity.fts_language,
        "document_table": identity.document_table,
    }


class _Malformed(Exception):
    pass


_KINDS = {str: "a string", int: "an integer", bool: "true or false", dict: "an object"}


def _get(obj: dict, key: str, kind: type, prefix: str = ""):
    value = obj.get(key)
    if isinstance(value, kind) and not (kind is int and isinstance(value, bool)):
        return value
    if key not in obj:
        raise _Malformed(f"{prefix}{key} is missing")
    raise _Malformed(f"{prefix}{key} is not {_KINDS[kind]}")


def _parse(path: Path, data: object) -> IndexIdentity:
    if not isinstance(data, dict):
        raise _Malformed("it is not a JSON object")
    fmt = _get(data, "format", int)
    if fmt > FORMAT:
        writer = data.get("minirag")
        writer = f"minirag {writer}" if isinstance(writer, str) else "a newer minirag"
        raise IdentityError(
            f"{path} was written in index format {fmt} by {writer}, newer than format "
            f"{FORMAT}, the one this minirag ({__version__}) reads. Upgrade minirag-mcp to "
            "open this index, or point DB_PATH at another folder."
        )
    if fmt != FORMAT:
        raise _Malformed(f"format {fmt} is not one minirag writes")
    minirag = _get(data, "minirag", str)
    generation = _get(data, "generation", int)
    embedding = _get(data, "embedding", dict)
    chunking = _get(data, "chunking", dict)
    probe = embedding.get("probe")
    if not isinstance(probe, list) or not all(
        isinstance(x, int | float) and not isinstance(x, bool) for x in probe
    ):
        raise _Malformed("embedding.probe is not a list of numbers")
    return IndexIdentity(
        format=fmt,
        minirag=minirag,
        generation=generation,
        embedding=EmbeddingScheme(
            model=_get(embedding, "model", str, "embedding."),
            dim=_get(embedding, "dim", int, "embedding."),
            prefixes=_get(embedding, "prefixes", str, "embedding."),
        ),
        probe=tuple(float(x) for x in probe),
        chunking=ChunkingScheme(
            scheme_version=_get(chunking, "scheme_version", int, "chunking."),
            token_budget=_get(chunking, "token_budget", int, "chunking."),
        ),
        fts_language=_get(data, "fts_language", str),
        document_table=_get(data, "document_table", bool),
    )


def _malformed(path: Path, detail: str) -> IdentityError:
    return IdentityError(
        f"{path} is not a valid minirag index description: {detail}. It records the scheme "
        "this index was built with, so the index is not opened without it; restore the "
        "file, or point DB_PATH at an empty folder to build a new index."
    )


def read_identity(db_path: Path) -> IndexIdentity | None:
    """The identity recorded in `db_path`, or None when it has none."""
    path = db_path / INDEX_FILE
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except NotADirectoryError as e:
        raise _open_error(db_path, e) from e
    except OSError as e:
        raise IdentityError(
            f"{path} cannot be read: {e.strerror or e}. It records the scheme this index was "
            "built with, so the index is not opened until it can be read."
        ) from e
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise _malformed(path, f"it is not valid JSON ({e})") from e
    try:
        return _parse(path, data)
    except _Malformed as e:
        raise _malformed(path, str(e)) from None


def _write_new(path: Path, data: bytes) -> None:
    """Create `path` holding `data`, durably; FileExistsError when it already exists."""
    f = open(path, "xb")
    try:
        with f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _temp_file(db_path: Path, identity: IndexIdentity) -> tuple[Path, bytes]:
    try:
        db_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _open_error(db_path, e) from e
    # One name per writer: two processes stamping one folder must not share a file.
    tmp = db_path / f"{INDEX_FILE}.{uuid.uuid4().hex}.tmp"
    data = (json.dumps(_to_json(identity), ensure_ascii=False) + "\n").encode("utf-8")
    _write_new(tmp, data)
    return tmp, data


def write_identity(db_path: Path, identity: IndexIdentity) -> None:
    """Replace `db_path`'s identity in one step: a reader sees the old file or the new one."""
    tmp, _ = _temp_file(db_path, identity)
    try:
        os.replace(tmp, db_path / INDEX_FILE)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


# What link(2) answers on a filesystem without hard links (FAT, exFAT, some shares).
_NO_HARD_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})


def _lock_path(db_path: Path) -> Path:
    # Beside the index, not in it: the lock has to outlast the renames it orders.
    return db_path.with_name(f".{db_path.name}.identity.lock")


def _warn_unlocked(db_path: Path, why: str) -> None:
    warnings.warn(
        f"{why}, so this change to {db_path} runs unlocked: a stamp or swap of it by another "
        "process at the same moment is not kept apart from this one.",
        RuntimeWarning,
        stacklevel=4,  # past this helper, the context manager and contextlib
    )


@contextmanager
def _identity_lock(db_path: Path) -> Iterator[None]:
    """Hold the lock that orders stamps, swaps and swap recovery of `db_path`, waiting for it.

    Held only for a few renames and small writes. A lock that cannot be taken degrades as
    the sync lock does: it warns and the block runs unlocked, because refusing would leave
    an index on a platform or a folder without flock unopenable.
    """
    path = _lock_path(db_path)
    if fcntl is None:
        _warn_unlocked(db_path, f"fcntl is unavailable on this platform, so {path} is not taken")
        yield
        return
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        _warn_unlocked(db_path, f"Could not take the lock at {path} ({e.strerror or e})")
        yield
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as e:
            _warn_unlocked(db_path, f"Could not take the lock at {path} ({e.strerror or e})")
        yield
    finally:
        os.close(fd)


def _link_new(db_path: Path, identity: IndexIdentity) -> IndexIdentity:
    path = db_path / INDEX_FILE
    tmp, data = _temp_file(db_path, identity)
    try:
        try:
            os.link(tmp, path)
        except OSError as e:
            if e.errno not in _NO_HARD_LINKS:
                raise
            # Still never over another file, but a reader can see this one half-written.
            _write_new(path, data)
    except FileExistsError:
        found = read_identity(db_path)
        if found is None:
            raise IdentityError(
                f"{path} exists but holds nothing readable, so {db_path} is not stamped; remove it."
            ) from None
        return found
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return identity


def create_identity(db_path: Path, identity: IndexIdentity) -> IndexIdentity:
    """Stamp `db_path` with `identity` unless it has one; the identity it holds after.

    Never replaces a recorded identity, and never lands inside a swap, so a stamp cannot
    put a new, empty index in place of an existing one.
    """
    found = read_identity(db_path)
    if found is not None:
        return found
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _open_error(db_path, e) from e
    with _identity_lock(db_path):
        found = read_identity(db_path)
        if found is not None:
            return found
        staging, previous = staging_dir(db_path), previous_dir(db_path)
        if staging.is_dir() and previous.is_dir():
            # A swap that died between its renames left db_path missing; stamping it would
            # leave an empty index where both of these belong.
            _finish_swap(db_path)
            found = read_identity(db_path)
            if found is None:
                raise IdentityError(
                    f"{db_path} holds no {INDEX_FILE} while {staging} and {previous} both "
                    "exist, so a rebuild may be swapping one of them in; it is not stamped as "
                    f"a new index. If no sync is running, move {db_path} aside so that the "
                    f"next open finishes the swap of {staging}, or remove {staging} to start "
                    f"{db_path} anew."
                )
            return found
        return _link_new(db_path, identity)


def is_legacy_index(db_path: Path) -> bool:
    return (db_path / _LEGACY_TABLE).is_dir() and not os.path.lexists(db_path / INDEX_FILE)


def legacy_error(db_path: Path, roots: Sequence[Path]) -> LegacyIndexError:
    from minirag_mcp.config import default_db_path

    return LegacyIndexError(
        f"{db_path} holds a minirag 1.x index (a LanceDB table and no {INDEX_FILE}). "
        "minirag 2 never writes to a 1.x index. Leave DB_PATH unset to use "
        f"{default_db_path(roots)}, or point it at an empty folder."
    )


def differences(
    recorded: IndexIdentity, embedding: EmbeddingScheme, chunking: ChunkingScheme
) -> list[str]:
    """What a rebuild with these schemes would change, one phrase per difference."""
    was, old = recorded.embedding, recorded.chunking
    found = []
    if was.model != embedding.model:
        found.append(f"model {was.model} -> {embedding.model}")
    if was.prefixes != embedding.prefixes:
        found.append(f"query/passage prefixes {was.prefixes} -> {embedding.prefixes}")
    if was.model == embedding.model and was.dim != embedding.dim:
        found.append(f"vector width {was.dim} -> {embedding.dim}")
    if old.scheme_version != chunking.scheme_version:
        found.append(f"chunking scheme {old.scheme_version} -> {chunking.scheme_version}")
    if old.token_budget != chunking.token_budget:
        found.append(f"chunk budget {old.token_budget} -> {chunking.token_budget} tokens")
    return found


def rebuild_kind(
    recorded: IndexIdentity, embedding: EmbeddingScheme, chunking: ChunkingScheme
) -> str:
    """The rebuild needed: "reingest" when the chunks change, "reembed" when only the
    vectors do, else "none"."""
    if recorded.chunking != chunking:
        return "reingest"
    if recorded.embedding != embedding:
        return "reembed"
    return "none"


def index_state(config: Config, embedder) -> tuple[IndexState, str]:
    """The index's state against the configuration `embedder` was built for, and why.

    Reads files and fastembed's registry, and never loads a model or calls an endpoint. It
    may write, though: an interrupted swap is finished first (`recover_swap`).
    """
    from minirag_mcp.embedder import known_model
    from minirag_mcp.store import IndexOpenError

    db = config.db_path
    try:
        recover_swap(db)
        recorded = read_identity(db)
    except (IdentityError, IndexOpenError) as e:
        return IndexState.UNAVAILABLE, str(e)
    if recorded is None:
        if is_legacy_index(db):
            return IndexState.UNAVAILABLE, str(legacy_error(db, config.roots))
        return IndexState.READY, ""
    model = recorded.embedding.model
    if (
        model != embedder.model_name
        and model not in config.embed_endpoints
        and not known_model(model)
    ):
        return IndexState.UNAVAILABLE, (
            f"this index was built with {model}, which cannot be loaded here: no embedding "
            "endpoint is configured for it and fastembed does not know it"
        )
    found = differences(
        recorded,
        EmbeddingScheme(embedder.model_name, embedder.dim, embedder.prefixes),
        ChunkingScheme(SCHEME_VERSION, config.token_budget),
    )
    if found:
        return IndexState.REBUILD_PENDING, "; ".join(found)
    return IndexState.READY, ""


def staging_dir(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".next")


def previous_dir(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".previous")


def _check_previous(previous: Path) -> None:
    """Refuse to swap when `previous` holds anything but the index kept from the last swap."""
    if not os.path.lexists(previous):
        return
    if previous.is_symlink() or not previous.is_dir() or read_identity(previous) is None:
        raise IdentityError(
            f"{previous} is where a rebuild keeps the index it replaces, but it holds no "
            f"minirag index ({INDEX_FILE} is missing), so it is not deleted and nothing is "
            "swapped. Move it elsewhere and sync again."
        )


def _holds_generation(db_path: Path, generation: int) -> bool:
    if not db_path.is_dir():
        return False
    try:
        found = read_identity(db_path)
    except IdentityError:
        return False
    return found is not None and found.generation == generation


def _rollback_failed(db_path: Path, error: OSError, undo: OSError) -> str:
    staging, previous = staging_dir(db_path), previous_dir(db_path)
    if os.path.lexists(db_path):
        return (
            f"{db_path} was created again while {staging} was being swapped in for it, so the "
            f"swap failed ({error.strerror or error}) and {previous}, the index it was to "
            f"replace, could not be put back ({undo.strerror or undo}). What is at {db_path} "
            f"now is neither of them: move it aside, then rename {staging} or {previous} to "
            f"{db_path}."
        )
    return (
        f"{db_path} is missing: swapping in {staging} failed ({error.strerror or error}), and "
        f"putting back {previous}, the index it was to replace, failed too "
        f"({undo.strerror or undo}). The next open finishes the swap; or rename {staging} or "
        f"{previous} to {db_path}."
    )


def swap_in(db_path: Path) -> None:
    """Put the index built in `staging_dir(db_path)` in place, one generation up.

    The replaced index becomes `previous_dir(db_path)`, and the one kept there before is
    dropped. Between the two renames `db_path` does not exist; `recover_swap` finishes a
    swap whose process died there.
    """
    staging, previous = staging_dir(db_path), previous_dir(db_path)
    staged = read_identity(staging)
    if staged is None:
        raise IdentityError(
            f"{staging} holds no {INDEX_FILE}, so it is not a complete index and is not "
            f"swapped in for {db_path}."
        )
    if is_legacy_index(db_path):
        raise LegacyIndexError(
            f"{db_path} holds a minirag 1.x index (a LanceDB table and no {INDEX_FILE}). "
            f"minirag 2 never writes to a 1.x index, so {staging} is not swapped in for it."
        )
    _check_previous(previous)
    current = read_identity(db_path)
    generation = max(staged.generation, current.generation if current else 0) + 1
    write_identity(staging, replace(staged, generation=generation))
    if os.path.lexists(previous):
        shutil.rmtree(previous)
    with _identity_lock(db_path):
        had_index = os.path.lexists(db_path)
        if had_index:
            os.rename(db_path, previous)
        try:
            os.rename(staging, db_path)
        except OSError as e:
            # Unlocked (see _identity_lock), another process's recover_swap can move the
            # staged index in between the renames.
            if _holds_generation(db_path, generation):
                return
            if had_index:
                try:
                    os.rename(previous, db_path)
                except OSError as undo:
                    raise _open_failure(_rollback_failed(db_path, e, undo)) from e
            raise


def _finish_swap(db_path: Path) -> bool:
    """`recover_swap` for a caller already holding the identity lock."""
    staging, previous = staging_dir(db_path), previous_dir(db_path)
    if os.path.lexists(db_path) or not previous.is_dir():
        return False
    if read_identity(staging) is None:
        return False
    try:
        os.rename(staging, db_path)
    except OSError as e:
        if os.path.lexists(db_path):  # unlocked, the swapping process can get there first
            return False
        raise _open_failure(
            f"{db_path} is missing: a rebuild stopped while swapping in {staging}, and "
            f"finishing that swap failed: {e.strerror or e}. Rename {staging} to {db_path} to "
            f"finish it; {previous} holds the index it replaced."
        ) from e
    return True


def recover_swap(db_path: Path) -> bool:
    """Finish a swap whose process died between its two renames; True when one was.

    Such a process left no index at `db_path`, the replaced one in `previous_dir`, and the
    complete new one in `staging_dir`. Opening `db_path` as it is would start an empty index.
    """
    if os.path.lexists(db_path) or not previous_dir(db_path).is_dir():
        return False
    with _identity_lock(db_path):
        return _finish_swap(db_path)


def rebuild_running(db_path: Path) -> bool:
    """Whether a rebuild of `db_path` is in progress: its staging folder and the sync lock.

    The staging folder is checked first. Asking about the lock takes it for an instant,
    which a sync starting at that instant would see as contention.
    """
    return staging_dir(db_path).is_dir() and is_held(db_path)
