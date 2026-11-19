import errno
import fcntl
import json
import os
import re
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from fastembed import TextEmbedding

import minirag_mcp.embedder as embedder_mod
import minirag_mcp.identity as identity_mod
from conftest import stamp_identity
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.config import default_db_path, load_config
from minirag_mcp.embedder import E5_LARGE, MINILM
from minirag_mcp.endpoints import Endpoint
from minirag_mcp.identity import (
    FORMAT,
    INDEX_FILE,
    ChunkingScheme,
    EmbeddingScheme,
    IdentityError,
    IndexIdentity,
    IndexState,
    LegacyIndexError,
    RebuildInfo,
    create_identity,
    differences,
    index_state,
    is_legacy_index,
    legacy_error,
    previous_dir,
    read_identity,
    rebuild_kind,
    rebuild_running,
    recover_swap,
    staging_dir,
    swap_in,
    update_identity,
    write_identity,
)
from minirag_mcp.lock import sync_lock
from minirag_mcp.store import IndexOpenError, Store

EMB = EmbeddingScheme(E5_LARGE, 1024, "e5")
PROBE = tuple((i - 512) / 1024 for i in range(1024))
CHUNK = ChunkingScheme(SCHEME_VERSION, 110)
IDENT = IndexIdentity(
    format=FORMAT,
    minirag="2.0.0a1",
    generation=1,
    embedding=EMB,
    probe=PROBE,
    chunking=CHUNK,
    fts_language="Russian",
    document_table=True,
)
LAYOUT = {
    "format": 2,
    "minirag": "2.0.0a1",
    "generation": 1,
    "embedding": {"model": E5_LARGE, "dim": 1024, "prefixes": "e5", "probe": list(PROBE)},
    "chunking": {"scheme_version": SCHEME_VERSION, "token_budget": 110},
    "fts_language": "Russian",
    "document_table": True,
}
UNLOADABLE = (
    "this index was built with {model}, which cannot be loaded here: no embedding endpoint "
    "is configured for it and fastembed does not know it"
)


class RegistryOnly:
    """fastembed's registry without its models: building one is a test failure."""

    list_supported_models = staticmethod(TextEmbedding.list_supported_models)

    def __init__(self, *args, **kwargs) -> None:
        raise AssertionError("index_state built an embedding model")


@pytest.fixture
def registry_only(monkeypatch) -> None:
    monkeypatch.setattr(embedder_mod, "TextEmbedding", RegistryOnly)


def config(tmp_path: Path, **env: str):
    return load_config({"BASE_DIR": str(tmp_path), **env}, cwd=tmp_path)


def make_index(path: Path, generation: int, marker: str) -> None:
    """An index folder told apart from the others by `marker`."""
    write_identity(path, replace(IDENT, generation=generation))
    (path / "marker").write_text(marker, encoding="utf-8")


def marker(path: Path) -> str:
    return (path / "marker").read_text(encoding="utf-8")


def generation(path: Path) -> int:
    ident = read_identity(path)
    assert ident is not None, path
    return ident.generation


def test_identity_round_trips_in_the_spec_layout(tmp_path):
    db = tmp_path / "db"
    write_identity(db, IDENT)
    assert json.loads((db / INDEX_FILE).read_text(encoding="utf-8")) == LAYOUT
    assert read_identity(db) == IDENT


def test_a_missing_file_reads_as_none(tmp_path):
    assert read_identity(tmp_path / "nowhere") is None
    assert read_identity(tmp_path) is None


@pytest.mark.parametrize(
    ("raw", "says"),
    [
        ("{", "not valid JSON"),
        ("[]", "not a JSON object"),
        ('{"format": 2}', "minirag is missing"),
        (
            json.dumps({**LAYOUT, "embedding": {**LAYOUT["embedding"], "dim": "1024"}}),
            "embedding.dim is not an integer",
        ),
        (
            json.dumps({**LAYOUT, "embedding": {**LAYOUT["embedding"], "probe": [0.1, "x"]}}),
            "probe",
        ),
        (json.dumps({**LAYOUT, "generation": True}), "generation is not an integer"),
        (json.dumps({**LAYOUT, "format": 3}), "index format 3 by minirag 2.0.0a1, newer than"),
    ],
)
def test_an_unreadable_or_newer_identity_is_an_error(tmp_path, raw, says):
    (tmp_path / INDEX_FILE).write_text(raw, encoding="utf-8")
    with pytest.raises(IdentityError, match=re.escape(str(tmp_path / INDEX_FILE))) as excinfo:
        read_identity(tmp_path)
    assert says in str(excinfo.value)


def test_an_index_file_that_cannot_be_read_is_an_error(tmp_path):
    (tmp_path / INDEX_FILE).mkdir()
    with pytest.raises(IdentityError, match="cannot be read"):
        read_identity(tmp_path)


def test_a_file_as_db_path_is_an_index_open_error(tmp_path, fake_embedder):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(IndexOpenError, match="DB_PATH"):
        read_identity(blocker)
    with pytest.raises(IndexOpenError, match="DB_PATH"):
        write_identity(blocker, IDENT)
    assert blocker.read_text(encoding="utf-8") == "x"
    assert not is_legacy_index(blocker)
    state, why = index_state(config(tmp_path, DB_PATH=str(blocker)), fake_embedder)
    assert state is IndexState.UNAVAILABLE and "DB_PATH" in why


def test_a_failed_write_leaves_the_old_file(tmp_path, monkeypatch):
    db = tmp_path / "db"
    write_identity(db, IDENT)
    before = (db / INDEX_FILE).read_bytes()

    def no_space(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(identity_mod.os, "replace", no_space)
    with pytest.raises(OSError, match="No space"):
        write_identity(db, replace(IDENT, generation=2))
    assert (db / INDEX_FILE).read_bytes() == before
    assert [p.name for p in db.iterdir()] == [INDEX_FILE]


def test_concurrent_writers_never_share_a_temp_file(tmp_path, monkeypatch):
    db = tmp_path / "db"
    write_identity(db, IDENT)
    both_writing = threading.Barrier(2, timeout=10)
    real_fsync = os.fsync

    def fsync_once_both_are_writing(fd):
        both_writing.wait()
        real_fsync(fd)

    monkeypatch.setattr(identity_mod.os, "fsync", fsync_once_both_are_writing)
    candidates = [replace(IDENT, generation=2), replace(IDENT, generation=3)]
    errors: list[BaseException] = []

    def write(identity: IndexIdentity) -> None:
        try:
            write_identity(db, identity)
        except BaseException as e:
            errors.append(e)

    writers = [threading.Thread(target=write, args=(c,)) for c in candidates]
    for w in writers:
        w.start()
    for w in writers:
        w.join()
    assert errors == []
    assert read_identity(db) in candidates
    assert [p.name for p in db.iterdir()] == [INDEX_FILE]


def test_create_identity_stamps_a_new_folder_and_never_overwrites(tmp_path):
    db = tmp_path / "db"
    fresh = IndexIdentity.fresh(EMB, CHUNK, "English")
    assert create_identity(db, fresh) == fresh
    assert read_identity(db) == fresh
    write_identity(db, IDENT)
    recorded = (db / INDEX_FILE).read_bytes()
    assert create_identity(db, fresh) == IDENT
    assert (db / INDEX_FILE).read_bytes() == recorded
    assert [p.name for p in db.iterdir()] == [INDEX_FILE]


def test_create_identity_without_hard_links_still_never_overwrites(tmp_path, monkeypatch):
    def no_hard_links(src, dst):
        raise OSError(errno.ENOTSUP, "Operation not supported")

    monkeypatch.setattr(identity_mod.os, "link", no_hard_links)
    db = tmp_path / "db"
    fresh = IndexIdentity.fresh(EMB, CHUNK, "English")
    assert create_identity(db, fresh) == fresh
    assert read_identity(db) == fresh
    assert create_identity(db, IDENT) == fresh
    assert read_identity(db) == fresh
    assert [p.name for p in db.iterdir()] == [INDEX_FILE]


def test_create_identity_refuses_an_unreadable_entry_in_the_way(tmp_path):
    db = tmp_path / "db"
    db.mkdir()
    (db / INDEX_FILE).symlink_to(tmp_path / "nowhere")
    with pytest.raises(IdentityError, match="holds nothing readable"):
        create_identity(db, IndexIdentity.fresh(EMB, CHUNK, "English"))
    assert (db / INDEX_FILE).is_symlink()
    assert sorted(p.name for p in db.iterdir()) == [INDEX_FILE]


def test_a_stamp_racing_a_swap_never_replaces_the_index(tmp_path, monkeypatch):
    """Another process found no identity at db, because a swap had just moved it away."""
    db = tmp_path / "db"
    make_index(db, 1, "serving")
    make_index(staging_dir(db), 1, "rebuilt")
    fresh = IndexIdentity.fresh(EMB, CHUNK, "English")
    b_found_nothing = threading.Event()
    stamped: list[IndexIdentity | BaseException] = []
    real_read, real_rename = identity_mod.read_identity, os.rename

    def read(path: Path) -> IndexIdentity | None:
        found = real_read(path)
        if threading.current_thread().name == "B" and path == db and found is None:
            b_found_nothing.set()
        return found

    def stamp() -> None:
        try:
            stamped.append(create_identity(db, fresh))
        except BaseException as e:
            stamped.append(e)
        finally:
            b_found_nothing.set()

    b = threading.Thread(target=stamp, name="B")

    def start_b_between_the_renames(src, dst):
        if Path(src) == staging_dir(db) and b.ident is None:
            b.start()
            b_found_nothing.wait(10)
        real_rename(src, dst)

    monkeypatch.setattr(identity_mod, "read_identity", read)
    monkeypatch.setattr(identity_mod.os, "rename", start_b_between_the_renames)
    swap_in(db)
    b.join(10)
    assert (generation(db), marker(db)) == (2, "rebuilt")
    assert stamped == [read_identity(db)]
    assert (generation(previous_dir(db)), marker(previous_dir(db))) == (1, "serving")


def test_a_stamp_that_passed_its_checks_before_a_swap_cannot_land_inside_it(tmp_path, monkeypatch):
    """B is about to create its file when A moves the index away, then B links it."""
    db = tmp_path / "db"
    make_index(db, 1, "serving")
    make_index(staging_dir(db), 1, "rebuilt")
    fresh = IndexIdentity.fresh(EMB, CHUNK, "English")
    b_writing, b_linked, a_moved_db = threading.Event(), threading.Event(), threading.Event()
    outcome: dict[str, object] = {}
    real_temp, real_link, real_rename = identity_mod._temp_file, os.link, os.rename

    def temp(path, identity):
        if threading.current_thread().name == "B":
            b_writing.set()
            a_moved_db.wait(2)
        return real_temp(path, identity)

    def link(src, dst):
        try:
            return real_link(src, dst)
        finally:
            if threading.current_thread().name == "B":
                b_linked.set()

    def rename(src, dst):
        if threading.current_thread().name == "A" and Path(src) == db:
            b_writing.wait(10)
            real_rename(src, dst)
            a_moved_db.set()
            return
        if threading.current_thread().name == "A" and Path(src) == staging_dir(db):
            b_linked.wait(10)
        real_rename(src, dst)

    def run(name: str, call) -> None:
        try:
            outcome[name] = call()
        except BaseException as e:
            outcome[name] = e
        finally:
            if name == "B":
                b_writing.set()
                b_linked.set()

    monkeypatch.setattr(identity_mod, "_temp_file", temp)
    monkeypatch.setattr(identity_mod.os, "link", link)
    monkeypatch.setattr(identity_mod.os, "rename", rename)
    threads = [
        threading.Thread(target=run, args=("B", lambda: create_identity(db, fresh)), name="B"),
        threading.Thread(target=run, args=("A", lambda: swap_in(db)), name="A"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert outcome["A"] is None
    assert outcome["B"] in (replace(IDENT, generation=1), replace(IDENT, generation=2))
    assert (generation(db), marker(db)) == (2, "rebuilt")
    assert (generation(previous_dir(db)), marker(previous_dir(db))) == (1, "serving")
    assert not staging_dir(db).exists()


def _lock_taken(path: Path) -> bool:
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def test_the_identity_lock_sits_beside_the_index_and_covers_every_change(tmp_path, monkeypatch):
    db = tmp_path / "db"
    lock_file = tmp_path / ".db.identity.lock"
    seen: list[tuple[str, bool]] = []
    real_link, real_rename = os.link, os.rename

    def link(src, dst):
        seen.append(("link", _lock_taken(lock_file)))
        real_link(src, dst)

    def rename(src, dst):
        seen.append((f"{Path(src).name} -> {Path(dst).name}", _lock_taken(lock_file)))
        real_rename(src, dst)

    monkeypatch.setattr(identity_mod.os, "link", link)
    monkeypatch.setattr(identity_mod.os, "rename", rename)
    create_identity(db, replace(IDENT, generation=1))
    make_index(staging_dir(db), 1, "rebuilt")
    swap_in(db)
    assert lock_file.is_file() and not list(db.glob("*.lock"))
    # A swap that died between its renames, finished by recover_swap.
    make_index(staging_dir(db), 2, "rebuilt again")
    shutil.rmtree(previous_dir(db))
    real_rename(db, previous_dir(db))
    assert recover_swap(db) is True
    assert seen == [
        ("link", True),
        ("db -> db.previous", True),
        ("db.next -> db", True),
        ("db.next -> db", True),
    ]
    assert lock_file.is_file()


def test_an_update_lands_only_on_the_generation_it_was_read_from(tmp_path, monkeypatch):
    """A swapped-in identity is never overwritten with fields of the one it replaced."""
    db = tmp_path / "db"
    lock_file = tmp_path / ".db.identity.lock"
    write_identity(db, IDENT)
    locked: list[bool] = []
    real_replace = os.replace

    def spy(src, dst):
        locked.append(_lock_taken(lock_file))
        real_replace(src, dst)

    monkeypatch.setattr(identity_mod.os, "replace", spy)
    assert update_identity(db, 1, fts_language="English") == replace(IDENT, fts_language="English")
    assert read_identity(db) == replace(IDENT, fts_language="English")
    assert locked == [True]
    write_identity(db, replace(IDENT, generation=2))
    assert update_identity(db, 1, fts_language="German") is None
    assert read_identity(db) == replace(IDENT, generation=2)
    shutil.rmtree(db)
    assert update_identity(db, 2, fts_language="German") is None
    assert not db.exists()


def test_an_existing_identity_is_returned_without_taking_the_lock(tmp_path):
    db = tmp_path / "db"
    write_identity(db, IDENT)
    assert create_identity(db, IndexIdentity.fresh(EMB, CHUNK, "English")) == IDENT
    assert not (tmp_path / ".db.identity.lock").exists()


def test_an_identity_lock_that_cannot_be_taken_warns_and_proceeds(tmp_path, monkeypatch):
    fresh = IndexIdentity.fresh(EMB, CHUNK, "English")
    (tmp_path / ".db.identity.lock").mkdir()
    with pytest.warns(RuntimeWarning, match=re.escape(str(tmp_path / ".db.identity.lock"))):
        assert create_identity(tmp_path / "db", fresh) == fresh
    monkeypatch.setattr(identity_mod, "fcntl", None)
    with pytest.warns(RuntimeWarning, match="fcntl is unavailable"):
        assert create_identity(tmp_path / "other", fresh) == fresh


def test_no_fresh_stamp_while_a_swap_may_be_under_way(tmp_path):
    db = tmp_path / "db"
    db.mkdir()
    make_index(previous_dir(db), 1, "serving")
    make_index(staging_dir(db), 2, "rebuilt")
    with pytest.raises(IdentityError, match=re.escape(str(staging_dir(db)))):
        create_identity(db, IndexIdentity.fresh(EMB, CHUNK, "English"))
    assert list(db.iterdir()) == []


def test_a_lancedb_table_without_identity_is_legacy(tmp_path):
    Store(tmp_path / "db", dim=8)
    assert is_legacy_index(tmp_path / "db")
    write_identity(tmp_path / "db", IDENT)
    assert not is_legacy_index(tmp_path / "db")
    assert not is_legacy_index(tmp_path / "empty")


def test_differences_name_what_changed():
    assert differences(IDENT, EMB, CHUNK) == []
    assert differences(IDENT, EmbeddingScheme(MINILM, 384, "none"), CHUNK) == [
        f"model {E5_LARGE} -> {MINILM}",
        "query/passage prefixes e5 -> none",
    ]
    assert differences(IDENT, replace(EMB, dim=512), CHUNK) == ["vector width 1024 -> 512"]
    assert differences(IDENT, replace(EMB, model="x/other", dim=512), CHUNK) == [
        f"model {E5_LARGE} -> x/other"
    ]
    bigger = CHUNK.token_budget + 90
    assert differences(IDENT, EMB, replace(CHUNK, token_budget=bigger)) == [
        f"chunk budget {CHUNK.token_budget} -> {bigger} tokens"
    ]
    assert differences(IDENT, EMB, replace(CHUNK, scheme_version=SCHEME_VERSION + 1)) == [
        f"chunking scheme {SCHEME_VERSION} -> {SCHEME_VERSION + 1}"
    ]


def test_rebuild_kind():
    other_emb = replace(EMB, prefixes="none")
    other_chunk = replace(CHUNK, token_budget=CHUNK.token_budget + 1)
    assert rebuild_kind(IDENT, EMB, CHUNK) == "none"
    assert rebuild_kind(IDENT, other_emb, CHUNK) == "reembed"
    assert rebuild_kind(IDENT, replace(EMB, dim=512), CHUNK) == "reembed"
    assert rebuild_kind(IDENT, EMB, other_chunk) == "reingest"
    assert rebuild_kind(IDENT, other_emb, other_chunk) == "reingest"


def test_rebuild_info_to_dict():
    info = RebuildInfo("reembed", (f"model {MINILM} -> {E5_LARGE}",), 2, 40, 1)
    assert info.to_dict() == {
        "kind": "reembed",
        "reasons": [f"model {MINILM} -> {E5_LARGE}"],
        "generation": 2,
        "reembedded": 40,
        "restored": 1,
    }


def test_index_state_reads_files_and_never_loads_a_model(tmp_path, fake_embedder, registry_only):
    cfg = config(tmp_path)
    assert index_state(cfg, fake_embedder) == (IndexState.READY, "")
    assert not cfg.db_path.exists()
    stamp_identity(cfg, fake_embedder)
    assert index_state(cfg, fake_embedder) == (IndexState.READY, "")
    bigger = cfg.token_budget + 90
    assert index_state(replace(cfg, token_budget=bigger), fake_embedder) == (
        IndexState.REBUILD_PENDING,
        f"chunk budget {cfg.token_budget} -> {bigger} tokens",
    )
    other_language = "Russian" if cfg.fts_language != "Russian" else "English"
    assert index_state(replace(cfg, fts_language=other_language), fake_embedder) == (
        IndexState.READY,
        "",
    )


def test_stamp_identity_writes_a_fresh_identity(tmp_path, fake_embedder):
    cfg = config(tmp_path)
    written = stamp_identity(cfg, fake_embedder)
    assert read_identity(cfg.db_path) == written
    assert written == IndexIdentity(
        format=FORMAT,
        minirag=written.minirag,
        generation=1,
        embedding=EmbeddingScheme("fake", 8, "none"),
        probe=(),
        chunking=ChunkingScheme(SCHEME_VERSION, cfg.token_budget),
        fts_language=cfg.fts_language,
        document_table=True,
    )


def test_index_state_refuses_a_legacy_index_naming_the_v2_path(tmp_path, fake_embedder):
    db = tmp_path / "old"
    Store(db, dim=8)
    cfg = config(tmp_path, DB_PATH=str(db))
    state, why = index_state(cfg, fake_embedder)
    v2 = cfg.roots[0] / ".minirag" / "v2"
    assert state is IndexState.UNAVAILABLE
    assert why == str(legacy_error(db, cfg.roots))
    assert why == (
        f"{db} holds a minirag 1.x index (a LanceDB table and no {INDEX_FILE}). minirag 2 never "
        f"writes to a 1.x index. Leave DB_PATH unset to use {v2}, or point it at an empty folder."
    )
    assert default_db_path(cfg.roots) == v2
    assert not (db / INDEX_FILE).exists()


def test_index_state_needs_a_way_to_load_the_recorded_model(tmp_path, fake_embedder, registry_only):
    cfg = config(tmp_path)
    fresh = IndexIdentity.fresh(
        EmbeddingScheme("x/unknown", 8, "none"),
        ChunkingScheme(SCHEME_VERSION, cfg.token_budget),
        cfg.fts_language,
    )
    write_identity(cfg.db_path, fresh)
    assert index_state(cfg, fake_embedder) == (
        IndexState.UNAVAILABLE,
        UNLOADABLE.format(model="x/unknown"),
    )
    routed = replace(cfg, embed_endpoints={"x/unknown": Endpoint("http://127.0.0.1:9", "x")})
    assert index_state(routed, fake_embedder) == (
        IndexState.REBUILD_PENDING,
        "model x/unknown -> fake",
    )
    write_identity(cfg.db_path, replace(fresh, embedding=EmbeddingScheme(MINILM, 384, "none")))
    assert index_state(cfg, fake_embedder) == (
        IndexState.REBUILD_PENDING,
        f"model {MINILM} -> fake",
    )


def test_index_state_reports_an_unreadable_identity(tmp_path, fake_embedder):
    cfg = config(tmp_path)
    cfg.db_path.mkdir(parents=True)
    (cfg.db_path / INDEX_FILE).write_text("{", encoding="utf-8")
    state, why = index_state(cfg, fake_embedder)
    assert state is IndexState.UNAVAILABLE and str(cfg.db_path / INDEX_FILE) in why


def test_staging_and_previous_sit_beside_the_index(tmp_path):
    assert staging_dir(tmp_path / "v2") == tmp_path / "v2.next"
    assert previous_dir(tmp_path / "v2") == tmp_path / "v2.previous"


def test_swap_in_bumps_generation_and_keeps_one_previous(tmp_path):
    db = tmp_path / "db"
    make_index(db, 1, "first")
    make_index(staging_dir(db), 1, "second")
    swap_in(db)
    assert (generation(db), marker(db)) == (2, "second")
    assert (generation(previous_dir(db)), marker(previous_dir(db))) == (1, "first")
    assert not staging_dir(db).exists()

    make_index(staging_dir(db), 2, "third")
    swap_in(db)
    assert (generation(db), marker(db)) == (3, "third")
    assert (generation(previous_dir(db)), marker(previous_dir(db))) == (2, "second")
    assert sorted(p.name for p in tmp_path.iterdir()) == [".db.identity.lock", "db", "db.previous"]

    make_index(staging_dir(db), 7, "fourth")
    swap_in(db)
    assert generation(db) == 8


def test_swap_in_refuses_a_staging_without_identity(tmp_path):
    db = tmp_path / "db"
    make_index(db, 1, "serving")
    staging_dir(db).mkdir()
    (staging_dir(db) / "marker").write_text("half-built", encoding="utf-8")
    with pytest.raises(IdentityError, match=re.escape(str(staging_dir(db)))):
        swap_in(db)
    assert (generation(db), marker(db)) == (1, "serving")
    assert marker(staging_dir(db)) == "half-built"
    assert not previous_dir(db).exists()


def test_a_failed_second_rename_puts_the_index_back(tmp_path, monkeypatch):
    db = tmp_path / "db"
    make_index(db, 1, "serving")
    make_index(staging_dir(db), 1, "rebuilt")
    real_rename = os.rename

    def refuse_the_staging(src, dst):
        if Path(src) == staging_dir(db):
            raise OSError(errno.EACCES, "Permission denied")
        real_rename(src, dst)

    monkeypatch.setattr(identity_mod.os, "rename", refuse_the_staging)
    with pytest.raises(OSError, match="Permission denied"):
        swap_in(db)
    assert (generation(db), marker(db)) == (1, "serving")
    assert marker(staging_dir(db)) == "rebuilt"
    assert not previous_dir(db).exists()


def test_a_failed_rollback_names_every_folder_and_the_swap_is_finished_later(tmp_path, monkeypatch):
    db = tmp_path / "db"
    make_index(db, 1, "serving")
    make_index(staging_dir(db), 1, "rebuilt")
    real_rename = os.rename

    def only_the_first_rename(src, dst):
        if Path(dst) == db:
            raise OSError(errno.EACCES, "Permission denied")
        real_rename(src, dst)

    monkeypatch.setattr(identity_mod.os, "rename", only_the_first_rename)
    with pytest.raises(IndexOpenError) as excinfo:
        swap_in(db)
    for folder in (db, previous_dir(db), staging_dir(db)):
        assert str(folder) in str(excinfo.value)
    assert f"{db} is missing" in str(excinfo.value)
    assert "The next open finishes the swap" in str(excinfo.value)
    monkeypatch.setattr(identity_mod.os, "rename", real_rename)
    assert recover_swap(db) is True
    assert (generation(db), marker(db)) == (2, "rebuilt")
    assert marker(previous_dir(db)) == "serving"


def test_a_failed_rollback_onto_a_recreated_index_folder_says_so(tmp_path, monkeypatch):
    """Something made db between the renames: neither rename can land, and recovery cannot."""
    db = tmp_path / "db"
    make_index(db, 1, "serving")
    make_index(staging_dir(db), 1, "rebuilt")
    real_rename = os.rename

    def db_reappears(src, dst):
        if Path(src) == staging_dir(db):
            db.mkdir()
            (db / "intruder").write_text("x", encoding="utf-8")
        real_rename(src, dst)

    monkeypatch.setattr(identity_mod.os, "rename", db_reappears)
    with pytest.raises(IndexOpenError) as excinfo:
        swap_in(db)
    message = str(excinfo.value)
    for folder in (db, previous_dir(db), staging_dir(db)):
        assert str(folder) in message
    assert "is missing" not in message and "next open" not in message
    assert f"{db} was created again" in message
    assert recover_swap(db) is False


def test_swap_in_never_deletes_a_previous_that_is_not_an_index(tmp_path):
    db = tmp_path / "db"
    make_index(db, 1, "serving")
    make_index(staging_dir(db), 1, "rebuilt")
    previous_dir(db).mkdir()
    (previous_dir(db) / "notes.txt").write_text("the user's", encoding="utf-8")
    with pytest.raises(IdentityError, match=re.escape(str(previous_dir(db)))):
        swap_in(db)
    assert (previous_dir(db) / "notes.txt").read_text(encoding="utf-8") == "the user's"
    assert (generation(db), marker(db)) == (1, "serving")
    assert (generation(staging_dir(db)), marker(staging_dir(db))) == (1, "rebuilt")


def test_swap_in_refuses_a_1x_index(tmp_path):
    db = tmp_path / "db"
    Store(db, dim=8)
    make_index(staging_dir(db), 1, "rebuilt")
    with pytest.raises(LegacyIndexError, match=re.escape(str(db))):
        swap_in(db)
    assert is_legacy_index(db)
    assert not previous_dir(db).exists()
    assert marker(staging_dir(db)) == "rebuilt"


def test_a_swap_finished_by_another_process_is_not_a_failure(tmp_path, monkeypatch):
    """Another process's recover_swap can move the staging in between our two renames."""
    db = tmp_path / "db"
    make_index(db, 1, "serving")
    make_index(staging_dir(db), 1, "rebuilt")
    real_rename = os.rename

    def beaten_to_it(src, dst):
        real_rename(src, dst)
        if Path(src) == staging_dir(db):
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(src))

    monkeypatch.setattr(identity_mod.os, "rename", beaten_to_it)
    swap_in(db)
    assert (generation(db), marker(db)) == (2, "rebuilt")
    assert (generation(previous_dir(db)), marker(previous_dir(db))) == (1, "serving")


class Killed(BaseException):
    """The process dying: nothing after it runs, no handler sees it."""


def test_recover_swap_finishes_an_interrupted_swap(tmp_path, monkeypatch):
    db = tmp_path / "db"
    make_index(db, 1, "serving")
    make_index(staging_dir(db), 1, "rebuilt")
    real_rename = os.rename

    def killed_before_the_second(src, dst):
        if Path(src) == staging_dir(db):
            raise Killed
        real_rename(src, dst)

    monkeypatch.setattr(identity_mod.os, "rename", killed_before_the_second)
    with pytest.raises(Killed):
        swap_in(db)
    monkeypatch.setattr(identity_mod.os, "rename", real_rename)
    assert not db.exists() and previous_dir(db).is_dir() and staging_dir(db).is_dir()

    assert recover_swap(db) is True
    assert (generation(db), marker(db)) == (2, "rebuilt")
    assert (generation(previous_dir(db)), marker(previous_dir(db))) == (1, "serving")
    assert not staging_dir(db).exists()
    assert recover_swap(db) is False


def test_index_state_finishes_an_interrupted_swap_before_reading(tmp_path, fake_embedder):
    cfg = config(tmp_path)
    db = cfg.db_path
    stamped = stamp_identity(replace(cfg, db_path=previous_dir(db)), fake_embedder)
    write_identity(staging_dir(db), replace(stamped, generation=2))
    assert index_state(cfg, fake_embedder) == (IndexState.READY, "")
    assert generation(db) == 2 and not staging_dir(db).exists()


def test_an_unfinishable_swap_is_unavailable_never_a_fresh_index(
    tmp_path, monkeypatch, fake_embedder
):
    cfg = config(tmp_path)
    db = cfg.db_path
    stamped = stamp_identity(replace(cfg, db_path=previous_dir(db)), fake_embedder)
    write_identity(staging_dir(db), replace(stamped, generation=2))

    def refused(src, dst):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(identity_mod.os, "rename", refused)
    with pytest.raises(IndexOpenError, match=re.escape(str(staging_dir(db)))):
        recover_swap(db)
    state, why = index_state(cfg, fake_embedder)
    assert state is IndexState.UNAVAILABLE and str(staging_dir(db)) in why
    assert not db.exists()


def test_recover_swap_after_another_process_finished_it(tmp_path, monkeypatch):
    db = tmp_path / "db"
    make_index(previous_dir(db), 1, "serving")
    make_index(staging_dir(db), 2, "rebuilt")
    real_rename = os.rename

    def beaten_to_it(src, dst):
        real_rename(src, dst)
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(src))

    monkeypatch.setattr(identity_mod.os, "rename", beaten_to_it)
    assert recover_swap(db) is False
    assert (generation(db), marker(db)) == (2, "rebuilt")


def test_rebuild_running_needs_both_the_staging_dir_and_the_lock(tmp_path, monkeypatch):
    db = tmp_path / "db"
    db.mkdir()
    probed: list[Path] = []
    real_is_held = identity_mod.is_held

    def recording(path: Path) -> bool:
        probed.append(path)
        return real_is_held(path)

    monkeypatch.setattr(identity_mod, "is_held", recording)
    assert not rebuild_running(db)
    with sync_lock(db):
        assert not rebuild_running(db)
        # Probing takes the lock for an instant; without a staging dir no write may do it.
        assert probed == []
        staging_dir(db).mkdir()
        assert rebuild_running(db)
    assert not rebuild_running(db)
    assert probed == [db, db]


@pytest.mark.parametrize("probe", [[0.5] * 3, [0.5] * 1025])
def test_a_probe_of_the_wrong_width_is_malformed(tmp_path, probe):
    layout = json.loads(json.dumps(LAYOUT))
    layout["embedding"]["probe"] = probe
    (tmp_path / INDEX_FILE).write_text(json.dumps(layout), encoding="utf-8")
    with pytest.raises(IdentityError, match="embedding.probe has"):
        read_identity(tmp_path)
