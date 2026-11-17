import pytest

from minirag_mcp.config import ConfigError, load_config
from minirag_mcp.results import fts_status
from minirag_mcp.store import FTS_LANGUAGE_FILENAME, ChunkRecord, Store


def rec(source, i, text):
    return ChunkRecord(
        id=f"{source}#{i}",
        source=source,
        source_type="file",
        title="T",
        chunk_index=i,
        text=text,
        vector=[0.1] * 8,
        file_hash="h",
        mtime=1.0,
        ingested_at="2026-08-07T00:00:00+00:00",
    )


CONTRACT = "Подписание договора с поставщиком"


def _fts_ids(store, query: str) -> list[str]:
    rows = store._table.search(query, query_type="fts", fts_columns="text").to_list()
    return [r["id"] for r in rows]


def _english_store_with_a_contract(db):
    s = Store(db, dim=8)
    s.replace_source("/c.md", [rec("/c.md", 0, CONTRACT)])
    # The English stemmer leaves Cyrillic alone, so another case of the word is another word.
    assert _fts_ids(s, "договорами") == []
    return s


def test_fts_language_defaults_to_english(tmp_path):
    assert load_config({}, cwd=tmp_path).fts_language == "English"


def test_fts_language_is_matched_case_insensitively(tmp_path):
    cfg = load_config({"RAG_FTS_LANGUAGE": " russian "}, cwd=tmp_path)
    assert cfg.fts_language == "Russian"  # the only spelling LanceDB accepts


def test_unknown_fts_language_is_rejected_with_the_valid_ones(tmp_path):
    with pytest.raises(ConfigError, match="Russian") as e:
        load_config({"RAG_FTS_LANGUAGE": "Ukrainian"}, cwd=tmp_path)
    assert "Ukrainian" in str(e.value)


def test_a_new_index_stems_in_the_configured_language(tmp_path):
    s = Store(tmp_path / "db", dim=8, fts_language="Russian")
    s.replace_source("/c.md", [rec("/c.md", 0, CONTRACT)])
    assert _fts_ids(s, "договорами") == ["/c.md#0"]
    assert s.fts_index_language == "Russian"


def test_changing_the_language_rebuilds_an_existing_index(tmp_path):
    """Without the rebuild, setting RAG_FTS_LANGUAGE on an index that already exists
    would change nothing: minirag only ever creates an FTS index that is missing."""
    db = tmp_path / "db"
    _english_store_with_a_contract(db)

    s2 = Store(db, dim=8, fts_language="Russian")
    assert _fts_ids(s2, "договорами") == ["/c.md#0"]
    assert s2.fts_index_language == "Russian"
    assert s2.chunk_count() == 1
    assert (db / FTS_LANGUAGE_FILENAME).read_text(encoding="utf-8").strip() == "Russian"


def test_an_index_with_no_record_is_taken_as_english(tmp_path):
    """Indexes written before the setting existed were built by FTS(), i.e. English."""
    db = tmp_path / "db"
    _english_store_with_a_contract(db)
    (db / FTS_LANGUAGE_FILENAME).unlink()

    s2 = Store(db, dim=8, fts_language="Russian")
    assert _fts_ids(s2, "договорами") == ["/c.md#0"]


def test_reopening_in_the_same_language_does_not_rebuild(tmp_path, monkeypatch):
    db = tmp_path / "db"
    s1 = Store(db, dim=8, fts_language="Russian")
    s1.replace_source("/c.md", [rec("/c.md", 0, CONTRACT)])
    calls = []
    real = type(s1._table).create_index

    def counting(self, *args, **kwargs):
        calls.append(args)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(type(s1._table), "create_index", counting)
    Store(db, dim=8, fts_language="Russian")
    assert calls == []


def test_a_failed_rebuild_keeps_the_old_index_and_says_so(tmp_path, monkeypatch):
    db = tmp_path / "db"
    s1 = _english_store_with_a_contract(db)

    def boom(*args, **kwargs):
        raise RuntimeError("Retryable commit conflict for version 65")

    monkeypatch.setattr(type(s1._table), "create_index", boom)
    with pytest.warns(RuntimeWarning, match="RAG_FTS_LANGUAGE"):
        s2 = Store(db, dim=8, fts_language="Russian")
    assert s2.fts_index_language == "English"
    assert s2.missing_fts_indices == ()
    assert _fts_ids(s2, "договора") == ["/c.md#0"]  # the English index still answers
    status = fts_status(s2)
    assert status["ftsLanguage"] == "Russian"
    assert "English" in status["ftsWarning"]

    monkeypatch.undo()  # the record was not advanced, so the next open retries
    s3 = Store(db, dim=8, fts_language="Russian")
    assert _fts_ids(s3, "договорами") == ["/c.md#0"]
    assert "ftsWarning" not in fts_status(s3)
