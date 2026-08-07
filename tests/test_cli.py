import json
from pathlib import Path

import pytest

import minirag_mcp.cli as cli

# Captured at import time, before the autouse fixture below ever patches
# cli._make_embedder — lets tests that need the real factory restore it.
_REAL_MAKE_EMBEDDER = cli._make_embedder


@pytest.fixture(autouse=True)
def fake_model(monkeypatch, fake_embedder):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)


def run(tokens):
    return cli.app(tokens, result_action="return_value", exit_on_error=False)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha body about tokens and auth.")
    (tmp_path / "sub" / "b.md").write_text("# Beta\n\nERR_CONNECTION_REFUSED appears here.")
    return tmp_path


def test_ingest_directory_recursive_and_query(corpus, capsys):
    run(["ingest", str(corpus), "--base-dir", str(corpus), "--json"])
    ingested = json.loads(capsys.readouterr().out)["ingested"]
    assert sorted(Path(i["source"]).name for i in ingested) == ["a.md", "b.md"]

    run(["query", "ERR_CONNECTION_REFUSED", "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"], "expected results"


def test_ingest_single_file(corpus, capsys):
    run(["ingest", str(corpus / "a.md"), "--base-dir", str(corpus)])
    assert "a.md" in capsys.readouterr().out


def test_list_and_status_and_delete(corpus, capsys):
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["list", "--base-dir", str(corpus), "--json"])
    files = json.loads(capsys.readouterr().out)["files"]
    assert len(files) == 2 and all(f["state"] == "ingested" for f in files)

    run(["status", "--base-dir", str(corpus), "--json"])
    st = json.loads(capsys.readouterr().out)
    assert st["chunkCount"] >= 2

    run(["delete", str(corpus / "a.md"), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["list", "--base-dir", str(corpus), "--json"])
    files = json.loads(capsys.readouterr().out)["files"]
    by_source = {f["source"]: f["state"] for f in files}
    assert by_source[str(corpus / "a.md")] == "not_ingested"  # still on disk, gone from index


def test_read_full_source(corpus, capsys):
    run(["ingest", str(corpus / "a.md"), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["read", str(corpus / "a.md"), "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "Alpha body" in payload["text"] and payload["chunkCount"] >= 1


def test_sync_and_read_neighbors(corpus, capsys):
    run(["sync", "--base-dir", str(corpus), "--json"])
    counts = json.loads(capsys.readouterr().out)["counts"]
    assert counts["ingested"] == 2 and counts["failed"] == 0

    run(
        [
            "read-neighbors",
            "--file-path",
            str(corpus / "a.md"),
            "--chunk-index",
            "0",
            "--base-dir",
            str(corpus),
            "--json",
        ]
    )
    assert json.loads(capsys.readouterr().out)["chunks"]


def test_ingest_url_mocked(corpus, capsys, monkeypatch, public_dns):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url: ParsedDoc("# R\n\nRemote body.", "R"))
    run(["ingest-url", "https://example.com/p", "--base-dir", str(corpus)])
    assert "example.com" in capsys.readouterr().out


def test_list_scope_stops_at_a_path_component_boundary(tmp_path, capsys):
    (tmp_path / "proj").mkdir()
    (tmp_path / "project-secret").mkdir()
    (tmp_path / "proj" / "a.md").write_text("# A\n\nProject body long enough to index.")
    (tmp_path / "project-secret" / "b.md").write_text("# B\n\nSecret body long enough to index.")
    run(["ingest", str(tmp_path / "project-secret"), "--base-dir", str(tmp_path)])
    capsys.readouterr()
    run(["list", "--base-dir", str(tmp_path), "--scope", str(tmp_path / "proj"), "--json"])
    files = json.loads(capsys.readouterr().out)["files"]
    assert [f["source"] for f in files] == [str(tmp_path / "proj" / "a.md")]


def test_error_exits_nonzero(corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        run(["delete", str(corpus / "never-ingested.md"), "--base-dir", str(corpus)])
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_env_used_when_no_flags(corpus, capsys, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(corpus))
    run(["ingest", str(corpus), "--json"])
    ingested = json.loads(capsys.readouterr().out)["ingested"]
    assert sorted(Path(i["source"]).name for i in ingested) == ["a.md", "b.md"]


def test_unknown_model_name_is_clean_error(corpus, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_make_embedder", _REAL_MAKE_EMBEDDER)  # use the real factory
    with pytest.raises(SystemExit) as exc:
        run(["status", "--base-dir", str(corpus), "--model-name", "totally-bogus-model"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "totally-bogus-model" in err and "Traceback" not in err


def test_ingest_partial_failure_reports_both(corpus, capsys):
    bad = corpus / "bad.xyz"
    bad.write_text("unsupported")
    with pytest.raises(SystemExit) as exc:
        run(["ingest", str(corpus / "a.md"), str(bad), "--base-dir", str(corpus), "--json"])
    assert exc.value.code == 1
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert [i["source"] for i in payload["ingested"]] == [str(corpus / "a.md")]
    assert payload["failed"] and "bad.xyz" in payload["failed"][0]["source"]


def test_ingest_dedupes_overlapping_arguments(corpus, capsys):
    run(["ingest", str(corpus), str(corpus / "sub"), "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    sources = [i["source"] for i in payload["ingested"]]
    assert len(sources) == len(set(sources)) == 2


def test_status_and_read_match_server_fields(corpus, capsys):
    run(["ingest", str(corpus / "a.md"), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["status", "--base-dir", str(corpus), "--json"])
    assert "hybridWeight" in json.loads(capsys.readouterr().out)
    run(["read", str(corpus / "a.md"), "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert {"source", "sourceType", "title", "chunkCount", "text"} <= set(payload)


def test_query_reports_sources_like_the_server(corpus, capsys):
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["query", "tokens and auth", "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert {"results", "sources"} <= set(payload)
    assert payload["sources"], "expected an aggregated sources list"
    assert all({"source", "title", "hits"} == set(s) for s in payload["sources"])
    # sources are the distinct result sources, in rank order, with hit counts
    ranked = list(dict.fromkeys(r["source"] for r in payload["results"]))
    assert [s["source"] for s in payload["sources"]] == ranked
    assert sum(s["hits"] for s in payload["sources"]) == len(payload["results"])


def test_query_human_output_lists_sources(corpus, capsys):
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["query", "tokens and auth", "--base-dir", str(corpus)])
    out = capsys.readouterr().out
    assert "Sources:" in out and "a.md" in out


def test_read_neighbors_matches_server_fields(corpus, capsys):
    run(["ingest", str(corpus / "a.md"), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(
        [
            "read-neighbors",
            "--file-path",
            str(corpus / "a.md"),
            "--chunk-index",
            "0",
            "--base-dir",
            str(corpus),
            "--json",
        ]
    )
    chunks = json.loads(capsys.readouterr().out)["chunks"]
    assert chunks
    assert all({"source", "title", "chunkIndex", "text"} <= set(c) for c in chunks)
    assert all(c["source"] == str(corpus / "a.md") for c in chunks)


def test_sync_exits_nonzero_when_a_file_fails(corpus, capsys, monkeypatch):
    monkeypatch.setenv("MAX_FILE_SIZE", "5")  # every corpus file is bigger than this
    with pytest.raises(SystemExit) as exc:
        run(["sync", "--base-dir", str(corpus), "--json"])
    assert exc.value.code == 1
    out = capsys.readouterr()
    counts = json.loads(out.out)["counts"]
    assert counts["failed"] == 2 and counts["ingested"] == 0
    assert "MAX_FILE_SIZE" in out.err  # per-file warnings still printed


def test_status_reports_config_error_and_exits_zero(corpus, capsys, monkeypatch):
    monkeypatch.setenv("BASE_DIRS", "{not json")
    run(["status", "--json"])  # must not raise SystemExit
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] and "BASE_DIRS" in payload["configError"]
    assert "roots" not in payload


def test_other_commands_still_fail_loudly_on_config_error(corpus, capsys, monkeypatch):
    monkeypatch.setenv("BASE_DIRS", "{not json")
    with pytest.raises(SystemExit) as exc:
        run(["list", "--json"])
    assert exc.value.code == 1
    assert "BASE_DIRS" in capsys.readouterr().err
