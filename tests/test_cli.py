import json

import pytest

import minirag_mcp.cli as cli


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
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    out = capsys.readouterr().out
    assert "2" in out  # 2 files ingested

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
    run(["sync", "--base-dir", str(corpus)])
    out = capsys.readouterr().out
    assert "ingested" in out

    run([
        "read-neighbors", "--file-path", str(corpus / "a.md"), "--chunk-index", "0",
        "--base-dir", str(corpus), "--json",
    ])
    assert json.loads(capsys.readouterr().out)["chunks"]


def test_ingest_url_mocked(corpus, capsys, monkeypatch):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    monkeypatch.setattr(pmod, "parse_url", lambda url: ParsedDoc("# R\n\nRemote body.", "R"))
    run(["ingest-url", "https://example.com/p", "--base-dir", str(corpus)])
    assert "example.com" in capsys.readouterr().out


def test_error_exits_nonzero(corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        run(["delete", str(corpus / "never-ingested.md"), "--base-dir", str(corpus)])
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().err.lower()


def test_env_used_when_no_flags(corpus, capsys, monkeypatch):
    monkeypatch.setenv("BASE_DIR", str(corpus))
    run(["ingest", str(corpus)])
    assert "2" in capsys.readouterr().out
