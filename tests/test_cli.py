import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import minirag_mcp.cli as cli
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.lock import sync_lock
from minirag_mcp.store import MAX_TOP_K, Store

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


def test_cli_read_reconstructs_a_multi_chunk_table_like_the_server(corpus, capsys):
    """Same reconstruction as the server's read_file: the two share `join_document`,
    and this is what catches them drifting."""
    header = "| № | Показатель | Периодичность |"
    delimiter = "| --- | --- | --- |"
    rows = [
        f"| {i} | Показатель номер {i} по форме статистической отчетности | месяц |"
        for i in range(30)
    ]
    doc = corpus / "table.md"
    doc.write_text(
        "# 1 Отчетность\n\n## 1.1 Формы\n\n" + "\n".join([header, delimiter, *rows]),
        encoding="utf-8",
    )
    run(["ingest", str(doc), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["read", str(doc), "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunkCount"] > 3
    text = payload["text"]
    assert text.count(header) == 1 and text.count(delimiter) == 1
    assert text.count("1 Отчетность > 1.1 Формы") == 1
    assert [ln for ln in text.split("\n") if ln.startswith("| ") and "---" not in ln] == [
        header,
        *rows,
    ]


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

    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nRemote body.", "R"))
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


def test_sync_refuses_while_another_process_holds_the_lock(corpus, capsys):
    """A second sync must say so on stderr and exit non-zero, not run anyway.

    The lock is held from this process, which contends identically to a foreign
    one (flock keys on the open file description). See tests/test_lock.py for the
    genuinely cross-process coverage.
    """
    with sync_lock(corpus / ".minirag" / "lancedb"):
        with pytest.raises(SystemExit) as exc:
            run(["sync", "--base-dir", str(corpus)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "syncing this index" in captured.err and "DB_PATH" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""  # no counts printed for a run that never happened


def test_sync_completes_and_exits_zero_when_the_lock_file_cannot_be_opened(corpus):
    """The advisory lock must never be the reason a sync fails.

    A root-owned `.sync.lock` left by one `sudo minirag-mcp sync` is the realistic
    shape of this; chmod 000 reproduces it without sudo. Before the lock existed
    this synced cleanly, and it has to keep doing so — with a warning, not 26
    lines of PermissionError traceback and exit 1.

    Run as a real child process, because what is under test is the process's exit
    status and which stream the warning lands on. stdout is the MCP stdio channel
    and carries nothing but the payload; pytest's in-process capture would show
    neither, since it intercepts warnings before they reach a stream. Everything
    else, argv parsing included, is the CLI as shipped — except two onnxruntime
    consumers, stubbed out in sys.modules before `minirag_mcp.cli` is even
    imported:

    - `minirag_mcp.embedder` imports fastembed at module level, so patching
      `cli._make_embedder` after the fact is too late to keep fastembed's own
      `import onnxruntime` from running.
    - `markitdown` — used for real by the ingest this test still exercises —
      unconditionally constructs a `magika.Magika()` per `MarkItDown()`, and
      that constructor eagerly builds a real `onnxruntime.InferenceSession`
      for its file-type model. There is no constructor flag to skip it.

    Either one loading onnxruntime is enough on its own to reproduce the crash
    this test guards against: a short-lived interpreter with onnxruntime's
    thread pool still alive can abort at finalization (`PyGILState_Release:
    thread state ... must be current when releasing`) on some platforms —
    observed on ubuntu-latest + Python 3.11, not on macOS or newer Python. The
    lock degradation is what is under test, not embedding or file-type
    sniffing, so both are faked: the fake magika reports a non-"ok" status,
    which sends markitdown down its normal extension-based fallback path for
    a plain .md file — the ingest is still real, just not routed through a
    model.
    """
    if os.geteuid() == 0:  # pragma: no cover - the suite is not meant to run as root
        pytest.skip("running as root: chmod 000 denies nothing")
    lock_file = corpus / ".minirag" / "lancedb" / ".sync.lock"
    lock_file.parent.mkdir(parents=True)
    lock_file.write_text("")
    lock_file.chmod(0o000)

    script = (
        "import sys, types\n"
        "fake_fastembed = types.ModuleType('fastembed')\n"
        "fake_fastembed.TextEmbedding = object\n"
        "sys.modules['fastembed'] = fake_fastembed\n"
        "fake_magika = types.ModuleType('magika')\n"
        "fake_magika.Magika = type('Magika', (), {\n"
        "    '__init__': lambda self, *a, **k: None,\n"
        "    'identify_stream': lambda self, stream: types.SimpleNamespace(status='error'),\n"
        "})\n"
        "sys.modules['magika'] = fake_magika\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from conftest import FakeEmbedder\n"
        "import minirag_mcp.cli as cli\n"
        "cli._make_embedder = lambda cfg: FakeEmbedder()\n"
        f"cli.app(['sync', '--base-dir', {str(corpus)!r}, '--json'])\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("BASE_DIR", "BASE_DIRS", "DB_PATH")}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=180, env=env
        )
    finally:
        lock_file.chmod(0o644)  # or the tmp dir cannot be cleaned up

    assert proc.returncode == 0, proc.stderr
    counts = json.loads(proc.stdout)["counts"]
    assert counts["ingested"] == 2 and counts["failed"] == 0  # the work really happened
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "Could not take the sync lock" in proc.stderr
    assert "Permission denied" in proc.stderr  # the reason, from the errno
    assert str(lock_file) in proc.stderr  # and which path
    assert "sync lock" not in proc.stdout  # stdout stays the MCP stdio channel


def test_ingest_is_not_blocked_by_a_held_sync_lock(corpus, capsys):
    """The lock is scoped to sync: single-file ingests stay unblocked."""
    with sync_lock(corpus / ".minirag" / "lancedb"):
        run(["ingest", str(corpus / "a.md"), "--base-dir", str(corpus), "--json"])
    assert json.loads(capsys.readouterr().out)["ingested"]


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


def test_query_top_k_is_clamped_and_must_be_positive(corpus, capsys, monkeypatch):
    """An oversized --top-k is capped rather than refused; the store never sees it raw."""
    seen: list[int] = []
    real = Store.search

    def spy(self, *args, **kwargs):
        seen.append(kwargs["top_k"])
        return real(self, *args, **kwargs)

    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    capsys.readouterr()
    monkeypatch.setattr(Store, "search", spy)
    run(["query", "tokens and auth", "--base-dir", str(corpus), "--top-k", "100000000", "--json"])
    assert json.loads(capsys.readouterr().out)["results"], "a clamped query still returns results"

    for bad in ("0", "-5"):
        with pytest.raises(SystemExit) as exc:
            run(["query", "tokens and auth", "--base-dir", str(corpus), "--top-k", bad])
        assert exc.value.code == 1
        assert "--top-k must be at least 1" in capsys.readouterr().err
    assert seen == [MAX_TOP_K]  # clamped, and the two refused runs never reached the store


def test_cli_query_returns_the_same_parent_fields_as_the_server(corpus, capsys):
    """The two interfaces share `result_dict`; this is what catches them drifting."""
    (corpus / "spec.md").write_text(
        "# 1 Хранение\n\n"
        + " ".join(f"Правило {i} про хранение товарно-материальных запасов." for i in range(40)),
        encoding="utf-8",
    )
    run(["ingest", str(corpus / "spec.md"), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["query", "хранение запасов", "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    hit = next(r for r in payload["results"] if r["source"].endswith("spec.md"))
    assert set(hit) == {
        "text",
        "source",
        "title",
        "chunkIndex",
        "score",
        "distance",
        "parentId",
    }
    assert hit["text"] in payload["parents"][hit["parentId"]]
    assert len(payload["parents"]) < len(payload["results"]), "sections are not repeated"


def test_cli_status_reports_the_chunking_scheme(corpus, capsys):
    run(["ingest", str(corpus / "a.md"), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["status", "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunkScheme"] == SCHEME_VERSION
    assert payload["staleChunkCount"] == 0
