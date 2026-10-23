import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import minirag_mcp.cli as cli
from minirag_mcp import ocr
from minirag_mcp.chunker import SCHEME_VERSION
from minirag_mcp.labels.model import INSTALL_HINT, ModelError
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


def test_list_reports_the_ocr_engine_in_json_and_in_the_human_listing(corpus, capsys, monkeypatch):
    """How a document entered the index is user-visible, so it has to survive on both
    of the CLI's output paths — and match the MCP `list_files` field of the same name
    (tests/test_server.py)."""
    from minirag_mcp import ocr

    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "recognized invoice text")
    scan = corpus / "scan.png"
    scan.write_bytes(b"png bytes irrelevant, ocr is faked")
    run(["ingest", str(scan), "--base-dir", str(corpus)])
    capsys.readouterr()

    run(["list", "--base-dir", str(corpus), "--json"])
    by_source = {f["source"]: f for f in json.loads(capsys.readouterr().out)["files"]}
    assert by_source[str(scan)]["ocrEngine"] == "rapidocr"
    assert by_source[str(corpus / "a.md")]["ocrEngine"] == ""

    run(["list", "--base-dir", str(corpus)])
    lines = {
        ln.split("] ", 1)[1].split(" (", 1)[0]: ln
        for ln in capsys.readouterr().out.split("\n")
        if "] " in ln
    }
    assert lines[str(scan)].endswith(" [ocr:rapidocr]")
    assert "[ocr:" not in lines[str(corpus / "a.md")]


def test_list_reports_an_indexed_source_this_install_cannot_read(corpus, capsys, monkeypatch):
    """The CLI half of the same fact the MCP `list_files` state reports
    (tests/test_server.py): a retained source the install cannot scan is still listed."""
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_image", lambda path, config: "recognized invoice text")
    scan = corpus / "scan.png"
    scan.write_bytes(b"png bytes irrelevant, ocr is faked")
    run(["ingest", str(scan), "--base-dir", str(corpus)])
    capsys.readouterr()

    monkeypatch.setattr(ocr, "available", lambda: False)
    run(["list", "--base-dir", str(corpus), "--json"])
    by_source = {f["source"]: f for f in json.loads(capsys.readouterr().out)["files"]}
    assert by_source[str(scan)]["state"] == "unreadable"
    assert by_source[str(scan)]["chunkCount"] > 0


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
        run(["list", "--base-dir", str(corpus), "--model-name", "totally-bogus-model"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "totally-bogus-model" in err and "Traceback" not in err


def test_status_reports_an_unknown_model_as_an_index_error_like_the_server(
    corpus, capsys, monkeypatch
):
    """`status` is the diagnosis command in both shells: the server's answers this with
    indexError instead of the counts, so the CLI's does too, and exits 0."""
    monkeypatch.setattr(cli, "_make_embedder", _REAL_MAKE_EMBEDDER)
    run(["status", "--base-dir", str(corpus), "--model-name", "totally-bogus-model", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "totally-bogus-model" in payload["indexError"]
    assert "chunkCount" not in payload


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
    assert all({"source", "title", "hits", "displayPath"} == set(s) for s in payload["sources"])
    # sources are the distinct result sources, in rank order, with hit counts
    ranked = list(dict.fromkeys(r["source"] for r in payload["results"]))
    assert [s["source"] for s in payload["sources"]] == ranked
    assert sum(s["hits"] for s in payload["sources"]) == len(payload["results"])
    # `displayPath` is the same source relative to the root that holds it — the
    # string the citation policy asks an answer to copy. The CLI computes it from
    # the same config the server does, so the two interfaces cannot disagree.
    for s in payload["sources"]:
        assert s["displayPath"] and not s["displayPath"].startswith("/")
        assert s["source"] == f"{corpus}/{s['displayPath']}"


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


def test_cli_status_names_the_ocr_engine_when_the_extra_is_installed(corpus, capsys, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    run(["status", "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ocr"] == ocr.ENGINE_RAPIDOCR
    assert "ocrHint" not in payload


def test_cli_status_says_ocr_is_unavailable_and_how_to_get_it(corpus, capsys, monkeypatch):
    """The feature's whole failure mode is the extra being absent, and until now the one
    command you run to diagnose an install said nothing at all about it."""
    monkeypatch.setattr(ocr, "available", lambda: False)
    run(["status", "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ocr"] == "unavailable"
    assert "minirag-mcp[ocr]" in payload["ocrHint"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a chmod 000 folder anyway")
def test_ingest_of_a_folder_with_an_unlistable_subfolder_fails_and_names_it(corpus, capsys):
    """ "ingested 1 file(s)", exit 0, was the whole report, while everything under the
    subfolder was silently left out of what the user asked to ingest."""
    hidden = corpus / "sub"
    os.chmod(hidden, 0)
    try:
        with pytest.raises(SystemExit) as exc:
            run(["ingest", str(corpus), "--base-dir", str(corpus), "--json"])
    finally:
        os.chmod(hidden, 0o755)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    failed = json.loads(captured.out)["failed"]
    assert [f["source"] for f in failed] == [str(hidden)]
    assert "could not list this folder" in failed[0]["error"]
    assert f"warn: {hidden}:" in captured.err


CLASSIFY_TAXONOMY = """
[model]
path = "/models/fake.gguf"
temperature = 1.0

[facets.type]
values.minutes = { description = "Minutes. Decisions." }
values.report = { description = "Report. Findings." }
"""


def test_classify_try_prints_the_distribution_and_writes_nothing(
    corpus, capsys, monkeypatch, model_factory, fake_embedder
):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    (corpus / ".minirag").mkdir()
    (corpus / ".minirag" / "classify.toml").write_text(CLASSIFY_TAXONOMY)
    (corpus / "m.md").write_text("# Minutes\n\nMinutes of the weekly sync.")
    model_factory.answers = {"Minutes of": {"minutes": 0.8, "report": 0.2}}

    run(["classify", "try", str(corpus / "m.md"), "--base-dir", str(corpus), "--json"])
    out = json.loads(capsys.readouterr().out)

    assert out["labels"]["type"]["value"] == "minutes"
    assert out["distribution"]["type"] == {"minutes": 0.8, "report": 0.2}
    assert Store(corpus / ".minirag" / "lancedb", dim=fake_embedder.dim).chunk_count() == 0
    run(["classify", "try", str(corpus / "m.md"), "--base-dir", str(corpus)])
    human = capsys.readouterr().out
    assert "type: minutes" in human and "report" in human


def test_classify_try_without_classify_toml_fails_loudly(
    corpus, capsys, monkeypatch, model_factory
):
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    with pytest.raises(SystemExit) as e:
        run(["classify", "try", str(corpus / "a.md"), "--base-dir", str(corpus)])
    assert e.value.code == 1
    assert "document labels are off" in capsys.readouterr().err


def test_a_malformed_label_flag_names_the_expected_form(corpus, capsys):
    with pytest.raises(SystemExit) as e:
        run(["query", "alpha", "--label", "x", "--base-dir", str(corpus)])
    assert e.value.code == 1
    assert "--label takes facet=value, got 'x'" in capsys.readouterr().err


def test_list_sorted_by_date_from_the_cli(corpus, capsys, monkeypatch, model_factory):
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    model_factory.default = {"report": 1.0}
    (corpus / ".minirag").mkdir()
    (corpus / ".minirag" / "classify.toml").write_text(CLASSIFY_TAXONOMY)
    (corpus / "old.md").write_text("# Old\n\nDate: 1 January 2020.")
    (corpus / "new.md").write_text("# New\n\nDate: 1 January 2026.")
    # A document with no date in its text is dated by its file time.
    for name in ("a.md", "sub/b.md"):
        os.utime(corpus / name, (946684800, 946684800))  # 2000-01-01
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    (corpus / "fresh.md").write_text("# Fresh\n\nNever ingested, so it has no date.")
    capsys.readouterr()
    run(["list", "--sort", "date", "--base-dir", str(corpus), "--json"])
    files = json.loads(capsys.readouterr().out)["files"]
    names = [Path(f["source"]).name for f in files]
    assert names[:2] == ["new.md", "old.md"]
    assert sorted(names[2:4]) == ["a.md", "b.md"] and names[4] == "fresh.md"
    with pytest.raises(SystemExit):
        run(["list", "--sort", "size", "--base-dir", str(corpus)])
    assert "sort must be 'date'" in capsys.readouterr().err


def _with_taxonomy(corpus, monkeypatch, model_factory, text=CLASSIFY_TAXONOMY) -> Path:
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: True)
    monkeypatch.setattr(cli, "_make_model_factory", lambda: model_factory)
    (corpus / ".minirag").mkdir(exist_ok=True)
    path = corpus / ".minirag" / "classify.toml"
    path.write_text(text)
    return path


def test_a_filter_that_leaves_documents_out_says_so_in_human_output(
    corpus, capsys, monkeypatch, model_factory, public_dns
):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    _with_taxonomy(corpus, monkeypatch, model_factory)
    model_factory.error = ModelError("the classify model is not installed")
    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nAuth notes.", "R"))
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    run(["ingest-url", "https://example.com/p", "--base-dir", str(corpus)])
    capsys.readouterr()
    monkeypatch.setattr("minirag_mcp.labels.model.available", lambda: False)
    flags = ["--label", "type=minutes", "--date-from", "2000-01-01", "--base-dir", str(corpus)]
    expected = [
        "warn: 3 documents in scope were left out for having no label for a filtered facet",
        "warn: 1 document in scope was left out for having no date",
        "warn: labels for type come from the model, which cannot run here: " + INSTALL_HINT,
    ]

    run(["list", *flags])
    listed = capsys.readouterr()
    run(["query", "auth", *flags])
    queried = capsys.readouterr()

    assert listed.out.strip() == "no files found"
    assert listed.err.splitlines() == expected
    assert queried.out.strip() == "no results"
    assert queried.err.splitlines() == expected


def test_human_list_and_query_rows_show_the_date_and_the_labels(
    corpus, capsys, monkeypatch, model_factory
):
    _with_taxonomy(corpus, monkeypatch, model_factory)
    model_factory.answers = {"Minutes of": {"minutes": 1.0}}
    model_factory.default = {"report": 0.625, "minutes": 0.375}
    old, new = corpus / "old.md", corpus / "new.md"
    old.write_text("# Old\n\nMinutes of the kickoff. Date: 1 January 2020.")
    new.write_text("# New\n\nStatus report on the kickoff. Date: 1 January 2026.")
    run(["ingest", str(old), str(new), "--base-dir", str(corpus)])
    capsys.readouterr()

    run(["list", "--sort", "date", "--base-dir", str(corpus)])
    listed = capsys.readouterr().out.splitlines()
    run(["query", "kickoff", "--base-dir", str(corpus)])
    queried = capsys.readouterr().out.splitlines()

    assert listed[:2] == [
        f"[ingested] {new} (1 chunks) 2026-01-01 (document) type=report (uncertain)",
        f"[ingested] {old} (1 chunks) 2020-01-01 (document) type=minutes",
    ]
    assert f"  {old} (1 hit) 2020-01-01 (document) type=minutes" in queried
    assert f"  {new} (1 hit) 2026-01-01 (document) type=report (uncertain)" in queried


def test_ingest_loads_the_model_once_for_all_its_files(corpus, capsys, monkeypatch, model_factory):
    _with_taxonomy(corpus, monkeypatch, model_factory)
    model_factory.default = {"report": 1.0}
    for i in range(2):
        (corpus / f"n{i}.md").write_text(f"# N{i}\n\nStatus report {i}.")

    run(["ingest", str(corpus), "--base-dir", str(corpus)])

    assert capsys.readouterr().out.splitlines()[0] == "ingested 4 file(s):"
    assert len(model_factory.created) == 1
    assert len(model_factory.created[0].calls) == 4
    assert model_factory.created[0].closed


def test_ingest_warns_once_about_a_model_that_cannot_load_and_still_succeeds(
    corpus, capsys, monkeypatch, model_factory
):
    path = _with_taxonomy(corpus, monkeypatch, model_factory)
    model_factory.error = ModelError("the classify model is not installed")

    run(["ingest", str(corpus), "--base-dir", str(corpus)])

    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == "ingested 2 file(s):"
    assert captured.err.splitlines() == [
        f"warn: {path}: labels: the classify model is not installed"
    ]
    assert len(model_factory.calls) == 1


def test_ingest_url_warns_about_a_model_that_cannot_load_and_still_succeeds(
    corpus, capsys, monkeypatch, model_factory, public_dns
):
    import minirag_mcp.ingest.pipeline as pmod
    from minirag_mcp.ingest.parser import ParsedDoc

    path = _with_taxonomy(corpus, monkeypatch, model_factory)
    model_factory.error = ModelError("the classify model is not installed")
    monkeypatch.setattr(pmod, "parse_url", lambda url, **kw: ParsedDoc("# R\n\nRemote body.", "R"))

    run(["ingest-url", "https://example.com/p", "--base-dir", str(corpus)])

    captured = capsys.readouterr()
    assert "example.com" in captured.out
    assert captured.err.splitlines() == [
        f"warn: {path}: labels: the classify model is not installed"
    ]


def test_classify_try_fails_loudly_when_the_model_cannot_answer(
    corpus, capsys, monkeypatch, model_factory
):
    path = _with_taxonomy(corpus, monkeypatch, model_factory)
    model_factory.error = ModelError("the classify model is not installed")

    with pytest.raises(SystemExit) as e:
        run(["classify", "try", str(corpus / "a.md"), "--base-dir", str(corpus)])

    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "type: not set (the model did not answer)" in captured.out.splitlines()
    assert captured.err.splitlines() == [
        f"warn: {path}: labels: the classify model is not installed"
    ]


WITHIN_TAXONOMY = (
    CLASSIFY_TAXONOMY
    + """
[facets.team]
within = ["sub"]
default = "core"
values.core = { description = "The core team." }
"""
)


def test_classify_try_prints_every_facet_even_one_without_a_label(
    corpus, capsys, monkeypatch, model_factory
):
    _with_taxonomy(corpus, monkeypatch, model_factory, WITHIN_TAXONOMY)
    model_factory.default = {"report": 1.0}

    run(["classify", "try", str(corpus / "a.md"), "--base-dir", str(corpus)])
    human = capsys.readouterr()
    run(["classify", "try", str(corpus / "a.md"), "--base-dir", str(corpus), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert "type: report [model, 1.00]" in human.out.splitlines()
    assert "team: not set (outside within: sub)" in human.out.splitlines()
    assert human.err == ""
    assert payload["unset"] == {"team": "outside within: sub"}
    assert payload["problems"] == []


@pytest.mark.parametrize(
    ("name", "message"),
    [("missing.md", "No such file"), ("notes.xyz", "Unsupported file extension")],
)
def test_classify_try_on_a_file_it_cannot_read_is_an_error_not_a_traceback(
    corpus, capsys, monkeypatch, model_factory, name, message
):
    _with_taxonomy(corpus, monkeypatch, model_factory)
    (corpus / "notes.xyz").write_text("not a supported format")

    with pytest.raises(SystemExit) as e:
        run(["classify", "try", str(corpus / name), "--base-dir", str(corpus)])

    assert e.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ") and message in err


def test_find_human_output_lists_documents_with_their_counts(corpus, capsys):
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["find", "alpha", "--base-dir", str(corpus)])
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == "1 of 2 documents searched match."
    assert lines[1] == "alpha: alpha 1"
    assert lines[2].startswith("a.md (1 hit, in title)")
    assert "Alpha body about tokens" in lines[3]


def test_find_human_output_states_an_absence(corpus, capsys):
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["find", "kubernetes", "--base-dir", str(corpus)])
    out = capsys.readouterr().out
    assert out.startswith("0 of 2 documents searched match.")
    assert "None of the 2 documents searched contains" in out


def test_find_says_when_more_pages_remain(corpus, capsys):
    run(["ingest", str(corpus), "--base-dir", str(corpus)])
    capsys.readouterr()
    run(["find", "alpha", "err", "--any", "--limit", "1", "--base-dir", str(corpus)])
    out = capsys.readouterr().out
    assert "--offset 1" in out
