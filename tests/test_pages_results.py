"""Page numbers in what both shells return, and an index built before them."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

import minirag_mcp.cli as cli
from minirag_mcp.config import load_config
from minirag_mcp.ingest import pages
from minirag_mcp.ingest.pipeline import Pipeline
from minirag_mcp.server import create_app
from minirag_mcp.store import TABLE, Store
from pdf_builder import build_pdf

RESULT_KEYS = {"text", "source", "title", "chunkIndex", "score", "distance", "parentId"}


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "contract.pdf").write_bytes(
        build_pdf(
            [
                ["Parties and definitions of this agreement."],
                ["The indemnity clause limits liability to direct damages."],
                ["Termination requires ninety days written notice."],
            ]
        )
    )
    (root / "notes.md").write_text(
        "# Notes\n\nThe indemnity clause was discussed with counsel.", encoding="utf-8"
    )
    return root


@pytest.fixture
def shells(corpus, tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(corpus), "DB_PATH": str(tmp_path / "db")}
    flags = ["--base-dir", env["BASE_DIR"], "--db-path", env["DB_PATH"]]
    return env, flags


def run_cli(tokens, capsys) -> tuple[int, str]:
    try:
        cli.app(tokens, result_action="return_value", exit_on_error=False)
        code = 0
    except SystemExit as e:
        code = e.code
    return code, capsys.readouterr().out


async def call(env, fake_embedder, tool, args):
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        return (await c.call_tool(tool, args)).data


def _hit(payload, name):
    return next(r for r in payload["results"] if r["source"].endswith(name))


async def test_a_pdf_hit_names_its_page_in_both_shells_and_other_hits_are_unchanged(
    corpus, shells, capsys, fake_embedder
):
    env, flags = shells
    run_cli(["ingest", str(corpus), *flags], capsys)
    args = {"query": "indemnity clause", "topK": 20}
    mcp = await call(env, fake_embedder, "query_documents", args)
    code, out = run_cli(["query", "indemnity clause", "--top-k", "20", "--json", *flags], capsys)
    assert code == 0
    assert json.loads(out)["results"] == mcp["results"]

    pdf = next(
        r for r in mcp["results"] if "indemnity" in r["text"] and r["source"].endswith("pdf")
    )
    assert pdf["pages"] == "2"
    assert set(pdf) == RESULT_KEYS | {"pages"}
    assert set(_hit(mcp, "notes.md")) == RESULT_KEYS


async def test_the_cli_shows_the_page_of_a_pdf_hit(corpus, shells, capsys):
    _, flags = shells
    run_cli(["ingest", str(corpus), *flags], capsys)
    _, out = run_cli(["query", "indemnity clause", "--top-k", "20", *flags], capsys)
    lines = out.splitlines()
    [head] = [
        head
        for head, body in zip(lines, lines[1:], strict=False)
        if "contract.pdf#" in head and "indemnity clause limits" in body
    ]
    assert " p. 2 — " in head


async def test_neighbours_of_a_pdf_chunk_carry_their_pages_in_both_shells(
    corpus, shells, capsys, fake_embedder
):
    env, flags = shells
    run_cli(["ingest", str(corpus), *flags], capsys)
    path = str(corpus / "contract.pdf")
    args = {"filePath": path, "chunkIndex": 1, "before": 1, "after": 1}
    mcp = await call(env, fake_embedder, "read_chunk_neighbors", args)
    assert [c["pages"] for c in mcp["chunks"]] == ["1", "2", "3"]

    code, out = run_cli(
        ["read-neighbors", "--file-path", path, "--chunk-index", "1", "--json", *flags], capsys
    )
    assert code == 0 and json.loads(out) == mcp
    _, out = run_cli(["read-neighbors", "--file-path", path, "--chunk-index", "1", *flags], capsys)
    assert "p. 3" in out


async def test_find_mentions_gives_the_page_of_a_pdf_first_match(
    corpus, shells, capsys, fake_embedder
):
    env, flags = shells
    run_cli(["ingest", str(corpus), *flags], capsys)
    mcp = await call(env, fake_embedder, "find_mentions", {"terms": "termination"})
    [row] = mcp["documents"]
    assert row["firstMatch"]["pages"] == "3"
    code, out = run_cli(["find", "termination", "--json", *flags], capsys)
    assert code == 0 and json.loads(out) == mcp
    _, out = run_cli(["find", "termination", *flags], capsys)
    assert "p. 3" in out

    md = await call(env, fake_embedder, "find_mentions", {"terms": "counsel"})
    assert set(md["documents"][0]["firstMatch"]) == {"chunkIndex", "parentId", "snippet"}


# --- an index built before page numbers ------------------------------------------


def _age(db_path, dim):
    """Take the page column away, as an index written before it existed has none."""
    Store(db_path, dim=dim)._table.drop_columns(["pages"])


async def test_pdfs_indexed_before_page_numbers_are_reported_listed_and_re_read(
    corpus, shells, capsys, fake_embedder, tmp_path
):
    env, flags = shells
    run_cli(["ingest", str(corpus), *flags], capsys)
    _age(tmp_path / "db", fake_embedder.dim)

    st = await call(env, fake_embedder, "status", {})
    code, out = run_cli(["status", "--json", *flags], capsys)
    assert code == 0
    cli_st = json.loads(out)
    assert st["unpagedSourceCount"] == cli_st["unpagedSourceCount"] == 1
    assert "sync" in st["pagesWarning"] and st["pagesWarning"] == cli_st["pagesWarning"]

    listed = (await call(env, fake_embedder, "list_files", {}))["files"]
    states = {f["source"].rsplit("/", 1)[-1]: f["state"] for f in listed}
    assert states == {"contract.pdf": "stale_pages", "notes.md": "ingested"}

    pdf = _hit(await call(env, fake_embedder, "query_documents", {"query": "indemnity"}), "pdf")
    assert "pages" not in pdf

    code, out = run_cli(["sync", "--json", *flags], capsys)
    assert code == 0
    assert json.loads(out)["counts"]["ingested"] == 1

    st = await call(env, fake_embedder, "status", {})
    assert "unpagedSourceCount" not in st and "pagesWarning" not in st
    pdf = _hit(await call(env, fake_embedder, "query_documents", {"query": "indemnity"}), "pdf")
    assert pdf["pages"]


async def test_a_pdf_whose_pages_could_not_be_told_is_not_read_again_and_again(
    corpus, shells, capsys, fake_embedder, monkeypatch
):
    env, flags = shells
    monkeypatch.setattr(pages, "page_count", lambda path: 7)
    monkeypatch.setattr(pages, "page_prints", lambda path: [])
    run_cli(["ingest", str(corpus), *flags], capsys)
    pdf = _hit(await call(env, fake_embedder, "query_documents", {"query": "indemnity"}), "pdf")
    assert "pages" not in pdf

    st = await call(env, fake_embedder, "status", {})
    assert "unpagedSourceCount" not in st
    code, out = run_cli(["sync", "--json", *flags], capsys)
    assert code == 0 and json.loads(out)["counts"]["ingested"] == 0


def test_data_and_url_sources_are_never_counted_as_missing_pages(tmp_path, fake_embedder):
    cfg = load_config({"BASE_DIR": str(tmp_path)}, cwd=tmp_path)
    store = Store(cfg.db_path, dim=fake_embedder.dim)
    Pipeline(store, fake_embedder, cfg).ingest_data("Some notes about a.pdf", source="a.pdf")
    store._table.drop_columns(["pages"])
    store = Store(cfg.db_path, dim=fake_embedder.dim)
    assert [s.pages_recorded for s in store.list_sources()] == [False]
    assert not any(pages.predates_pages(s) for s in store.list_sources())
    assert store.unpaged_pdfs() == {}


async def test_a_pdf_named_in_capitals_is_a_pdf(corpus, shells, capsys, fake_embedder, tmp_path):
    env, flags = shells
    (corpus / "contract.pdf").rename(corpus / "CONTRACT.PDF")
    run_cli(["ingest", str(corpus), *flags], capsys)
    _age(tmp_path / "db", fake_embedder.dim)
    assert (await call(env, fake_embedder, "status", {}))["unpagedSourceCount"] == 1
    listed = (await call(env, fake_embedder, "list_files", {}))["files"]
    assert {f["source"].rsplit("/", 1)[-1]: f["state"] for f in listed}["CONTRACT.PDF"] == (
        "stale_pages"
    )


def test_status_counts_pdfs_without_pages_without_listing_every_document(
    corpus, tmp_path, fake_embedder, monkeypatch
):
    from minirag_mcp.results import pages_status

    cfg = load_config({"BASE_DIR": str(corpus), "DB_PATH": str(tmp_path / "db")})
    store = Store(cfg.db_path, dim=fake_embedder.dim)
    Pipeline(store, fake_embedder, cfg).ingest_file(corpus / "contract.pdf")
    _age(cfg.db_path, fake_embedder.dim)
    store = Store(cfg.db_path, dim=fake_embedder.dim)

    def slow(*args, **kwargs):
        raise AssertionError("status must not list every document to count these")

    monkeypatch.setattr(Store, "list_sources", slow)
    assert pages_status(store)["unpagedSourceCount"] == 1


def test_the_cli_says_pp_for_a_range():
    assert cli._on_pages("12-13") == " pp. 12-13"
    assert cli._on_pages("12") == " p. 12"
    assert cli._on_pages("") == ""


# --- an upgrade on an install that cannot read what it indexed ----------------------

LONG = "long enough text line to clear the default per-page threshold easily"


async def test_a_pdf_read_with_ocr_is_kept_by_an_install_without_ocr(
    tmp_path, capsys, monkeypatch, fake_embedder
):
    """Read again without OCR it would lose its scanned pages and then count as current;
    a scan with no text layer at all would fail every sync."""
    from minirag_mcp import ocr

    root = tmp_path / "docs"
    root.mkdir()
    (root / "mixed.pdf").write_bytes(build_pdf([[LONG], ""]))
    (root / "scan.pdf").write_bytes(build_pdf(["", ""]))
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    env = {"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db")}
    flags = ["--base-dir", env["BASE_DIR"], "--db-path", env["DB_PATH"]]
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(
        ocr, "ocr_pdf", lambda path, config, todo: {i: f"Recognized page {i}" for i in todo}
    )
    code, out = run_cli(["sync", "--json", *flags], capsys)
    assert code == 0 and json.loads(out)["counts"]["ingested"] == 2
    _age(tmp_path / "db", fake_embedder.dim)

    monkeypatch.setattr(ocr, "available", lambda: False)
    code, out = run_cli(["sync", "--json", *flags], capsys)
    report = json.loads(out)
    assert code == 0
    assert report["counts"]["ingested"] == 0 and report["counts"]["unreadable"] == 2
    assert all("[ocr]" in e["error"] for e in report["errors"]), report["errors"]
    store = Store(tmp_path / "db", dim=fake_embedder.dim)
    assert all(
        any("Recognized page" in c.text for c in store.all_chunks(str(root / name)))
        for name in ("mixed.pdf", "scan.pdf")
    )
    listed = (await call(env, fake_embedder, "list_files", {}))["files"]
    assert {f["state"] for f in listed} == {"unreadable"}
    warning = (await call(env, fake_embedder, "status", {}))["pagesWarning"]
    assert "2 of them were read with OCR" in warning

    monkeypatch.setattr(ocr, "available", lambda: True)
    code, out = run_cli(["sync", "--json", *flags], capsys)
    assert code == 0 and json.loads(out)["counts"]["ingested"] == 2
    store = Store(tmp_path / "db", dim=fake_embedder.dim)
    assert all(c.pages for c in store.all_chunks(str(root / "scan.pdf")))


# --- an index that could not gain the page column ----------------------------------


def _no_new_columns(monkeypatch, db_path):
    import lancedb

    from store_helpers import _raise_commit_conflict

    table_type = type(lancedb.connect(str(db_path)).open_table(TABLE))
    monkeypatch.setattr(table_type, "add_columns", _raise_commit_conflict)


@pytest.mark.filterwarnings("ignore:Could not add the pages column:RuntimeWarning")
async def test_an_index_without_the_page_column_still_takes_writes_and_says_how_to_add_it(
    corpus, shells, capsys, fake_embedder, monkeypatch, tmp_path
):
    env, flags = shells
    run_cli(["ingest", str(corpus), *flags], capsys)
    _age(tmp_path / "db", fake_embedder.dim)
    _no_new_columns(monkeypatch, tmp_path / "db")

    st = await call(env, fake_embedder, "status", {})
    assert st["unpagedSourceCount"] == 1 and "write access" in st["pagesWarning"]

    (corpus / "notes.md").write_text("# Notes\n\nRewritten after the upgrade.", encoding="utf-8")
    code, out = run_cli(["sync", "--json", *flags], capsys)
    report = json.loads(out)
    assert code == 0, report
    assert report["counts"]["ingested"] == 1 and report["counts"]["failed"] == 0

    monkeypatch.undo()
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    code, out = run_cli(["sync", "--json", *flags], capsys)
    assert code == 0 and json.loads(out)["counts"]["ingested"] == 1
    pdf = _hit(await call(env, fake_embedder, "query_documents", {"query": "indemnity"}), "pdf")
    assert pdf["pages"]
