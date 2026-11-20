"""RAG_MAX_TABLE_ROWS: a spreadsheet's data rows, counted, and the ones above the limit
kept out of the index, loudly."""

from __future__ import annotations

import asyncio
import json
import zipfile
from collections import Counter
from pathlib import Path

import openpyxl
import pytest
from fastmcp import Client
from openpyxl.styles import Font

import minirag_mcp.cli as cli
from minirag_mcp import ocr
from minirag_mcp.config import load_config
from minirag_mcp.handle import IndexHandle
from minirag_mcp.ingest import tables
from minirag_mcp.ingest.scanner import compute_diff, scan_roots
from minirag_mcp.ingest.tables import data_rows
from minirag_mcp.server import create_app

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def sheet(rows: str, *, dimension: str = "A1:C5", tail: str = "") -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<worksheet xmlns="{MAIN}">'
        f'<dimension ref="{dimension}"/><sheetData>{rows}</sheetData>{tail}</worksheet>'
    )


def valued(r: int) -> str:
    return f'<row r="{r}" spans="1:3"><c r="A{r}" t="n"><v>{r}</v></c></row>'


def xlsx(path: Path, *sheets: str, compression: int = zipfile.ZIP_DEFLATED) -> Path:
    """A zip holding only what the count reads: one worksheet part per sheet."""
    with zipfile.ZipFile(path, "w", compression) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("xl/sharedStrings.xml", "<sst><si><t>shared</t></si></sst>")
        for i, body in enumerate(sheets, 1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", body)
    return path


# --- counting ---------------------------------------------------------------------------


def test_an_xlsx_counts_its_valued_rows_not_the_dimension_it_declares(tmp_path):
    rows = "".join(valued(r) for r in range(1, 5))
    rows += '<row r="5"><c r="A5" t="inlineStr"><is><t>inline</t></is></c></row>'
    path = xlsx(tmp_path / "t.xlsx", sheet(rows, dimension="A1:XFD1048576"))
    assert data_rows(path, 10**7) == 5


def test_formatted_rows_without_a_value_do_not_count(tmp_path):
    rows = (
        valued(1)
        + '<row r="2" s="3" customFormat="1"/>'
        + '<row r="3" spans="1:3"><c r="A3" s="2"/><c r="B3" s="2"></c></row>'
        + '<row r="4"><c r="A4"><f>SUM(A1:A1)</f></c></row>'
        + valued(5)
    )
    assert data_rows(xlsx(tmp_path / "t.xlsx", sheet(rows)), 100) == 2


def test_a_namespaced_row_counts(tmp_path):
    body = (
        f'<x:worksheet xmlns:x="{MAIN}"><x:sheetData>'
        + "".join(f'<x:row r="{r}"><x:c r="A{r}"><x:v>{r}</x:v></x:c></x:row>' for r in (1, 2))
        + '<x:row r="3"><x:c r="A3" t="inlineStr"><x:is><x:t>a</x:t></x:is></x:c></x:row>'
        + '<x:row r="4"><x:c r="A4" s="1"/></x:row>'
        + "</x:sheetData></x:worksheet>"
    )
    assert data_rows(xlsx(tmp_path / "t.xlsx", body), 100) == 3


def test_row_breaks_never_start_a_row(tmp_path):
    """Were `<rowBreaks>` a row's start, the value after it would count a second time."""
    rows = '<row r="1"><c r="A1"><v>1</v></c><rowBreaks count="1"/><c r="B1"><v>2</v></c></row>'
    tail = '<rowBreaks count="1" manualBreakCount="1"><brk id="1" max="16383" man="1"/></rowBreaks>'
    assert data_rows(xlsx(tmp_path / "t.xlsx", sheet(rows, tail=tail)), 100) == 1


def test_a_row_cut_by_a_chunk_boundary_counts_once(tmp_path):
    """Wherever the 1 MiB boundary falls inside a row, from its start tag to its value."""
    first = '<row r="1" spans="1:3"><c r="A1" t="n"><v>1</v></c></row>'
    head = sheet("").split("<sheetData>")[0] + "<sheetData>"
    counted = {}
    for before in range(1, len(first)):
        pad = tables.CHUNK - len(head) - before
        body = head + " " * pad + first + valued(2) + "</sheetData></worksheet>"
        assert body.index(first) + before == tables.CHUNK
        counted[before] = data_rows(xlsx(tmp_path / f"t{before}.xlsx", body), 100)
    assert counted == dict.fromkeys(range(1, len(first)), 2)


def test_the_rows_of_every_sheet_are_summed(tmp_path):
    one = sheet("".join(valued(r) for r in range(1, 4)))
    two = sheet("".join(valued(r) for r in range(1, 5)))
    assert data_rows(xlsx(tmp_path / "t.xlsx", one, two), 100) == 7


def test_counting_stops_once_past_the_limit(tmp_path):
    """The sheet after the one that passed the limit is never read: here it cannot be."""
    one = sheet("".join(valued(r) for r in range(1, 6)))
    two = sheet(valued(1) + "<!-- SECOND SHEET -->")
    path = xlsx(tmp_path / "t.xlsx", one, two, compression=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"SECOND SHEET", b"SECOND SHEEP"))  # its CRC no longer holds
    assert data_rows(path, 7) is None
    assert data_rows(path, 3) == 4
    csv = tmp_path / "t.csv"
    csv.write_text("".join(f"{i},x\n" for i in range(10)))
    assert data_rows(csv, 3) == 4


def test_a_csv_counts_its_non_blank_lines(tmp_path):
    csv = tmp_path / "t.CSV"
    csv.write_bytes(b"a,b\n\n1,2\r\n   \n\t\r\n3,4\n5,6")
    assert data_rows(csv, 100) == 4


def test_what_is_not_a_readable_table_is_left_to_the_parse(tmp_path):
    (tmp_path / "zip.xlsx").write_bytes(b"not a zip at all")
    (tmp_path / "t.md").write_text("a\nb\n")
    assert data_rows(tmp_path / "zip.xlsx", 1) is None
    assert data_rows(tmp_path / "missing.csv", 1) is None
    assert data_rows(tmp_path / "t.md", 1) is None


def test_a_workbook_written_by_openpyxl_counts_as_its_values_say(tmp_path):
    book = openpyxl.Workbook()
    first = book.active
    for r in range(1, 7):
        first.cell(row=r, column=1, value=f"row {r}")
    for r in range(7, 40):
        first.cell(row=r, column=2).font = Font(bold=True)  # formatted, empty
    second = book.create_sheet("Second")
    second.append(["h1", "h2"])
    second.append([1, 2])
    path = tmp_path / "book.xlsx"
    book.save(path)
    assert data_rows(path, 100) == 8


# --- the diff and the sync -------------------------------------------------------------

LIMIT = 5
SKIPPED = f"skipped: more than {LIMIT} data rows exceeds RAG_MAX_TABLE_ROWS ({LIMIT})"


def csv(path: Path, rows: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{i},value {i}\n" for i in range(rows)), encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# Alpha\n\nAlpha body about budgets.")
    csv(root / "small.csv", LIMIT)
    csv(root / "big.csv", LIMIT * 3)
    return root


def handle(root: Path, tmp_path: Path, embedder, **env: str) -> IndexHandle:
    config = load_config({"BASE_DIR": str(root), "DB_PATH": str(tmp_path / "db"), **env}, cwd=root)
    return IndexHandle(config, embedder_factory=lambda scheme: embedder)


def run_sync(h: IndexHandle) -> tuple[dict[str, int], list[dict]]:
    counts, errors, rebuilt = h.run_sync(None)
    assert rebuilt is None
    return counts, errors


def test_a_diff_holds_new_tables_to_the_limit_and_only_with_one(root):
    entries = scan_roots([root])
    diff = compute_diff(entries, [], max_file_size=10**9, max_table_rows=LIMIT)
    assert diff.large_tables == [root / "big.csv"]
    assert sorted(diff.to_ingest) == [root / "a.md", root / "small.csv"]
    unlimited = compute_diff(entries, [], max_file_size=10**9)
    assert unlimited.large_tables == [] and len(unlimited.to_ingest) == 3


def test_a_sync_skips_a_table_over_the_limit_without_failing(root, tmp_path, fake_embedder):
    h = handle(root, tmp_path, fake_embedder, RAG_MAX_TABLE_ROWS=str(LIMIT))
    counts, errors = run_sync(h)
    assert (counts["scanned"], counts["ingested"], counts["largeTables"]) == (3, 2, 1)
    assert counts["failed"] == 0
    assert errors == [{"source": str(root / "big.csv"), "error": SKIPPED}]
    store = h.get().store
    assert store.get_source(str(root / "big.csv")) is None
    assert store.get_source(str(root / "small.csv")) is not None


def test_a_table_that_grows_over_the_limit_leaves_the_index(root, tmp_path, fake_embedder):
    h = handle(root, tmp_path, fake_embedder, RAG_MAX_TABLE_ROWS=str(LIMIT))
    run_sync(h)
    grown = csv(root / "small.csv", LIMIT + 1)
    counts, errors = run_sync(h)
    assert (counts["deleted"], counts["largeTables"], counts["failed"]) == (1, 2, 0)
    assert {"source": str(grown), "error": SKIPPED} in errors
    assert h.get().store.get_source(str(grown)) is None


def test_a_sync_without_a_limit_reads_every_table(root, tmp_path, fake_embedder):
    counts, errors = run_sync(handle(root, tmp_path, fake_embedder))
    assert (counts["ingested"], counts["largeTables"], errors) == (3, 0, [])


@pytest.mark.parametrize("workers", [1, 2])
def test_a_sync_counts_each_table_it_reads_once_and_an_unchanged_one_never(
    root, tmp_path, fake_embedder, monkeypatch, inline_workers, workers
):
    """Once in the diff, never again where the file is read, in this process or a worker."""
    monkeypatch.setattr(ocr, "available", lambda: False)
    for i in range(8):
        csv(root / f"t{i}.csv", 2)
    counted: Counter[Path] = Counter()
    real = tables.data_rows

    def counting(path, limit):
        counted[path] += 1
        return real(path, limit)

    monkeypatch.setattr(tables, "data_rows", counting)
    h = handle(
        root, tmp_path, fake_embedder, RAG_MAX_TABLE_ROWS=str(LIMIT), RAG_SYNC_WORKERS=str(workers)
    )
    counts, _ = run_sync(h)
    assert counts["ingested"] == 10 and inline_workers == ([2] if workers == 2 else [])
    assert counted == dict.fromkeys(
        [root / "big.csv", root / "small.csv"] + sorted(root.glob("t*.csv")), 1
    )
    counted.clear()
    counts, _ = run_sync(h)
    assert counts["skipped"] == 10 and counted == {root / "big.csv": 1}


def test_a_moved_table_keeps_its_chunks_and_is_not_counted(
    root, tmp_path, fake_embedder, monkeypatch
):
    """Moved, its bytes are the ones indexed: as unchanged, it is never read again."""
    run_sync(handle(root, tmp_path, fake_embedder))
    (root / "sub").mkdir()
    moved = (root / "big.csv").rename(root / "sub" / "big.csv")
    counted: list[Path] = []
    real = tables.data_rows
    monkeypatch.setattr(tables, "data_rows", lambda p, n: counted.append(p) or real(p, n))
    h = handle(root, tmp_path, fake_embedder, RAG_MAX_TABLE_ROWS=str(LIMIT))
    counts, errors = run_sync(h)
    assert (counts["moved"], counts["largeTables"], errors, counted) == (1, 0, [], [])
    assert h.get().store.get_source(str(moved)) is not None


# --- the tools and commands ------------------------------------------------------------


def run_cli(tokens, capsys) -> tuple[int, str, str]:
    try:
        cli.app(tokens, result_action="return_value", exit_on_error=False)
        code = 0
    except SystemExit as e:
        code = e.code
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def shells(root, tmp_path, monkeypatch, fake_embedder) -> dict[str, dict]:
    """An index per shell over `root`, both limited to LIMIT rows."""
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)
    monkeypatch.setenv("RAG_MAX_TABLE_ROWS", str(LIMIT))
    return {
        shell: {
            "BASE_DIR": str(root),
            "DB_PATH": str(tmp_path / shell),
            "RAG_MAX_TABLE_ROWS": str(LIMIT),
        }
        for shell in ("cli", "mcp")
    }


def flags(env: dict) -> list[str]:
    return ["--base-dir", env["BASE_DIR"], "--db-path", env["DB_PATH"]]


async def finished(client, job_id: str) -> dict:
    for _ in range(3000):
        status = (await client.call_tool("sync_status", {"jobId": job_id})).data
        if status["state"] in ("succeeded", "failed"):
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("the sync job did not finish")


async def test_both_shells_report_a_skipped_table_alike_and_neither_fails(
    root, shells, capsys, fake_embedder
):
    code, out, err = run_cli(["sync", *flags(shells["cli"]), "--json"], capsys)
    async with Client(create_app(load_config(shells["mcp"]), embedder=fake_embedder)) as c:
        status = await finished(c, (await c.call_tool("sync_start", {})).data["jobId"])
    reply = json.loads(out)
    assert code == 0 and status["state"] == "succeeded"
    assert reply["counts"] == status["counts"] and reply["errors"] == status["errors"]
    assert (reply["counts"]["largeTables"], reply["counts"]["failed"]) == (1, 0)
    assert reply["errors"] == [{"source": str(root / "big.csv"), "error": SKIPPED}]
    assert f"warn: {root / 'big.csv'}: {SKIPPED}" in err


async def test_both_shells_refuse_to_ingest_a_table_over_the_limit(
    root, shells, capsys, fake_embedder
):
    path = root / "big.csv"
    async with Client(create_app(load_config(shells["mcp"]), embedder=fake_embedder)) as c:
        r = await c.call_tool("ingest_file", {"filePath": str(path)}, raise_on_error=False)
    code, _, err = run_cli(["ingest", str(path), *flags(shells["cli"])], capsys)
    expected = f"{path} has more than {LIMIT} data rows; RAG_MAX_TABLE_ROWS is {LIMIT}"
    assert r.is_error and expected in r.content[0].text
    assert code == 1 and expected in err


def test_ingesting_a_folder_refuses_its_tables_over_the_limit(root, shells, capsys):
    code, out, err = run_cli(["ingest", str(root), *flags(shells["cli"]), "--json"], capsys)
    reply = json.loads(out)
    assert code == 1 and len(reply["ingested"]) == 2
    assert [f["source"] for f in reply["failed"]] == [str(root / "big.csv")]
    assert "RAG_MAX_TABLE_ROWS" in err


async def test_status_shows_the_limit_in_both_shells(root, shells, capsys, fake_embedder):
    async with Client(create_app(load_config(shells["mcp"]), embedder=fake_embedder)) as c:
        st = (await c.call_tool("status", {})).data
    code, out, _ = run_cli(["status", *flags(shells["mcp"]), "--json"], capsys)
    assert code == 0 and json.loads(out) == st and st["maxTableRows"] == LIMIT


async def test_status_says_nothing_of_a_limit_that_is_not_set(
    root, shells, capsys, monkeypatch, fake_embedder
):
    monkeypatch.delenv("RAG_MAX_TABLE_ROWS")
    env = {k: v for k, v in shells["mcp"].items() if k != "RAG_MAX_TABLE_ROWS"}
    async with Client(create_app(load_config(env), embedder=fake_embedder)) as c:
        st = (await c.call_tool("status", {})).data
    code, out, _ = run_cli(["status", *flags(env), "--json"], capsys)
    assert code == 0 and json.loads(out) == st and "maxTableRows" not in st
