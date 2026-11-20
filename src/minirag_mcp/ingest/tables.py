"""How many data rows a spreadsheet holds, counted without parsing it."""

from __future__ import annotations

import re
import zipfile
import zlib
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO

# The formats this package indexes as tables; .xls and .xlsm are not indexed at all.
TABLE_EXTENSIONS = frozenset({".xlsx", ".csv"})
CHUNK = 1 << 20
# Open XML SDK writes prefixed elements (`<x:row>`). The character after `row` keeps
# `<rowBreaks>` out.
_ROW = re.compile(rb"<(?:[A-Za-z_][\w.-]*:)?row[\s/>]")
_VALUE = re.compile(rb"<(?:[A-Za-z_][\w.-]*:)?(?:v|is)>")
_SHEET = re.compile(r"xl/worksheets/[^/]+\.xml")
# What stops a file being read as a zip or as bytes, and so stops its parse too.
_UNREADABLE = (OSError, EOFError, zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError)


def data_rows(path: Path, limit: int) -> int | None:
    """The rows of the table at `path` that hold a value, counted no further than
    `limit + 1`: a count above `limit` says only that there are more.

    An .xlsx is counted from its sheets' cells, never from the extent a sheet declares:
    a formatted whole column declares 1,048,576 rows. None for a file of another type, or
    one that cannot be read as a table, which its parse then reports.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return _csv_rows(path, limit)
        if suffix == ".xlsx":
            return _xlsx_rows(path, limit)
    except _UNREADABLE:
        return None
    return None


def over_limit(path: Path, limit: int | None) -> bool:
    """Whether `path` is a table with more than `limit` data rows; never without a limit."""
    if limit is None or path.suffix.lower() not in TABLE_EXTENSIONS:
        return False
    rows = data_rows(path, limit)
    return rows is not None and rows > limit


def _csv_rows(path: Path, limit: int) -> int:
    rows = 0
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                rows += 1
                if rows > limit:
                    break
    return rows


def _xlsx_rows(path: Path, limit: int) -> int:
    rows = 0
    with zipfile.ZipFile(path) as book:
        for name in book.namelist():
            if not _SHEET.fullmatch(name):
                continue
            with book.open(name) as part:
                rows += _sheet_rows(part, limit - rows)
            if rows > limit:
                break
    return rows


def _sheet_rows(part: BinaryIO, limit: int) -> int:
    """One sheet's rows holding a value, counted no further than `limit + 1`.

    A row is judged from its start to the next row's start, so the part of the stream
    after the last start found is carried into the next chunk: a row the chunk boundary
    cuts is judged once, whole.
    """
    rows = 0
    carry = b""
    while chunk := part.read(CHUNK):
        buf = carry + chunk
        starts = [m.start() for m in _ROW.finditer(buf)]
        if not starts:
            # Before the first row; only a tag the boundary cuts is carried.
            cut = buf.rfind(b"<")
            carry = buf[cut:] if cut >= 0 and buf.find(b">", cut) < 0 else b""
            continue
        for begin, end in pairwise(starts):
            if _VALUE.search(buf, begin, end):
                rows += 1
                if rows > limit:
                    return rows
        carry = buf[starts[-1] :]
    if _ROW.match(carry) and _VALUE.search(carry):
        rows += 1
    return rows
