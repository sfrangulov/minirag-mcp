"""How many data rows a spreadsheet holds, counted without parsing it."""

from __future__ import annotations

import re
import zipfile
import zlib
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO

try:
    from lzma import LZMAError
except ImportError:  # a Python built without liblzma: zipfile raises RuntimeError instead
    LZMAError = RuntimeError

# The formats this package indexes as tables; .xls and .xlsm are not indexed at all.
TABLE_EXTENSIONS = frozenset({".xlsx", ".csv"})
CHUNK = 1 << 20
# Open XML SDK writes prefixed elements (`<x:row>`). The character after `row` keeps
# `<rowBreaks>` out.
_ROW = re.compile(rb"<(?:[A-Za-z_][\w.-]*:)?row[\s/>]")
_VALUE = re.compile(rb"<(?:[A-Za-z_][\w.-]*:)?(?:v|is)>")
_END = re.compile(rb"</(?:[A-Za-z_][\w.-]*:)?sheetData>")
_SHEET = re.compile(r"xl/worksheets/[^/]+\.xml")
_BLANK = " \t\n\r\v\f\0"
# What stops a file being read as a zip or as bytes, and so stops its parse too. A
# ValueError is what zipfile raises for an entry name flagged UTF-8 that is not.
_UNREADABLE = (
    OSError,
    EOFError,
    ValueError,
    zipfile.BadZipFile,
    zlib.error,
    LZMAError,
    NotImplementedError,
    RuntimeError,
)


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
    # Latin-1 maps every byte to one character, so any encoding reads, and the universal
    # newlines end a line at CR, LF or CRLF alike. Only ASCII whitespace and NUL are blank:
    # a UTF-16 line ending leaves a NUL on a line of its own.
    with path.open(encoding="latin-1", newline=None) as fh:
        for line in fh:
            if line.strip(_BLANK):
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

    A row is judged from its start to the next row's start, or to the end of the sheet's
    rows, so the part of the stream after the last start found is carried into the next
    chunk: a row the chunk boundary cuts is judged once, whole. Nothing after the end of
    the rows is read; carried along, a long tail would be scanned again with every chunk.
    """
    rows = 0
    carry = b""
    while chunk := part.read(CHUNK):
        buf = carry + chunk
        found = _END.search(buf)
        stop = found.start() if found else len(buf)
        starts = [m.start() for m in _ROW.finditer(buf, 0, stop)]
        if not starts:
            if found:
                return rows
            # Before the first row; only a tag the boundary cuts is carried.
            cut = buf.rfind(b"<")
            carry = buf[cut:] if cut >= 0 and buf.find(b">", cut) < 0 else b""
            continue
        for begin, end in pairwise([*starts, stop] if found else starts):
            if _VALUE.search(buf, begin, end):
                rows += 1
                if rows > limit:
                    return rows
        if found:
            return rows
        carry = buf[starts[-1] :]
    if _ROW.match(carry) and _VALUE.search(carry):
        rows += 1
    return rows
