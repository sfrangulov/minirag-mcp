from datetime import datetime

import pytest

from minirag_mcp.labels.dates import (
    DocDate,
    date_from_header,
    date_from_name,
    document_date,
)


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("030325 Weekly sync-ru-RU", "2025-03-03"),
        ("180225Requirements Sales", "2025-02-18"),
        ("Protocol_2026-09-30_client", "2026-09-30"),
        ("20260814 status W1", "2026-08-14"),
        ("19022025 Sales requirements", "2025-02-19"),
        ("27-03-2025 Rates", "2025-03-27"),
        ("25-02-12 Purchasing part 2", "2025-02-12"),
        ("(1297) D_60 Report of 28.08.25", "2025-08-28"),
        ("api_v1.2.25", None),
        ("FS D_639 access_26.08_2", None),
        ("31022026 impossible", None),
    ],
)
def test_date_from_name(stem, expected):
    assert date_from_name(stem) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("General information\n**Date:** 26 августа 2026\nParticipants", "2026-08-26"),
        ("Дата: 3 сентября 2026\n", "2026-09-03"),
        ("**Workshop-Meeting Recording** August 5, 2026, 10:32AM 50m 59s", "2026-08-05"),
        (
            "Subject: Date: Date: From: To: RE: Workshop Friday, 21 August 2026 at 6:24",
            "2026-08-21",
        ),
        ("Notes\nMonday, 3 August 2026 at 9:00\nBody", "2026-08-03"),
        ("Notes\nПятница, 21 августа 2026\nBody", "2026-08-21"),
        ("Sent: Friday, August 21, 2026 at 6:57 PM", "2026-08-21"),
        (
            "# Call: demo. Transcript of the call 14.10.2026\n[00:00–02:00] hello",
            "2026-10-14",
        ),
        ("Исх. № 12/3 от 12.08.2026\nTo the bank", "2026-08-12"),
        ("Notes\nИсх. № 5 от 12.08.2026\nTo the bank", "2026-08-12"),
        (
            "Notes\n**Workshop-Meeting Recording** August 5, 2026, 10:32AM 50m",
            "2026-08-05",
        ),
        (
            "Notes\nDate: Date: "
            + "From: someone about the workshop agenda " * 2
            + "\nDate: 21 August 2026",
            "2026-08-21",
        ),
        # the trap: a cover letter whose only dates belong to the contract it cites
        (
            "To the currency control department\n"
            "Please amend contract No 1 dated 23.07.2025, reg. 2/269 of 28/01/2025.",
            None,
        ),
    ],
)
def test_date_from_header(text, expected):
    assert date_from_header(text) == expected


def test_a_date_past_the_header_window_is_ignored():
    text = "Intro\n" + "x" * 1600 + "\nDate: 01.02.2026"
    assert date_from_header(text) is None


def test_header_beats_name_and_name_beats_file_time():
    ts = datetime(2026, 7, 23, 12).timestamp()
    assert document_date("030325 sync", "Date: 01.02.2026", ts) == DocDate("2026-02-01", "document")
    assert document_date("030325 sync", "nothing", ts) == DocDate("2025-03-03", "name")


def test_file_time_is_the_last_resort_and_says_so():
    d = document_date("no date here", "nothing either", datetime(2026, 7, 23, 12).timestamp())
    assert d == DocDate("2026-07-23", "file")


def test_no_date_anywhere_is_none():
    assert document_date(None, "text", None) == DocDate("", "none")


def test_a_long_first_line_is_body_prose_not_a_title():
    line = "Please amend the contract dated 23.07.2025 " + "x" * 260
    assert len(line) > 200
    assert date_from_header(line) is None
