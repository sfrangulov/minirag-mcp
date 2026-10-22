"""Document date from the header, the file name or the file time."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

HEADER_CHARS = 1500

_EN_MONTHS = (
    "January February March April May June July August September October November December"
).split()
_EN = "|".join(_EN_MONTHS)
_RU_STEMS = ("январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр").split("|")
_RU = "|".join(_RU_STEMS)
_WEEKDAY = (
    "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    "понедельник|вторник|сред[аы]|четверг|пятниц[аы]|суббот[аы]|воскресенье"
)


@dataclass(frozen=True)
class DocDate:
    value: str
    source: str


def _valid(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _year(raw: str) -> int | None:
    if len(raw) == 2:
        return 2000 + int(raw)
    year = int(raw)
    return year if 2000 <= year <= 2099 else None


def _en_month(name: str) -> int:
    return [m.lower() for m in _EN_MONTHS].index(name.lower()) + 1


def _ru_month(word: str) -> int | None:
    for i, stem in enumerate(_RU_STEMS, start=1):
        if re.match(stem, word.lower()):
            return i
    return None


# (pattern, builder over the match) pairs; a builder returns an ISO date or None.
def _numeric(m: re.Match[str]) -> str | None:
    year = _year(m["y"])
    return _valid(year, int(m["m"]), int(m["d"])) if year else None


def _iso(m: re.Match[str]) -> str | None:
    return _valid(int(m["y"]), int(m["m"]), int(m["d"])) if _year(m["y"]) else None


def _russian(m: re.Match[str]) -> str | None:
    month = _ru_month(m["mon"])
    year = _year(m["y"])
    return _valid(year, month, int(m["d"])) if month and year else None


def _english(m: re.Match[str]) -> str | None:
    year = _year(m["y"])
    return _valid(year, _en_month(m["mon"]), int(m["d"])) if year else None


_EN_RE_MDY = rf"(?<![\w])(?P<mon>{_EN})\s+(?P<d>\d{{1,2}}),?\s+(?P<y>\d{{4}})(?!\d)"
_EN_RE_DMY = rf"(?<!\d)(?P<d>\d{{1,2}})\s+(?P<mon>{_EN})\s+(?P<y>\d{{4}})(?!\d)"
_RU_RE = rf"(?<!\d)(?P<d>\d{{1,2}})\s+(?P<mon>(?:{_RU})\w*)\s+(?P<y>\d{{4}})(?!\d)"

_FORMS: list[tuple[re.Pattern[str], object]] = [
    (re.compile(r"(?<!\d)(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{4}|\d{2})(?!\d)"), _numeric),
    (re.compile(r"(?<!\d)(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})(?!\d)"), _numeric),
    (re.compile(r"(?<!\d)(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?!\d)"), _iso),
    (re.compile(_RU_RE, re.IGNORECASE), _russian),
    (re.compile(_EN_RE_MDY, re.IGNORECASE), _english),
    (re.compile(_EN_RE_DMY, re.IGNORECASE), _english),
]
_WORDED_FORMS = _FORMS[3:]


def _first_date(segment: str, forms=None) -> str | None:
    """Leftmost valid date of any form in the segment."""
    hits: list[tuple[int, str]] = []
    for pattern, build in forms or _FORMS:
        for m in pattern.finditer(segment):
            value = build(m)  # type: ignore[operator]
            if value:
                hits.append((m.start(), value))
                break
    return min(hits)[1] if hits else None


_LABEL = re.compile(r"(?i)(?:\*\*)?\b(?:date|sent|дата|отправлено)\b(?:\*\*)?\s*:\s*(?:\*\*)?")
_TEAMS = re.compile(rf"(?i)\b(?:{_EN}) \d{{1,2}}, \d{{4}},? \d{{1,2}}:\d{{2}}")
_WEEKDAY_RE = re.compile(rf"(?i)\b(?:{_WEEKDAY}),?\s+")
_OUTGOING = re.compile(r"(?i)исх\.?\s*№?[^\n]{0,40}?\bот\s+(\d{1,2}[./]\d{1,2}[./]\d{2,4})")


def date_from_header(text: str) -> str | None:
    head = text[:HEADER_CHARS]

    for label in _LABEL.finditer(head):
        found = _first_date(head[label.end() : label.end() + 60])
        if found:
            return found

    for m in _TEAMS.finditer(head):
        found = _first_date(m.group(0), _WORDED_FORMS)
        if found:
            return found

    for m in _WEEKDAY_RE.finditer(head):
        for pattern, build in _WORDED_FORMS:
            hit = pattern.match(head, m.end())
            if hit and (found := build(hit)):  # type: ignore[operator]
                return found

    for m in _OUTGOING.finditer(head):
        found = _first_date(m.group(1))
        if found:
            return found

    first = next((line for line in head.splitlines() if line.strip()), "")
    return _first_date(first)


def _name_iso(stem: str) -> str | None:
    for m in re.finditer(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", stem):
        if _year(m[1]) and (v := _valid(int(m[1]), int(m[2]), int(m[3]))):
            return v
    return None


def _name_dmy_dash(stem: str) -> str | None:
    for m in re.finditer(r"(?<!\d)(\d{2})-(\d{2})-(\d{4})(?!\d)", stem):
        if _year(m[3]) and (v := _valid(int(m[3]), int(m[2]), int(m[1]))):
            return v
    return None


def _name_dotted(stem: str) -> str | None:
    for m in re.finditer(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})(?!\d)", stem):
        year = _year(m[3])
        if year and (v := _valid(year, int(m[2]), int(m[1]))):
            return v
    return None


def _name_eight_digits(stem: str) -> str | None:
    for m in re.finditer(r"(?<!\d)(\d{8})(?!\d)", stem):
        s = m[1]
        if s.startswith("20") and (v := _valid(int(s[:4]), int(s[4:6]), int(s[6:]))):
            return v
        if _year(s[4:]) and (v := _valid(int(s[4:]), int(s[2:4]), int(s[:2]))):
            return v
    return None


def _name_yy_mm_dd(stem: str) -> str | None:
    for m in re.finditer(r"(?<!\d)(\d{2})-(\d{2})-(\d{2})(?!\d)", stem):
        if v := _valid(2000 + int(m[1]), int(m[2]), int(m[3])):
            return v
    return None


def _name_leading_ddmmyy(stem: str) -> str | None:
    m = re.match(r"(\d{2})(\d{2})(\d{2})(?!\d)", stem)
    if m:
        return _valid(2000 + int(m[3]), int(m[2]), int(m[1]))
    return None


_NAME_RULE_FUNCS = (
    _name_iso,
    _name_dmy_dash,
    _name_dotted,
    _name_eight_digits,
    _name_yy_mm_dd,
    _name_leading_ddmmyy,
)


def date_from_name(stem: str) -> str | None:
    for rule in _NAME_RULE_FUNCS:
        found = rule(stem)
        if found:
            return found
    return None


def document_date(stem: str | None, text: str, file_time: float | None) -> DocDate:
    found = date_from_header(text)
    if found:
        return DocDate(found, "document")
    found = date_from_name(stem) if stem else None
    if found:
        return DocDate(found, "name")
    if file_time is not None:
        return DocDate(datetime.fromtimestamp(file_time).date().isoformat(), "file")
    return DocDate("", "none")
