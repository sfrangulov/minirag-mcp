"""Minimal .docx builder for tests: a document.xml body, the two parts that find it, and
any other `word/` part mammoth reads by its default name, such as `word/styles.xml`."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/'
    'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)
_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Target="word/document.xml" Type="http://schemas.openxmlformats.org/'
    'officeDocument/2006/relationships/officeDocument"/>'
    "</Relationships>"
)
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def run(text: str, props: str = "") -> str:
    """One run; `props` is its `w:rPr` content, such as `<w:strike/>`."""
    return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{text}</w:t></w:r>'


def para(*runs: str) -> str:
    return "<w:p>" + "".join(runs) + "</w:p>"


def table(*rows: tuple[str, ...]) -> str:
    """A table of paragraphs, one tuple of cell paragraphs per row."""
    cells = ("<w:tr>" + "".join(f"<w:tc>{c}</w:tc>" for c in row) + "</w:tr>" for row in rows)
    return "<w:tbl>" + "".join(cells) + "</w:tbl>"


def styles(*rprs: str) -> str:
    """A `word/styles.xml` with one character style per `w:rPr` content."""
    defined = "".join(
        f'<w:style w:type="character" w:styleId="S{i}"><w:name w:val="S{i}"/>'
        f"<w:rPr>{rpr}</w:rPr></w:style>"
        for i, rpr in enumerate(rprs)
    )
    return f'<?xml version="1.0" encoding="UTF-8"?><w:styles xmlns:w="{_W}">{defined}</w:styles>'


def build_docx(path: Path, body: str, parts: Mapping[str, str] | None = None) -> Path:
    document = (
        f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{_W}">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/document.xml", document)
        for name, text in (parts or {}).items():
            z.writestr(name, text)
    return path
