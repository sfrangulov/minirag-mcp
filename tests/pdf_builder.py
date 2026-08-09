"""Minimal PDF byte builder for tests: text pages and blank (scan-like) pages."""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: list[str]) -> bytes:
    """A valid single-font PDF. Each entry is the page's text; "" means no text.

    Cyrillic cannot be encoded in the standard-font byte strings this builder
    writes, so tests use ASCII markers; the OCR routing logic under test only
    counts characters.
    """
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # object numbers are 1-based

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    kids_placeholder = add(b"")  # pages node, patched below
    for text in pages:
        if text:
            stream = f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET".encode()
            content = add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
            body = (
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (kids_placeholder, font, content)
            )
        else:
            body = b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] >>" % kids_placeholder
        page_ids.append(add(body))
    kids = b" ".join(b"%d 0 R" % p for p in page_ids)
    objects[kids_placeholder - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        kids,
        len(page_ids),
    )
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % kids_placeholder)

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        catalog,
        xref_at,
    )
    return bytes(out)
