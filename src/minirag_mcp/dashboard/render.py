"""The dashboard's pages as self-contained HTML: one inlined stylesheet, no script, no
request off the machine. Every value that comes from an index is escaped here, once."""

from __future__ import annotations

from datetime import datetime
from functools import cache
from html import escape
from importlib import resources
from urllib.parse import quote

from minirag_mcp.dashboard.data import FileRow, FolderStats, IndexView, RootStats

KIND_FACET = "type"
KIND_COLORS = ("#002B49", "#5E8AB4", "#0085CA", "#8CC8E7", "#B2DAEF")
OTHER_COLOR = "#D9D9D9"
SUB_FOLDERS_SHOWN = 12
NO_REPORT = '<span class="dim">no report yet</span>'

_STATE_CLASS = {"ready": "ok", "rebuild pending": "warn", "syncing": "warn", "unavailable": "bad"}


@cache
def _css() -> str:
    return resources.files("minirag_mcp.dashboard").joinpath("style.css").read_text("utf-8")


def _e(value: object) -> str:
    return escape(str(value), quote=True)


def _n(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _when(stamp: object, *, year: bool = False) -> str:
    """`stamp` as escaped text: a report value that is not a timestamp is shown as it is."""
    if not stamp:
        return ""
    try:
        moment = datetime.fromisoformat(str(stamp)).astimezone()
    except (ValueError, OverflowError, OSError):
        return _e(stamp)
    return moment.strftime("%d.%m.%Y %H:%M" if year else "%d.%m %H:%M")


def _sync_summary(latest: dict) -> str:
    text = f"{_when(latest.get('finishedAt'))} &middot; {_e(latest.get('ranBy', ''))}"
    if latest.get("outcome") == "failed":
        text += ' &middot; <span class="bad">failed</span>'
    return text


def _index_href(name: str) -> str:
    return f"/dashboard/index/{quote(name, safe='')}"


def _folder_href(name: str, path: str) -> str:
    return f"{_index_href(name)}/folder?path={quote(path, safe='')}"


def _pct(part: int, whole: int) -> int:
    return round(100 * part / whole) if whole else 0


def _page(title: str, head: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(title)}</title><style>{_css()}</style></head><body>"
        f'<header class="top"><div class="wrap">{head}</div></header>'
        f'<main class="wrap">{body}</main></body></html>'
    )


def _brand_head(version: str, generated: str, middle: str) -> str:
    return (
        '<a class="brand" href="/dashboard">minirag</a>'
        f'<span class="meta">{middle}</span>'
        f'<span class="meta" style="margin-left:auto"><span class="ver">{_e(version)}</span>'
        f" &middot; generated {_e(generated)}</span>"
    )


def _footer(version: str, generated: str) -> str:
    return (
        f"<footer>minirag {_e(version)} &middot; generated {_e(generated)} &middot; read-only; "
        "counts come from the indexes themselves.</footer>"
    )


def _state(view: IndexView) -> str:
    cls = _STATE_CLASS.get(view.state, "warn")
    why = f' <span class="dim">&mdash; {_e(view.why)}</span>' if view.why else ""
    return f'<span class="state {cls}">{_e(view.state)}</span>{why}'


def _state_cell(view: IndexView) -> str:
    cls = _STATE_CLASS.get(view.state, "warn")
    return f'<span class="state {cls}">{_e(view.state)}</span>'


def _failures(view: IndexView) -> int | None:
    if view.coverage is None:
        return None
    errors = view.coverage.get("errors", [])
    return int(view.coverage.get("errorsTotal", len(errors)))


def _large(view: IndexView) -> int | None:
    return None if view.coverage is None else len(view.coverage.get("largeTables", []))


def _bar(parts: list[tuple[int, str]], whole: int) -> str:
    spans = "".join(
        f'<span style="width:{100 * n / whole:.2f}%;background:{color}"></span>'
        for n, color in parts
        if n > 0 and whole
    )
    return f'<div class="bar">{spans}</div>'


def _share(part: int, whole: int) -> str:
    return (
        f'<div class="share">{_bar([(part, "#0085CA")], whole)}'
        f'<span class="num">{_pct(part, whole)}%</span></div>'
    )


def _tile(figure: str, caption: str) -> str:
    return (
        f'<div class="tile"><div class="fig">{figure}</div>'
        f'<div class="cap">{_e(caption)}</div></div>'
    )


def _tiles(view: IndexView) -> str:
    tiles = [_tile(_n(view.documents), "documents"), _tile(_n(view.chunks), "chunks")]
    failed, large = _failures(view), _large(view)
    if failed is not None:
        fig = f'<span class="bad">{_n(failed)}</span>' if failed else _n(failed)
        tiles.append(_tile(fig, "failed"))
    if large is not None:
        tiles.append(_tile(_n(large), "large tables"))
    return f'<div class="tiles">{"".join(tiles)}</div>'


def _last_sync_line(view: IndexView) -> str:
    if view.report_problem:
        return (
            '<p class="read"><b>Last sync:</b> report unreadable &mdash; '
            f"{_e(view.report_problem)}</p>"
        )
    latest = view.latest
    if latest is None:
        return f'<p class="read"><b>Last sync:</b> {NO_REPORT}</p>'
    return f'<p class="read"><b>Last sync:</b> {_sync_summary(latest)}</p>'


def _roots_table(view: IndexView, *, linked: bool = True) -> str:
    rows = []
    for root in view.roots:
        path = _e(root.path)
        label = f'<a href="{_e(_folder_href(view.name, root.path))}">{path}</a>' if linked else path
        rows.append(
            f'<tr><td>{label}</td><td class="num">{_n(root.documents)}</td>'
            f'<td class="num">{_n(root.chunks)}</td>'
            f"<td>{_share(root.chunks, view.chunks)}</td></tr>"
        )
    if not rows:
        return '<p class="dim">No document root is configured.</p>'
    return (
        '<div class="scroll"><table><thead><tr><th>Where it comes from</th>'
        '<th class="num">Documents</th><th class="num">Chunks</th><th>Share of chunks</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _kinds(view: IndexView) -> str:
    facet = view.facets.get(KIND_FACET)
    if not facet:
        return ""
    counts = sorted(
        ((vid, int(v.get("count", 0))) for vid, v in facet.get("values", {}).items()),
        key=lambda item: (-item[1], item[0]),
    )
    counts = [c for c in counts if c[1] > 0]
    total = sum(n for _, n in counts)
    if not total:
        return ""
    top, rest = counts[: len(KIND_COLORS)], counts[len(KIND_COLORS) :]
    parts = [(n, KIND_COLORS[i]) for i, (_, n) in enumerate(top)]
    legend = [(vid, n, KIND_COLORS[i]) for i, (vid, n) in enumerate(top)]
    if rest:
        other = sum(n for _, n in rest)
        parts.append((other, OTHER_COLOR))
        legend.append(("other kinds", other, OTHER_COLOR))
    items = "".join(
        f'<span><i style="background:{color}"></i>{_e(name)} {_n(n)}</span>'
        for name, n, color in legend
    )
    uncertain = int(facet.get("uncertain", 0))
    return (
        "<h3>Kinds of documents</h3>"
        f'{_bar(parts, total)}<div class="legend">{items}</div>'
        f'<p class="read"><b>Read:</b> {_n(total)} documents carry a kind; '
        f"{_n(uncertain)} of those labels are uncertain.</p>"
    )


def _overview_row(view: IndexView) -> str:
    latest = view.latest
    if view.report_problem:
        sync = '<span class="bad">report unreadable</span>'
    elif latest is None:
        sync = NO_REPORT
    else:
        sync = _sync_summary(latest)
    failed, large = _failures(view), _large(view)
    failed_cell = (
        '<span class="dim">&ndash;</span>'
        if failed is None
        else (f'<span class="bad">{_n(failed)}</span>' if failed else "0")
    )
    large_cell = '<span class="dim">&ndash;</span>' if large is None else _n(large)
    return (
        f'<tr><td><a href="{_e(_index_href(view.name))}">{_e(view.name)}</a></td>'
        f"<td>{_state_cell(view)}</td>"
        f'<td class="num">{_n(view.documents)}</td><td class="num">{_n(view.chunks)}</td>'
        f'<td>{sync}</td><td class="num">{failed_cell}</td>'
        f'<td class="num">{large_cell}</td><td class="num">{_n(len(view.exclude))}</td></tr>'
    )


def _plural(n: int, word: str) -> str:
    return f"{_n(n)} {word}{'' if n == 1 else 's'}"


def _indexes(n: int) -> str:
    return f"{_n(n)} {'index' if n == 1 else 'indexes'}"


def render_overview(views: list[IndexView], *, version: str, generated: str) -> str:
    documents = sum(v.documents for v in views)
    chunks = sum(v.chunks for v in views)
    failed = sum(_failures(v) or 0 for v in views)
    large = sum(_large(v) or 0 for v in views)
    exclusions = sum(len(v.exclude) for v in views)
    summary = (
        "<h1>What the indexes hold</h1>"
        f'<p class="lead">{_indexes(len(views))}, '
        f"{_plural(documents, 'document')} in {_plural(chunks, 'chunk')}. "
        "Counts come from the indexes themselves.</p>"
        '<div class="scroll"><table><thead><tr><th>Index</th><th>State</th>'
        '<th class="num">Documents</th><th class="num">Chunks</th><th>Last sync</th>'
        '<th class="num">Failed</th><th class="num">Large tables</th>'
        '<th class="num">Exclusions</th></tr></thead>'
        f"<tbody>{''.join(_overview_row(v) for v in views)}</tbody>"
        f'<tfoot><tr><td>Total</td><td></td><td class="num">{_n(documents)}</td>'
        f'<td class="num">{_n(chunks)}</td><td></td><td class="num">{_n(failed)}</td>'
        f'<td class="num">{_n(large)}</td><td class="num">{_n(exclusions)}</td></tr></tfoot>'
        "</table></div>"
    )
    sections = []
    for v in views:
        desc = f'<p class="idx-desc">{_e(v.description)}</p>' if v.description else ""
        sections.append(
            '<section class="idx"><div class="idx-head">'
            f'<h2><a href="{_e(_index_href(v.name))}">{_e(v.name)}</a></h2>{_state(v)}</div>'
            f'{desc}<div class="grid"><div>{_roots_table(v)}{_kinds(v)}</div>'
            f"<div>{_tiles(v)}{_last_sync_line(v)}</div></div></section>"
        )
    head = _brand_head(version, generated, f"<b>{_indexes(len(views))}</b>")
    return _page(
        "minirag dashboard", head, summary + "".join(sections) + _footer(version, generated)
    )


def _folder_rows(view: IndexView, root: RootStats) -> str:
    out = []
    for folder in root.folders:
        out.append(_folder_row(view, folder))
    return "".join(out)


def _folder_link(view: IndexView, folder: FolderStats) -> str:
    return f'<a href="{_e(_folder_href(view.name, folder.path))}">{_e(folder.name)}</a>'


def _folder_row(view: IndexView, folder: FolderStats) -> str:
    shown = folder.children[:SUB_FOLDERS_SHOWN]
    subs = "".join(
        f'<div class="sub"><span></span><span>{_folder_link(view, c)}</span>'
        f'<span class="num">{_n(c.documents)}</span><span class="num">{_n(c.chunks)}</span>'
        f"<span>{_share(c.chunks, view.chunks)}</span></div>"
        for c in shown
    )
    more = len(folder.children) - len(shown)
    if more > 0:
        subs += (
            '<div class="sub"><span></span><span class="dim">'
            f"&hellip; {_plural(more, 'more folder')}</span>"
            "<span></span><span></span><span></span></div>"
        )
    return (
        f'<details><summary><span class="tw"></span><span>{_folder_link(view, folder)}</span>'
        f'<span class="num">{_n(folder.documents)}</span>'
        f'<span class="num">{_n(folder.chunks)}</span>'
        f"<span>{_share(folder.chunks, view.chunks)}</span></summary>{subs}</details>"
    )


def _folders(view: IndexView) -> str:
    blocks = []
    for root in view.roots:
        link = f'<a href="{_e(_folder_href(view.name, root.path))}">{_e(root.path)}</a>'
        blocks.append(
            f'<div class="root">{link} <span class="dim">'
            f"&middot; {_plural(root.documents, 'document')}"
            f" &middot; {_plural(root.chunks, 'chunk')}</span></div>"
            '<div class="thead"><span></span><span>Folder</span><span class="num">Documents</span>'
            '<span class="num">Chunks</span><span>Share of chunks</span></div>'
            f"{_folder_rows(view, root)}"
        )
    return "".join(blocks) or '<p class="dim">No document root is configured.</p>'


def _counts_text(counts: dict) -> str:
    return " &middot; ".join(f"{_e(k)} {_n(v)}" for k, v in counts.items() if isinstance(v, int))


def _last_sync_panel(view: IndexView) -> str:
    if view.report_problem:
        return (
            '<div class="note"><b>The last-sync report is unreadable.</b> '
            f"{_e(view.report_problem)}</div>"
        )
    latest = view.latest
    if latest is None:
        return f"<h3>Last sync</h3><p>{NO_REPORT}</p>"
    rows = [
        ("Finished", _when(latest.get("finishedAt"), year=True)),
        ("Ran by", _e(latest.get("ranBy", ""))),
        ("Scope", _e(latest.get("scope") or "whole index")),
        ("Outcome", _e(latest.get("outcome", ""))),
        ("Counts", _counts_text(latest.get("counts") or {})),
    ]
    dl = "".join(f"<dt>{_e(k)}</dt><dd>{v}</dd>" for k, v in rows)
    note = ""
    if latest.get("outcome") == "failed":
        note = (
            '<div class="note"><b>The last sync failed:</b> '
            f"{_e(latest.get('error') or 'no reason recorded')}</div>"
        )
    return f'<h3>Last sync</h3><div class="panel"><dl>{dl}</dl></div>{note}'


def _coverage(view: IndexView) -> str:
    cov = view.coverage
    if cov is None:
        return ""
    when = _when(cov.get("finishedAt"), year=True)
    later = ""
    latest = view.latest
    if latest is not None and latest.get("startedAt") != cov.get("startedAt"):
        scope = latest.get("scope")
        what = f"a sync of {_e(scope)}" if scope else "a sync"
        later = (
            f'<div class="note">{what} ran later, at {_when(latest.get("finishedAt"), year=True)}; '
            "the lists below are from the last full sync.</div>"
        )
    errors = cov.get("errors", [])
    total = int(cov.get("errorsTotal", len(errors)))
    out = [f"<h3>Failed at the last full sync &middot; {when}</h3>{later}"]
    if errors:
        rows = "".join(
            f"<tr><td>{_e(e.get('source', ''))}</td><td>{_e(e.get('error', ''))}</td></tr>"
            for e in errors
        )
        cap = (
            f'<p class="read">Showing {_n(len(errors))} of {_n(total)} failures.</p>'
            if total > len(errors)
            else ""
        )
        out.append(
            '<div class="scroll"><table><thead><tr><th>File</th><th>Error</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>{cap}"
        )
    else:
        out.append('<p class="good">Nothing failed.</p>')
    large = cov.get("largeTables", [])
    out.append(f"<h3>Skipped as large tables ({_n(len(large))})</h3>")
    if large:
        rows = "".join(f"<tr><td>{_e(s)}</td></tr>" for s in large)
        out.append(
            '<div class="scroll"><table><thead><tr><th>File</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )
    else:
        out.append('<p class="dim">None.</p>')
    return "".join(out)


def _index_panel(view: IndexView) -> str:
    search = f"full text: {view.fts_language}" if view.fts_language else "full text: default"
    if view.reranker:
        search += f"; reranker: {view.reranker}"
    limit = "no limit" if view.max_table_rows is None else f"{_n(view.max_table_rows)} rows"
    rows = [
        ("State", _state(view)),
        ("Model", _e(view.model)),
        ("Search", _e(search)),
        ("Large-table limit", _e(limit)),
    ]
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return f'<div class="panelbar">Index</div><div class="panel"><dl>{dl}</dl></div>'


def _exclusions(view: IndexView) -> str:
    items = "".join(f"<li><code>{_e(p)}</code></li>" for p in view.exclude)
    body = f'<ul class="pat">{items}</ul>' if items else '<p class="dim">No patterns.</p>'
    return f"<h3>Kept out by exclusion patterns ({_n(len(view.exclude))})</h3>{body}"


def render_index(view: IndexView, *, version: str, generated: str) -> str:
    head = _brand_head(
        version,
        generated,
        f'<span class="crumbs"><a href="/dashboard">All indexes</a> / {_e(view.name)}</span>',
    )
    desc = f'<p class="idx-desc">{_e(view.description)}</p>' if view.description else ""
    body = (
        f'<div class="idx-head"><h1>{_e(view.name)}</h1>{_state(view)}</div>{desc}'
        f'<div class="grid"><div><h3>Folders</h3>{_folders(view)}{_kinds(view)}'
        f"{_coverage(view)}{_exclusions(view)}</div>"
        f"<div>{_tiles(view)}{_last_sync_panel(view)}{_index_panel(view)}</div></div>"
        f"{_footer(version, generated)}"
    )
    return _page(f"{view.name} - minirag dashboard", head, body)


def render_folder(
    view: IndexView, path: str, files: list[FileRow], *, version: str, generated: str
) -> str:
    head = _brand_head(
        version,
        generated,
        '<span class="crumbs"><a href="/dashboard">All indexes</a> / '
        f'<a href="{_e(_index_href(view.name))}">{_e(view.name)}</a></span>',
    )
    if files:
        rows = "".join(
            f'<tr><td>{_e(f.title)}<br><span class="dim">{_e(f.source)}</span></td>'
            f'<td class="num">{_n(f.chunks)}</td><td>{_e(f.labels.get(KIND_FACET, ""))}</td>'
            f"<td>{_when(f.ingested, year=True)}</td></tr>"
            for f in files
        )
        table = (
            '<div class="scroll"><table><thead><tr><th>Title</th><th class="num">Chunks</th>'
            f"<th>Kind</th><th>Ingested</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    else:
        table = '<p class="dim">No indexed files under this folder.</p>'
    body = (
        f"<h1>{_e(view.name)}</h1>"
        f'<p class="lead"><code>{_e(path)}</code> &middot; {_plural(len(files), "file")}</p>'
        f"{table}{_footer(version, generated)}"
    )
    return _page(f"{path} - minirag dashboard", head, body)


def render_message(title: str, text: str) -> str:
    head = '<a class="brand" href="/dashboard">minirag</a>'
    return _page(title, head, f'<h1>{_e(title)}</h1><p class="lead">{_e(text)}</p>')
