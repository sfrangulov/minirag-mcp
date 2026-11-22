"""`serve`'s dashboard: pages behind a session cookie that a one-time code signs in, the
code handed only to a holder of the bearer token, and `/mcp` behind exactly the rule it
had before."""

from __future__ import annotations

import functools
import inspect
import json
import logging
import threading
import time
from dataclasses import dataclass
from html import escape
from urllib.parse import quote

import httpx
import pytest
import uvicorn
from starlette.testclient import TestClient

from http_helpers import AUTH, INITIALIZE, MCP_HEADERS, TOKEN, serving
from index_helpers import file_registry
from minirag_mcp import http_server
from minirag_mcp.dashboard import app as dashboard_mod
from minirag_mcp.dashboard.data import DashboardData
from minirag_mcp.dashboard.sessions import CODE_TTL, COOKIE, SESSION_TTL, Sessions
from minirag_mcp.http_server import build_http_app, serve
from minirag_mcp.identity import IndexState
from minirag_mcp.registry import IndexRegistry
from minirag_mcp.security import LOOPBACK_NAMES
from minirag_mcp.server import create_app
from minirag_mcp.sync_report import FORMAT, REPORT_FILE

NOT_SIGNED_IN = "Not signed in. Run `minirag-mcp dashboard` to open this page signed in."
UNAUTHORIZED = "Unauthorized: send Authorization: Bearer <the token in the server's token file>"
FOREIGN_ORIGIN = "Forbidden: Origin is not a loopback origin"
MISDIRECTED = "Misdirected Request: Host must be a loopback name"
NOT_FOUND = "Not Found: MCP is served at /mcp and /mcp/<index>"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass(frozen=True)
class Site:
    base: str
    registry: IndexRegistry
    clock: Clock

    def entry(self, name: str):
        return next(e for e in self.registry.entries if e.name == name)

    def root(self, name: str) -> str:
        return str(self.entry(name).config.roots[0])


@pytest.fixture
async def dashboard(indexes_file, fake_embedder, index_states, monkeypatch):
    """The app `serve` builds, with its dashboard, whose sessions keep the time of a clock
    the test moves."""
    clock = Clock()
    monkeypatch.setattr(http_server, "Sessions", functools.partial(Sessions, clock=clock))
    registry = file_registry(indexes_file, lambda c, s: fake_embedder)
    app = build_http_app(
        create_app(registry), token=TOKEN, allowed_hosts=LOOPBACK_NAMES, registry=registry
    )
    async with serving(app) as base:
        yield Site(base, registry, clock)


async def _code(http: httpx.AsyncClient, site: Site) -> str:
    r = await http.post(f"{site.base}/dashboard/code", headers=AUTH)
    assert r.status_code == 200, r.text
    return r.json()["code"]


async def _sign_in(http: httpx.AsyncClient, site: Site) -> httpx.Response:
    r = await http.get(f"{site.base}/dashboard/login", params={"code": await _code(http, site)})
    assert r.status_code == 303, r.text
    return r


def _ingest(site: Site, name: str) -> None:
    entry = site.entry(name)
    for path in sorted(entry.config.roots[0].iterdir()):
        entry.runtime().pipeline.ingest_file(path)


async def test_mcp_with_a_dashboard_cookie_and_no_bearer_is_refused(dashboard):
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        assert (await http.get(f"{dashboard.base}/dashboard")).status_code == 200
        cookie = {"Cookie": f"{COOKIE}={http.cookies[COOKIE]}"}
        for path in ("/mcp", "/mcp/alpha", "/api"):
            r = await http.post(
                f"{dashboard.base}{path}", json=INITIALIZE, headers={**MCP_HEADERS, **cookie}
            )
            assert (r.status_code, r.text) == (401, UNAUTHORIZED)
            assert r.headers["www-authenticate"] == "Bearer"
        served = await http.post(
            f"{dashboard.base}/mcp/alpha", json=INITIALIZE, headers={**MCP_HEADERS, **AUTH}
        )
        assert served.status_code == 200, served.text


@pytest.mark.parametrize(
    "path",
    [
        "/dashboard",
        "/dashboard/index/alpha",
        "/dashboard/index/alpha/folder?path=%2F",
        "/dashboard/index/nope",
        "/dashboard/anything",
    ],
)
@pytest.mark.parametrize(
    "headers", [{}, AUTH, {"Cookie": f"{COOKIE}=forged-0123456789abcdefghijklmnopqrstuvwxyz"}]
)
async def test_a_dashboard_page_without_a_cookie_is_refused_with_how_to_sign_in(
    dashboard, path, headers
):
    async with httpx.AsyncClient() as http:
        r = await http.get(f"{dashboard.base}{path}", headers=headers)
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("text/html")
    assert NOT_SIGNED_IN in r.text and "Traceback" not in r.text


async def test_a_code_signs_in_once(dashboard):
    login = f"{dashboard.base}/dashboard/login"
    async with httpx.AsyncClient() as http:
        code = await _code(http, dashboard)
        first = await http.get(login, params={"code": code})
        assert (first.status_code, first.headers["location"]) == (303, "/dashboard")
        assert (await http.get(f"{dashboard.base}/dashboard")).status_code == 200
    async with httpx.AsyncClient() as http:
        for params in ({"code": code}, {"code": "made-up"}, {}):
            again = await http.get(login, params=params)
            assert again.status_code == 401 and "set-cookie" not in again.headers
            assert "Run `minirag-mcp dashboard`" in again.text
        assert (await http.get(f"{dashboard.base}/dashboard")).status_code == 401


async def test_a_code_expires_after_a_minute(dashboard):
    login = f"{dashboard.base}/dashboard/login"
    async with httpx.AsyncClient() as http:
        early, late = await _code(http, dashboard), await _code(http, dashboard)
        dashboard.clock.now += CODE_TTL - 1
        assert (await http.get(login, params={"code": early})).status_code == 303
    dashboard.clock.now += 2
    async with httpx.AsyncClient() as http:
        r = await http.get(login, params={"code": late})
        assert r.status_code == 401 and "set-cookie" not in r.headers
        assert (await http.get(f"{dashboard.base}/dashboard")).status_code == 401


async def test_an_expired_session_is_told_how_to_sign_in(dashboard):
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        dashboard.clock.now += SESSION_TTL - 1
        assert (await http.get(f"{dashboard.base}/dashboard")).status_code == 200
        dashboard.clock.now += 2
        r = await http.get(f"{dashboard.base}/dashboard")
    assert r.status_code == 401
    assert NOT_SIGNED_IN in r.text and "Traceback" not in r.text


def test_sessions_count_wall_clock_time_sleep_included():
    # A monotonic clock stops while macOS sleeps, so a leaked cookie would outlive 12 h.
    assert inspect.signature(Sessions).parameters["clock"].default is time.time


async def test_the_cookie_is_scoped_to_dashboard(dashboard):
    async with httpx.AsyncClient() as http:
        r = await _sign_in(http, dashboard)
        [header] = r.headers.get_list("set-cookie")
        pair, *attributes = (part.strip() for part in header.split(";"))
        name, _, value = pair.partition("=")
        assert name == COOKIE and len(value) >= 32
        assert sorted(attributes) == sorted(
            ["Path=/dashboard", "HttpOnly", "SameSite=Strict", "Max-Age=43200"]
        )
        elsewhere = await http.post(
            f"{dashboard.base}/mcp/alpha", json=INITIALIZE, headers=MCP_HEADERS
        )
        assert "cookie" not in elsewhere.request.headers and elsewhere.status_code == 401


async def test_the_token_appears_nowhere(dashboard, caplog, capfd):
    caplog.set_level(logging.DEBUG)
    _ingest(dashboard, "alpha")
    base, root = dashboard.base, quote(dashboard.root("alpha"), safe="")
    async with httpx.AsyncClient() as http:
        answered = [await http.get(f"{base}/")]
        answered.append(await http.post(f"{base}/dashboard/code", headers=AUTH))
        code = answered[-1].json()["code"]
        answered.append(await http.get(f"{base}/dashboard"))
        for _ in range(2):
            answered.append(await http.get(f"{base}/dashboard/login", params={"code": code}))
        for path in (
            "/dashboard",
            "/dashboard/index/alpha",
            f"/dashboard/index/alpha/folder?path={root}",
            "/dashboard/index/nope",
            "/dashboard/index/alpha/folder?path=%2Fetc",
        ):
            answered.append(await http.get(f"{base}{path}"))
    assert [r.status_code for r in answered] == [303, 200, 401, 303, 401, 200, 200, 200, 404, 404]
    for r in answered:
        assert TOKEN not in r.text and TOKEN not in str(r.url)
        assert not [v for v in r.headers.values() if TOKEN in v]
    out, err = capfd.readouterr()
    assert TOKEN not in caplog.text + out + err


async def test_a_code_needs_the_bearer(dashboard):
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        for headers in ({}, {"Authorization": "Bearer not-the-token"}):
            r = await http.post(f"{dashboard.base}/dashboard/code", headers=headers)
            assert (r.status_code, r.text) == (401, UNAUTHORIZED)
            assert r.headers["www-authenticate"] == "Bearer"
            # The session went along, and did not stand in for the token.
            assert COOKIE in r.request.headers["cookie"]


@pytest.mark.parametrize(
    "refused, answer",
    [
        ({"Origin": "http://evil.example"}, (403, FOREIGN_ORIGIN)),
        ({"Origin": "null"}, (403, FOREIGN_ORIGIN)),
        ({"Host": "evil.example"}, (421, MISDIRECTED)),
    ],
)
@pytest.mark.parametrize(
    "method, path, served",
    [
        ("GET", "/", 303),
        ("GET", "/dashboard", 200),
        ("GET", "/dashboard/index/alpha", 200),
        ("GET", "/dashboard/login?code={code}", 303),
        ("POST", "/dashboard/code", 200),
    ],
)
async def test_a_foreign_origin_is_refused_on_dashboard_paths(
    dashboard, refused, answer, method, path, served
):
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        url = dashboard.base + path.format(code=await _code(http, dashboard))
        r = await http.request(method, url, headers={**AUTH, **refused})
        assert (r.status_code, r.text) == answer and "set-cookie" not in r.headers
        # The same request without that header is served: a refused login spent no code.
        assert (await http.request(method, url, headers=AUTH)).status_code == served


async def test_the_root_redirects_to_the_dashboard(dashboard):
    async with httpx.AsyncClient() as http:
        r = await http.get(f"{dashboard.base}/")
        assert (r.status_code, r.headers["location"], r.content) == (303, "/dashboard", b"")
        followed = await http.get(f"{dashboard.base}/", follow_redirects=True)
    assert followed.status_code == 401 and NOT_SIGNED_IN in followed.text


async def test_without_a_registry_there_is_no_dashboard(site):
    # Pin: what build_http_app served before it could serve a dashboard.
    async with httpx.AsyncClient() as http:
        root = await http.get(f"{site}/")
        page = await http.get(f"{site}/dashboard", headers=AUTH)
        code = await http.post(f"{site}/dashboard/code", headers=AUTH)
    assert (root.status_code, root.text) == (401, UNAUTHORIZED)
    assert (page.status_code, page.text) == (404, NOT_FOUND)
    assert (code.status_code, code.text) == (404, NOT_FOUND)


@pytest.mark.parametrize(
    "state, why",
    [
        (IndexState.UNAVAILABLE, "index.json is unreadable"),
        (IndexState.REBUILD_PENDING, "the model changed"),
        (IndexState.SYNCING, ""),
    ],
)
async def test_the_overview_serves_with_one_index_unavailable(dashboard, index_states, state, why):
    _ingest(dashboard, "alpha")
    index_states["beta"] = (state, why)
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        overview = await http.get(f"{dashboard.base}/dashboard")
        page = await http.get(f"{dashboard.base}/dashboard/index/beta")
    assert overview.status_code == 200 and page.status_code == 200
    for text in (overview.text, page.text):
        assert f">{state.value}</span>" in text and why in text
    # alpha and gamma, each in the summary table and in its own section.
    assert overview.text.count('<span class="state ok">ready</span>') == 4
    assert dashboard.root("alpha") in overview.text and dashboard.root("gamma") in overview.text


async def test_a_folder_outside_the_index_is_not_found(dashboard):
    _ingest(dashboard, "alpha")
    root = dashboard.root("alpha")
    folder = f"{dashboard.base}/dashboard/index/alpha/folder"
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        inside = await http.get(folder, params={"path": root})
        assert inside.status_code == 200 and "alpha.md" in inside.text
        for path in ("/etc", f"{root}/../beta", dashboard.root("beta"), ""):
            r = await http.get(folder, params={"path": path})
            assert r.status_code == 404 and "Not found" in r.text, path
        assert (await http.get(folder)).status_code == 404
        for url in ("/dashboard/index/nope", "/dashboard/index/nope/folder?path=%2F"):
            r = await http.get(f"{dashboard.base}{url}")
            assert r.status_code == 404 and "Not found" in r.text


async def test_pages_are_built_off_the_event_loop(dashboard, monkeypatch):
    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}
    overview, render = DashboardData.overview, dashboard_mod.render_overview

    def counted_overview(self):
        seen["statistics"] = threading.get_ident()
        return overview(self)

    def counted_render(*args, **kwargs):
        seen["rendering"] = threading.get_ident()
        return render(*args, **kwargs)

    monkeypatch.setattr(DashboardData, "overview", counted_overview)
    monkeypatch.setattr(dashboard_mod, "render_overview", counted_render)
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        assert (await http.get(f"{dashboard.base}/dashboard")).status_code == 200
    assert set(seen) == {"statistics", "rendering"}
    assert loop_thread not in seen.values()


async def test_a_page_that_cannot_be_built_is_an_error_page_not_a_traceback(
    dashboard, monkeypatch, caplog
):
    def broken(*args, **kwargs):
        raise RuntimeError("detail-only-the-log-may-hold")

    monkeypatch.setattr(dashboard_mod, "render_overview", broken)
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        r = await http.get(f"{dashboard.base}/dashboard")
    assert r.status_code == 500 and r.headers["content-type"].startswith("text/html")
    assert "Traceback" not in r.text and "detail-only-the-log-may-hold" not in r.text
    assert "the server&#x27;s log says why" in r.text
    assert "detail-only-the-log-may-hold" in caplog.text


def _report(stamp: str, errors_total: object) -> dict:
    latest = {
        "startedAt": stamp,
        "finishedAt": stamp,
        "ranBy": "cli",
        "scope": None,
        "outcome": "succeeded",
        "error": None,
        "counts": {"ingested": 1},
    }
    coverage = {
        "startedAt": stamp,
        "finishedAt": stamp,
        "errors": [{"source": "/a.pdf", "error": "boom"}],
        "errorsTotal": errors_total,
        "largeTables": [],
    }
    return {"format": FORMAT, "latest": latest, "coverage": coverage}


@pytest.mark.parametrize(
    "report, shown",
    [
        (_report("0001-01-01T00:00:00+23:59", 1), "0001-01-01T00:00:00+23:59"),
        (_report("2026-11-22T14:00:00+00:00", "many"), "errorsTotal is not a count"),
        (_report("2026-11-22T14:00:00+00:00", None), "errorsTotal is not a count"),
    ],
)
async def test_a_malformed_report_is_shown_never_a_500(dashboard, report, shown):
    _ingest(dashboard, "alpha")
    db_path = dashboard.entry("alpha").config.db_path
    (db_path / REPORT_FILE).write_text(json.dumps(report), encoding="utf-8")
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        pages = [
            await http.get(f"{dashboard.base}{p}") for p in ("/dashboard", "/dashboard/index/alpha")
        ]
    for page in pages:
        assert page.status_code == 200 and shown in page.text


DELTA = '\n[index.delta]\ndescription = "Delta documents"\nroots = ["~/corpora/delta"]\n'


async def test_the_overview_follows_an_edit_of_the_indexes_file(dashboard, indexes_file):
    (indexes_file.roots["alpha"].parent / "delta").mkdir()
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        before = await http.get(f"{dashboard.base}/dashboard")
        indexes_file.write(indexes_file.text + DELTA)
        after = await http.get(f"{dashboard.base}/dashboard")
        page = await http.get(f"{dashboard.base}/dashboard/index/delta")
    assert before.status_code == 200 and "/dashboard/index/delta" not in before.text
    assert after.status_code == 200 and "4 indexes" in after.text
    assert 'href="/dashboard/index/delta"' in after.text and "Delta documents" in after.text
    assert page.status_code == 200 and "Delta documents" in page.text


async def test_an_edit_that_does_not_load_is_shown_beside_the_indexes_still_served(
    dashboard, indexes_file
):
    async with httpx.AsyncClient() as http:
        await _sign_in(http, dashboard)
        assert "did not load" not in (await http.get(f"{dashboard.base}/dashboard")).text
        indexes_file.write(indexes_file.text + "\n[index.delta]\nroots = 5\n")
        page = await http.get(f"{dashboard.base}/dashboard")
    error = dashboard.registry.config_error
    assert error is not None and "delta" in error
    assert page.status_code == 200
    banner = f"<b>indexes.toml did not load:</b> {escape(error.rstrip('.'))}; the indexes it"
    assert f'<div class="note">{banner}' in page.text
    assert "3 indexes" in page.text
    for name in ("alpha", "beta", "gamma"):
        assert f'href="/dashboard/index/{name}"' in page.text


CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src data:; form-action 'none'; "
    "frame-ancestors 'none'; base-uri 'none'"
)


async def test_every_dashboard_response_is_kept_out_of_caches_and_locked_down(
    dashboard, monkeypatch
):
    _ingest(dashboard, "alpha")
    base, root = dashboard.base, quote(dashboard.root("alpha"), safe="")
    answered: dict[str, httpx.Response] = {}
    async with httpx.AsyncClient() as http:
        answered["root redirect"] = await http.get(f"{base}/")
        answered["not signed in"] = await http.get(f"{base}/dashboard")
        answered["code without the token"] = await http.post(f"{base}/dashboard/code")
        answered["foreign origin"] = await http.get(
            f"{base}/dashboard", headers={"Origin": "http://evil.example"}
        )
        answered["foreign host"] = await http.get(
            f"{base}/dashboard", headers={"Host": "evil.example"}
        )
        answered["code"] = await http.post(f"{base}/dashboard/code", headers=AUTH)
        code = answered["code"].json()["code"]
        answered["refused code"] = await http.get(
            f"{base}/dashboard/login", params={"code": "made-up"}
        )
        answered["login redirect"] = await http.get(
            f"{base}/dashboard/login", params={"code": code}
        )
        answered["overview"] = await http.get(f"{base}/dashboard")
        answered["index page"] = await http.get(f"{base}/dashboard/index/alpha")
        answered["folder page"] = await http.get(f"{base}/dashboard/index/alpha/folder?path={root}")
        answered["no such index"] = await http.get(f"{base}/dashboard/index/nope")
        answered["no such page"] = await http.get(f"{base}/dashboard/anything")

        def broken(*args, **kwargs):
            raise RuntimeError("page failed")

        monkeypatch.setattr(dashboard_mod, "render_overview", broken)
        answered["page that failed"] = await http.get(f"{base}/dashboard")
    assert {k: r.status_code for k, r in answered.items()} == {
        "root redirect": 303,
        "not signed in": 401,
        "code without the token": 401,
        "foreign origin": 403,
        "foreign host": 421,
        "code": 200,
        "refused code": 401,
        "login redirect": 303,
        "overview": 200,
        "index page": 200,
        "folder page": 200,
        "no such index": 404,
        "no such page": 404,
        "page that failed": 500,
    }
    for kind, r in answered.items():
        assert r.headers.get_list("cache-control") == ["no-store"], kind
        assert r.headers.get_list("content-security-policy") == [CSP], kind


async def test_mcp_responses_keep_their_headers(dashboard):
    async with httpx.AsyncClient() as http:
        r = await http.post(
            f"{dashboard.base}/mcp/alpha", json=INITIALIZE, headers={**MCP_HEADERS, **AUTH}
        )
        refused = await http.post(f"{dashboard.base}/mcp/alpha", json=INITIALIZE)
    assert r.status_code == 200 and refused.status_code == 401
    for answer in (r, refused):
        assert "content-security-policy" not in answer.headers


@pytest.fixture
def started(monkeypatch) -> list[uvicorn.Config]:
    """The configurations uvicorn was asked to serve, none of them served."""
    configs: list[uvicorn.Config] = []

    async def serve_nothing(self, sockets=None) -> None:
        configs.append(self.config)

    monkeypatch.setattr(uvicorn.Server, "serve", serve_nothing)
    return configs


def test_serve_hosts_the_dashboard(indexes_file, started, capsys):
    serve(indexes_file.path, indexes_file.env)
    [config] = started
    capsys.readouterr()
    client = TestClient(config.app, base_url="http://127.0.0.1")
    root = client.get("/", follow_redirects=False)
    page = client.get("/dashboard")
    assert (root.status_code, root.headers["location"]) == (303, "/dashboard")
    assert page.status_code == 401 and NOT_SIGNED_IN in page.text
