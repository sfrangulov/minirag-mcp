"""The CLI over indexes.toml: `--index` selects one of its indexes, `--config` names the
file, `indexes` lists them, and `--index` without a command serves that index on stdio."""

from __future__ import annotations

import json
import sys

import pytest
from fastmcp import Client, FastMCP

import minirag_mcp.cli as cli
from minirag_mcp.__main__ import main
from minirag_mcp.embedder import MINILM


@pytest.fixture(autouse=True)
def fake_model(monkeypatch, fake_embedder):
    monkeypatch.setattr(cli, "_make_embedder", lambda cfg: fake_embedder)


@pytest.fixture
def home(indexes_file, monkeypatch):
    """The process's HOME is the one the file's `~/corpora/...` roots are under."""
    monkeypatch.setenv("HOME", indexes_file.env["HOME"])
    return indexes_file


def run_meta(tokens, capsys) -> tuple[int, str, str]:
    try:
        cli.app.meta(tokens, result_action="return_value", exit_on_error=False)
        code = 0
    except SystemExit as e:
        code = e.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _cfg(indexes_file) -> list[str]:
    return ["--config", str(indexes_file.path)]


def test_index_selects_a_named_index_from_indexes_toml(home, capsys):
    cfg = _cfg(home)
    assert run_meta([*cfg, "--index", "beta", "sync", "--json"], capsys)[0] == 0
    code, out, _ = run_meta([*cfg, "--index", "beta", "query", "bananas", "--json"], capsys)
    payload = json.loads(out)
    assert code == 0 and payload["index"] == "beta"
    assert payload["sources"]
    assert all(s["source"].startswith(f"{home.roots['beta']}/") for s in payload["sources"])
    assert not (home.roots["alpha"] / ".minirag").exists()


def test_index_may_follow_the_command(home, capsys):
    cfg = _cfg(home)
    assert run_meta(["sync", "--index", "beta", *cfg], capsys)[0] == 0
    code, out, _ = run_meta(["query", "bananas", "--index", "beta", *cfg, "--json"], capsys)
    payload = json.loads(out)
    assert code == 0 and payload["index"] == "beta"
    assert [s["source"] for s in payload["sources"]] == [str(home.roots["beta"] / "beta.md")]


def test_minirag_config_selects_the_file(home, capsys, monkeypatch):
    monkeypatch.setenv("MINIRAG_CONFIG", str(home.path))
    code, out, err = run_meta(["--index", "beta", "status", "--json"], capsys)
    payload = json.loads(out)
    assert code == 0, err
    assert (payload["index"], payload["roots"]) == ("beta", [str(home.roots["beta"])])


def test_an_unknown_index_is_refused_naming_the_configured_ones(home, capsys):
    code, _, err = run_meta([*_cfg(home), "--index", "delta", "query", "x"], capsys)
    assert code == 1 and "Unknown index 'delta'. Reachable indexes: alpha, beta, gamma." in err


def test_index_cannot_be_combined_with_the_environment_index_flags(home, tmp_path, capsys):
    tokens = [*_cfg(home), "--index", "beta", "query", "x", "--db-path", str(tmp_path / "db")]
    code, _, err = run_meta(tokens, capsys)
    assert code == 1 and "--db-path" in err and "indexes.toml" in err
    assert "[index.NAME]" in err and not (tmp_path / "db").exists()


def test_without_index_the_cli_reads_the_environment_as_before(home, tmp_path, capsys):
    root = home.roots["alpha"]
    env_flags = ["--base-dir", str(root), "--db-path", str(tmp_path / "db")]
    assert run_meta(["sync", *env_flags], capsys)[0] == 0
    code, out, _ = run_meta(["query", "apricots", *env_flags, "--json"], capsys)
    payload = json.loads(out)
    assert code == 0 and payload["index"] == "default"
    assert [s["source"] for s in payload["sources"]] == [str(root / "alpha.md")]
    assert not (root / ".minirag").exists()


def test_config_without_index_is_refused_rather_than_ignored(home, tmp_path, capsys):
    """Every command but `indexes` reads the environment's index unless --index names one;
    a --config it would not read is an error, not a silently ignored flag."""
    env_flags = ["--base-dir", str(home.roots["alpha"]), "--db-path", str(tmp_path / "db")]
    for command in (["query", "x"], ["status"]):
        code, out, err = run_meta([*_cfg(home), *command, *env_flags], capsys)
        assert (code, out) == (1, ""), command
        assert "--config" in err and "--index" in err, command
    assert not (tmp_path / "db").exists()


def test_indexes_lists_every_index_in_the_file(home, capsys):
    code, out, _ = run_meta([*_cfg(home), "indexes", "--json"], capsys)
    payload = json.loads(out)
    assert code == 0 and payload["default"] is None and "configError" not in payload
    rows = payload["indexes"]
    assert [r["name"] for r in rows] == ["alpha", "beta", "gamma"]
    assert all(r["reachable"] and r["state"] == "ready" for r in rows)
    assert [(r["description"], r["language"]) for r in rows] == [
        ("Alpha documents", "English"),
        ("Beta documents", "Russian"),
        ("Gamma documents", "English"),
    ]
    code, out, _ = run_meta([*_cfg(home), "indexes"], capsys)
    assert code == 0 and out.splitlines() == [
        "alpha [ready] 0 documents, 0 chunks, fake, English — Alpha documents",
        "beta [ready] 0 documents, 0 chunks, fake, Russian — Beta documents",
        "gamma [ready] 0 documents, 0 chunks, fake, English — Gamma documents",
    ]


def test_indexes_prints_why_an_index_is_unavailable_in_place_of_its_counts(home, capsys):
    blocker = home.path.parent.parent / "blocker"
    blocker.write_text("a file where gamma's index folder should be")
    home.write(home.text + 'db = "~/blocker"\n')
    code, out, _ = run_meta([*_cfg(home), "indexes", "--json"], capsys)
    gamma = json.loads(out)["indexes"][2]
    assert code == 0 and gamma["state"] == "unavailable" and str(blocker) in gamma["stateReason"]
    code, out, _ = run_meta([*_cfg(home), "indexes"], capsys)
    assert code == 0 and out.splitlines()[2] == (
        f"gamma [unavailable] {gamma['stateReason']}, {MINILM}, English — Gamma documents"
    )


def test_a_missing_indexes_toml_is_a_clean_error(tmp_path, capsys):
    missing = tmp_path / "nowhere" / "indexes.toml"
    for command in (["list"], ["status"], ["indexes"]):
        code, out, err = run_meta(["--config", str(missing), "--index", "a", *command], capsys)
        assert (code, out) == (1, ""), command
        assert str(missing) in err and "Traceback" not in err, command


def test_status_with_index_fails_on_a_file_that_does_not_load(home, capsys):
    home.write("[index.alpha\n")
    code, out, err = run_meta([*_cfg(home), "--index", "x", "status"], capsys)
    assert (code, out) == (1, "") and str(home.path) in err


def test_a_term_after_the_end_of_options_still_reaches_find(home, capsys):
    root = home.roots["alpha"]
    (root / "flags.md").write_text("# Flags\n\nPass -verbose to see more.", encoding="utf-8")
    assert run_meta([*_cfg(home), "--index", "alpha", "sync"], capsys)[0] == 0
    code, out, err = run_meta(
        [*_cfg(home), "--index", "alpha", "find", "--json", "--", "-verbose"], capsys
    )
    assert code == 0, err
    assert [d["displayPath"] for d in json.loads(out)["documents"]] == ["flags.md"]


@pytest.fixture
def served(monkeypatch) -> list[FastMCP]:
    """The servers `run` was called on, none of them run."""
    started: list[FastMCP] = []
    monkeypatch.setattr(FastMCP, "run", lambda self, *a, **kw: started.append(self))
    return started


async def test_index_without_a_command_serves_that_index_alone_on_stdio(home, capsys, served):
    code, out, err = run_meta([*_cfg(home), "--index", "beta"], capsys)
    assert (code, out) == (0, ""), err
    [server] = served
    beta_md = home.roots["beta"] / "beta.md"
    async with Client(server) as c:
        await c.call_tool("ingest_file", {"filePath": str(beta_md)})
        found = (await c.call_tool("query_documents", {"query": "bananas"})).data
        listed = (await c.call_tool("list_indexes", {})).data
        other = await c.call_tool(
            "query_documents", {"query": "x", "index": "alpha"}, raise_on_error=False
        )
    assert found["index"] == "beta"
    assert [s["source"] for s in found["sources"]] == [str(beta_md)]
    assert listed["default"] == "beta"
    assert [(r["name"], r["reachable"]) for r in listed["indexes"]] == [
        ("alpha", False),
        ("beta", True),
        ("gamma", False),
    ]
    assert other.is_error and other.content[0].text == (
        "This server serves only index 'beta'. Index 'alpha' is served by its own entry in "
        "the client's MCP configuration (`minirag-mcp --index alpha`): use that server's "
        "tools, or tell the user to add that entry."
    )
    assert not (home.roots["alpha"] / ".minirag").exists()


async def test_the_stdio_server_of_one_index_points_a_path_of_another_at_its_entry(
    home, capsys, served
):
    run_meta([*_cfg(home), "--index", "beta"], capsys)
    [server] = served
    alpha_md = home.roots["alpha"] / "alpha.md"
    async with Client(server) as c:
        r = await c.call_tool("read_file", {"filePath": str(alpha_md)}, raise_on_error=False)
    assert r.is_error and r.content[0].text == (
        f"Index 'beta': {alpha_md} is outside its roots ({home.roots['beta']}) but under "
        "those of index 'alpha', which is served by its own entry in the client's MCP "
        "configuration (`minirag-mcp --index alpha`): use that server's tools, or tell the "
        "user to add that entry."
    )


@pytest.mark.parametrize("problem", ["unknown index", "missing file"])
def test_index_without_a_command_fails_before_serving(home, tmp_path, capsys, served, problem):
    config = str(home.path if problem == "unknown index" else tmp_path / "missing.toml")
    code, out, err = run_meta(["--config", config, "--index", "delta"], capsys)
    assert (code, out, served) == (1, "", [])
    if problem == "unknown index":
        assert "Unknown index 'delta'. Reachable indexes: alpha, beta, gamma." in err
    else:
        assert config in err


# It reads sys.argv on purpose, which cyclopts warns about under pytest.
@pytest.mark.filterwarnings("ignore:Cyclopts application invoked without tokens")
def test_the_entry_point_reads_the_options_around_the_command(home, capsys, monkeypatch, served):
    monkeypatch.setattr(sys, "argv", ["minirag-mcp", *_cfg(home), "--index", "beta"])
    with pytest.raises(SystemExit) as exited:
        main()
    assert exited.value.code == 0 and len(served) == 1


def test_config_alone_is_refused_rather_than_serving_the_environment(home, capsys, served):
    code, out, err = run_meta(_cfg(home), capsys)
    assert (code, out, served) == (1, "", [])
    assert "--index NAME" in err
