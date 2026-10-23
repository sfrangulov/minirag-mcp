"""Build dist/minirag-mcp.mcpb, the Claude Desktop extension.

    uv run python mcpb/build.py [--out PATH] [--keep-staging DIR]

Needs Node.js for `npx`: packing and manifest validation are done by the official
MCPB CLI, pinned below.

The bundle is a `uv` server: it carries this project's wheel, a pyproject.toml that
depends on it, and a uv.lock; Claude Desktop runs `uv sync` in the unpacked bundle
at install time and installs the dependencies for the user's own platform. Nothing
compiled is shipped in the bundle itself.

The lock is resolved under the pins of the repository's uv.lock, so the extension
runs the dependency versions the test suite ran against, not whatever the index
offers on the day of the build.

manifest.json carries no version: pyproject.toml holds the only version number, and
the build writes it into the packed manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "mcpb"

MCPB_CLI = "@anthropic-ai/mcpb@2.1.2"

# Capped at the newest Python the CI matrix tests. uv picks the interpreter for the
# bundle from this range, downloading one when the machine has none in it.
REQUIRES_PYTHON = ">=3.11,<3.14"
PYTHONS = ("3.11", "3.12", "3.13")

# A dependency without a wheel for the user's machine fails at install time, inside
# Claude Desktop, with no compiler to fall back on, and `uv lock` cannot see it. The
# manifest's `platforms` names an OS, not an architecture, so it admits both lists.
SUPPORTED_TARGETS = ("aarch64-apple-darwin", "x86_64-pc-windows-msvc")
# lancedb publishes no wheel for either of these.
UNSUPPORTED_TARGETS = ("x86_64-apple-darwin", "aarch64-pc-windows-msvc")

# The oldest macOS the bundle installs on, set by onnxruntime's arm64 wheels.
# --python-platform carries an architecture but no OS version, and uv otherwise
# assumes an older macOS than those wheels declare.
MACOS_FLOOR = "14.0"

# `uv run python mcpb/build.py` exports the repository's own environment, which the
# uv commands below must not mistake for the bundle's.
ENV = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def run(cmd: list[str], cwd: Path, env: dict[str, str] = ENV) -> subprocess.CompletedProcess:
    # Resolved here because Windows cannot start npx.cmd by its bare name.
    exe = shutil.which(cmd[0])
    if exe is None:
        sys.exit(f"{cmd[0]} is not on PATH")
    return subprocess.run([exe, *cmd[1:]], cwd=cwd, env=env, capture_output=True, text=True)


def check(cmd: list[str], cwd: Path) -> str:
    proc = run(cmd, cwd)
    if proc.returncode != 0:
        sys.exit(f"{' '.join(cmd)}\nexited {proc.returncode}\n{proc.stdout}{proc.stderr}")
    return proc.stdout


def pins() -> list[str]:
    out = check(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--format",
            "requirements.txt",
        ],
        ROOT,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def bundle_pyproject(version: str, wheel: str, constraints: list[str]) -> str:
    lines = [
        "[project]",
        'name = "minirag-mcp-desktop-extension"',
        f'version = "{version}"',
        f'requires-python = "{REQUIRES_PYTHON}"',
        f'dependencies = ["minirag-mcp=={version}"]',
        "",
        "[tool.uv]",
        "package = false",
        # Claude Desktop installs with a bare `uv sync`, so the bundle's own settings
        # are the only way to move bytecode compilation off the first start, which
        # Desktop waits on, and into the install, which shows a progress message.
        "compile-bytecode = true",
        # A machine with no wheel for some package fails the install at once, naming
        # the package, rather than attempting a native build it has no toolchain for.
        "no-build = true",
        "constraint-dependencies = [",
        *(f"    {json.dumps(c)}," for c in constraints),
        "]",
        "",
        "[tool.uv.sources]",
        f'minirag-mcp = {{ path = "wheels/{wheel}" }}',
        "",
    ]
    return "\n".join(lines)


def installs_from_wheels(staging: Path, target: str, python: str) -> str | None:
    """None when the locked set installs on `target` without building anything."""
    cmd = [
        "uv",
        "sync",
        "--frozen",
        "--dry-run",
        "--no-build",
        "--python-platform",
        target,
        "--python",
        python,
    ]
    proc = run(cmd, staging, {**ENV, "MACOSX_DEPLOYMENT_TARGET": MACOS_FLOOR})
    if proc.returncode == 0:
        return None
    errors = [line for line in proc.stderr.splitlines() if line.startswith("error")]
    return errors[0] if errors else proc.stderr.strip()


def check_wheels(staging: Path) -> None:
    failures = [
        f"{target}, Python {python}: {error}"
        for target in SUPPORTED_TARGETS
        for python in PYTHONS
        if (error := installs_from_wheels(staging, target, python))
    ]
    if failures:
        sys.exit("The locked dependencies need a compiler on:\n" + "\n".join(failures))
    for target in UNSUPPORTED_TARGETS:
        error = installs_from_wheels(staging, target, PYTHONS[-1])
        state = f"still not installable ({error})" if error else "NOW INSTALLABLE"
        print(f"{target}: {state}")


def build(out: Path, staging: Path) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = project["version"]

    wheels = staging / "wheels"
    check(["uv", "build", "--wheel", "--out-dir", str(wheels)], ROOT)
    (wheel,) = [p.name for p in wheels.glob("*.whl")]

    template = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if "version" in template:
        sys.exit("mcpb/manifest.json must not carry a version; pyproject.toml supplies it")
    manifest = {}
    for key, value in template.items():
        manifest[key] = value
        if key == "name":
            manifest["version"] = version
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    (staging / "server").mkdir()
    shutil.copy2(HERE / "server" / "main.py", staging / "server" / "main.py")
    (staging / "pyproject.toml").write_text(
        bundle_pyproject(version, wheel, pins()), encoding="utf-8"
    )
    check(["uv", "lock"], staging)
    check_wheels(staging)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    print(check(["npx", "--yes", MCPB_CLI, "pack", str(staging), str(out)], ROOT), end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "minirag-mcp.mcpb")
    parser.add_argument(
        "--keep-staging",
        type=Path,
        help="assemble the bundle in this (new) directory and leave it there",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if args.keep_staging:
        args.keep_staging.mkdir(parents=True)
        build(out, args.keep_staging.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="minirag-mcpb-") as tmp:
            build(out, Path(tmp))


if __name__ == "__main__":
    main()
