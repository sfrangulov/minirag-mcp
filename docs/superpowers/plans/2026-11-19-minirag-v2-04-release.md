# minirag v2 — release and migration (plan 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v2 as the test pre-release `2.0.0a1` without disturbing 1.1.1 as the default install. Then move the owner's three corpora and clients to it, one verification at a time, with 1.x kept as the rollback.

**Architecture:** Part A changes only release tooling and docs in the repository. It adds pre-release parts to `bump-my-version`, a version guard in `publish-mcp.yml`, a v2 README section (around plan 3's LaunchAgent template), the joint check of the final pipeline with the English baseline re-taken (spec section 6), then the PR and the tag. Part B changes nothing in the repository. It is operational work on the owner's Mac (g7q.13): backups, router presets, `indexes.toml`, a server LaunchAgent, v2 index builds, a measurement on the private ruler, the client switch, and a recorded rollback. Every command in Part B states the output that means it passed.

**Tech Stack:** bump-my-version 1.5.1, GitHub Actions, PyPI trusted publishing, the official MCP Registry, uv/uvx, launchd, llama.cpp router (`llama-server --models-preset`), LanceDB, fastmcp `Client`.

**Spec:** `docs/superpowers/specs/2026-11-19-minirag-v2-design.md` (section 7, anti-goals), with the shared names in `docs/superpowers/plans/2026-11-19-minirag-v2-00-contract.md`. Plans 1–3 are finished before this one starts; this plan uses their CLI and status names and does not redefine them.

## Global Constraints

- v2 ships as a test (pre-release) version; 1.1.1 stays what `pip install minirag-mcp` and `uvx minirag-mcp` resolve to.
- Versions `2.0.0aN` are tagged from `main`; there is no long-lived v2 branch. A 1.x patch, if ever, comes from `release/1.x` cut from the `v1.1.1` tag.
- `pyproject.toml` holds the only version number; releases go through `bump-my-version`. A PyPI version and a registry version can never be reused.
- v2 must never rewrite a 1.x index. The owner's 1.x indexes and old configs stay until v2 is confirmed, so rolling back is a config edit.
- Router: `--models-max 4`; at least 8192 tokens per classify slot.
- Server defaults (spec): `127.0.0.1:8765`, `indexes.toml` and `token` in the minirag user config directory, URLs `/mcp/<index>`.
- Private evaluation material, machine paths, corpus names and client data never enter the repository. Part B names them by placeholder; concrete values live in the private handover (see the bd memory `enterpriserag-bench-handover` for its location).
- Every step that edits a file outside the repository first runs a backup command into `$BACKUPS`.
- English everywhere; comments state constraints only; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; the PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

## Review Focus

1. **A reader of the README on `main`, or someone following the registry's `websiteUrl`, runs `uvx minirag-mcp` and gets 1.1.1, whose defaults differ from the page.** The page must say so before anything else. Owner: Task 3, Step 4.
2. **A pre-release could reach the registry through a manual dispatch or a semver-style tag spelling.** The guard keys on the version, not on the trigger. Owner: Task 1, Steps 2 and 4, which test six spellings.
3. **Unpinned installs, the owner's included, could move to 2.0.0a1 once it is on PyPI.** They must keep resolving 1.1.1. Owner: Task 6, Step 7.
4. **A typo in the router presets takes down every model the router serves, the owner's chat model included.** llama.cpp exits on an unknown key, and KeepAlive turns that into a restart loop. Owner: Task 8, Steps 3 and 5 (throwaway router first, chat model checked after the reload).
5. **A helper script that builds its minirag command from a client entry breaks when that entry becomes an HTTP URL.** It must be found and repointed before the switch. Owner: Task 7, Step 1 and Task 13, Step 4.

## Registry finding (researched 2026-11-19; Task 1 re-verifies and records it)

- `2.0.0a1` is the PEP 440 spelling, and the registry does not accept it as semver. Its `IsSemanticVersion` uses Go's `golang.org/x/mod/semver`, which requires a `-` before the pre-release part; `0a1` is not a numeric patch.
- The registry's code (`internal/service/versioning.go`, `CompareVersions`, unchanged since 2025-08-26; deployed release v1.8.9) says that "if one is semver and one is not, the semver version is always considered higher". A new version gets `isLatest` only when it compares higher than the current latest. By the code, `2.0.0a1` would therefore **not** displace `1.1.1`.
- The registry's own guide (`docs/modelcontextprotocol-io/versioning.mdx`) says the opposite: "If parsing fails, the version will always be marked as 'latest'."
- The semver spelling `2.0.0-alpha.1` *would* outrank `1.1.1`, because semver puts it between 1.1.1 and 2.0.0.
- Live state: `/v0.1/servers/io.github.sfrangulov%2Fminirag-mcp/versions/latest` is `1.1.1`, `isLatest: true`. The entry holds 8 versions; nothing is ever withdrawn.
- **Decision:** `publish-mcp.yml` registers only `vMAJOR.MINOR.PATCH` tags. There is no spelling for which "safe" is verified: the code and the docs disagree, and the only test is an irreversible publish. A test release also offers nothing to registry users.

---

## Part A — release mechanics (repository, branch `feat/v2-retrieval-experiments`)

Commands below assume `SCRATCH=$(mktemp -d)` has run once in the shell.

### Task 1: Record the registry finding and register final releases only

**Files:**
- Modify: `.github/workflows/publish-mcp.yml` (new step after "Assert server.json agrees with the tag being published"; `if:` on the four steps after it)
- Modify: `README.md` (section "The MCP Registry entry")

**Interfaces:**
- Produces: the step `id: kind`, whose output `final` is `'true'` or `'false'`. Every later step in the job carries `if: steps.kind.outputs.final == 'true'`.

- [ ] **Step 1: Re-verify the finding**

```bash
gh api repos/modelcontextprotocol/registry/contents/internal/service/versioning.go --jq .content | base64 -d | sed -n '/^func CompareVersions/,/^}/p' | grep -n "semver is always higher"
gh api repos/modelcontextprotocol/registry/contents/docs/modelcontextprotocol-io/versioning.mdx --jq .content | base64 -d | grep -n "always be marked as"
curl -s "https://registry.modelcontextprotocol.io/v0.1/servers/io.github.sfrangulov%2Fminirag-mcp/versions/latest" | jq -r '.server.version, ._meta["io.modelcontextprotocol.registry/official"].isLatest'
```

Expected: one line from the code, one from the guide, then `1.1.1` and `true`. If either text has changed, update the Registry finding section above before going on. The decision stands either way unless the guide now *explicitly* excludes pre-releases from `isLatest`; in that case stop and ask the owner.

- [ ] **Step 2: Write the failing check**

```bash
uv run --with pyyaml python -c "import yaml; s = yaml.safe_load(open('.github/workflows/publish-mcp.yml'))['jobs']['publish']['steps']; print(next(x['run'] for x in s if x.get('id') == 'kind'))" > "$SCRATCH/kind.sh"
```

Expected now: `StopIteration`, because no step has `id: kind` yet.

- [ ] **Step 3: Add the step and gate the publishing steps**

Add this step right after the server.json assertion, so that a pre-release tag still has its server.json versions checked:

```yaml
      # The registry's code ranks a non-semver version (2.0.0a1) below every semver one,
      # but its versioning guide says such a version is always marked latest, and a semver
      # pre-release (2.0.0-alpha.1) outranks 1.1.1 by semver itself. A registry version
      # can never be withdrawn, so only vMAJOR.MINOR.PATCH is registered.
      - name: Register final releases only
        id: kind
        env:
          REF_NAME: ${{ github.ref_name }}
        run: |
          version=${REF_NAME#v}
          if [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "final=true" >> "$GITHUB_OUTPUT"
          else
            echo "final=false" >> "$GITHUB_OUTPUT"
            echo "::notice::${version} is not a final release; the MCP Registry entry is left as it is"
            echo "${version} is not a final release: not registered with the MCP Registry." >> "$GITHUB_STEP_SUMMARY"
          fi
```

Add `if: steps.kind.outputs.final == 'true'` to the four later steps: "Wait for the release on PyPI…", "Install mcp-publisher", "Authenticate…" and "Publish server.json…".

- [ ] **Step 4: Run the check and actionlint**

```bash
uv run --with pyyaml python -c "import yaml; s = yaml.safe_load(open('.github/workflows/publish-mcp.yml'))['jobs']['publish']['steps']; print(next(x['run'] for x in s if x.get('id') == 'kind'))" > "$SCRATCH/kind.sh"
for t in v1.1.2 v2.0.0 v2.0.0a1 v2.0.0rc1 v2.0.0-alpha.1 v2.0.0.dev1; do
  printf '%s ' "$t"; REF_NAME=$t GITHUB_OUTPUT=/dev/stdout GITHUB_STEP_SUMMARY=/dev/null bash "$SCRATCH/kind.sh" | grep '^final='
done
grep -c "steps.kind.outputs.final == 'true'" .github/workflows/publish-mcp.yml
actionlint .github/workflows/publish-mcp.yml && echo actionlint-ok
```

Expected: `v1.1.2 final=true`, `v2.0.0 final=true`, and `final=false` for the other four. Then `4`, then `actionlint-ok`.

- [ ] **Step 5: Record the finding in the README and on the epic**

Add one paragraph to "The MCP Registry entry". It says that pre-release tags are not registered and that their run ends green with a notice. It gives the reason in two sentences (code vs guide, `2.0.0-alpha.1` outranking 1.1.1), names the check date and registry release v1.8.9, and states that `server.json` is still version-checked on a pre-release tag. Then:

```bash
bd note minirag-mcp-g7q "Registry and pre-releases (2026-11-19, registry v1.8.9): code ranks non-semver 2.0.0a1 below semver 1.1.1; versioning guide says non-semver is always latest; 2.0.0-alpha.1 would outrank 1.1.1. publish-mcp.yml now registers only vMAJOR.MINOR.PATCH."
```

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/publish-mcp.yml README.md
git commit -m "ci(registry): register final releases only" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: Pre-release parts for bump-my-version; release.yml verified as is

**Files:**
- Modify: `pyproject.toml` (`[tool.bumpversion]`: `parse`, `serialize`, two `parts` tables)
- Modify: `README.md` (section "Releasing": step 1 and step 3, and a short "A 1.x patch" paragraph)
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Produces: bump parts `pre_l` (values `a`, `b`, `rc`, `final`; `final` is omitted from the version) and `pre_n` (first value 1). `bump major|minor|patch` still produces a final version. A cycle starts with `bump --new-version X.Y.Za1`.

- [ ] **Step 1: Write the failing test**

Add this to `tests/test_scaffold.py`, reusing its `ROOT` and `PYPROJECT`:

```python
BUMP = Path(sys.executable).with_name("bump-my-version")

@pytest.mark.parametrize(
    ("current", "part", "expected"),
    [
        ("1.1.1", "patch", "1.1.2"),
        ("1.1.1", "major", "2.0.0"),
        ("2.0.0a1", "pre_n", "2.0.0a2"),
        ("2.0.0a2", "pre_l", "2.0.0b1"),
        ("2.0.0b1", "pre_l", "2.0.0rc1"),
        ("2.0.0rc1", "pre_l", "2.0.0"),
        ("2.0.0", "patch", "2.0.1"),
    ],
)
def test_bump_walks_pre_releases_to_the_final_release(current, part, expected):
    proc = subprocess.run(
        [str(BUMP), "show", "--config-file", str(PYPROJECT), "--current-version", current,
         "--increment", part, "-f", "json", "new_version"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["new_version"] == expected
```

The docstring states the one constraint the values encode: `release.yml` compares the tag with the wheel filename, which carries the normalized PEP 440 form, so `2.0.0a1` and not `2.0.0-alpha.1`.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_scaffold.py -k bump_walks -q`
Expected: `4 failed, 3 passed`. The `pre_n`/`pre_l` cases exit 2.

- [ ] **Step 3: Configure the parts in `[tool.bumpversion]`**

```toml
parse = '(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:(?P<pre_l>a|b|rc)(?P<pre_n>\d+))?'
serialize = ["{major}.{minor}.{patch}{pre_l}{pre_n}", "{major}.{minor}.{patch}"]

[tool.bumpversion.parts.pre_l]
values = ["a", "b", "rc", "final"]
optional_value = "final"
# major/minor/patch reset pre_l to its first value; "final" keeps `bump patch` a final release.
first_value = "final"

[tool.bumpversion.parts.pre_n]
first_value = "1"
```

The existing `[[tool.bumpversion.files]]` entries stay as they are; they already match any version string.

- [ ] **Step 4: Run the test, then the suite**

Run: `uv run pytest tests/test_scaffold.py -q && uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Rehearse release.yml's version check on a pre-release build (expected: no change needed)**

```bash
git worktree add "$SCRATCH/wt" HEAD
(cd "$SCRATCH/wt" && uv run bump-my-version bump --new-version 2.0.0a1 --no-commit --no-tag && uv build -q && ls dist)
uv run --with pyyaml python -c "import yaml; s = yaml.safe_load(open('.github/workflows/release.yml'))['jobs']['build']['steps']; print(next(x['run'] for x in s if x.get('name') == 'Assert the built version matches the tag being released'))" > "$SCRATCH/assert.sh"
(cd "$SCRATCH/wt" && REF=refs/tags/v2.0.0a1 REF_NAME=v2.0.0a1 bash "$SCRATCH/assert.sh" && echo match)
(cd "$SCRATCH/wt" && REF=refs/tags/v2.0.0-alpha.1 REF_NAME=v2.0.0-alpha.1 bash "$SCRATCH/assert.sh"; echo "exit $?")
git worktree remove --force "$SCRATCH/wt"
```

Expected: `minirag_mcp-2.0.0a1-py3-none-any.whl` and `minirag_mcp-2.0.0a1.tar.gz`. Then `match`. Then `built version '2.0.0a1' does not match tag 'v2.0.0-alpha.1'` followed by `exit 1`. So release.yml already accepts a PEP 440 pre-release tag and refuses any other spelling, and stays unchanged. Its header comment already explains why pre-releases publish.

- [ ] **Step 6: Update "Releasing" in the README**

In step 1, add the pre-release commands, with one comment each:

```bash
uv run bump-my-version bump --new-version 2.0.0a1   # start a pre-release cycle
uv run bump-my-version bump pre_n                   # 2.0.0a1 -> 2.0.0a2
uv run bump-my-version bump pre_l                   # a -> b -> rc -> final: 2.0.0rc1 -> 2.0.0
```

Add two facts. First, `pre_n` and `pre_l` only make sense on a pre-release: on a final version they produce an invalid version such as `1.1.1final2`, and the way back is `git tag -d` plus `git reset --hard HEAD~1` before anything is pushed. Second, the tag must be the PEP 440 spelling (`v2.0.0a1`).

In step 3, add: for a pre-release, `gh release create vX --verify-tag --prerelease --notes-file <notes>`. Marking it as a pre-release keeps GitHub's "Latest" on the last final release.

Add a short paragraph "A 1.x patch" taken from spec section 7: cut `release/1.x` from `v1.1.1`, cherry-pick, bump `patch` there, and tag from that branch. `release.yml` is already on `main`, which is what GitHub requires.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src tests && uv run ruff format --check src tests
git add pyproject.toml tests/test_scaffold.py README.md
git commit -m "build(release): pre-release parts for bump-my-version" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: README v2 section, reference sections, stale-fact sweep

> **Carried in from plan 1's review (2026-11-19):** `server.json` still defaults `MODEL_NAME`
> to MiniLM (move to e5 or drop the default); stale "~220 MB" slow-suite text in CLAUDE.md and
> tests/conftest.py, and the labels/model.py docstring calling huggingface_hub an optional
> extra; document that DB_PATH must not be a mount point (rebuild swaps rename the folder) and
> make no power-loss durability claims (macOS fsync is not F_FULLFSYNC). The owner granted full
> rights on 2026-11-19: steps that wait for the owner's go-ahead proceed without waiting.

> **Added 2026-11-20 (owner):** the README gets its own section on setting up llama-server
> from scratch, at least for macOS (Apple silicon, Metal): installing llama.cpp (Homebrew),
> getting the three GGUFs (multilingual-e5-large for embeddings, bge-reranker-v2-m3 for
> reranking, the classify model pinned in `classify.toml`), a router preset file with one
> section per model (embedding / rerank with `pooling = rank` / classify with >= 8192 tokens
> per slot) and `--models-max`, an API key file, running it under launchd, checking each route
> with `curl` (`/v1/embeddings`, `/v1/rerank`, `/props`), and pointing minirag at it
> (`indexes.toml` `[embedding."<id>"]`, `[rerank]`, `[classify]`, and the env-var equivalents).
> Every command in the section is run on this Mac against a throwaway router on a spare port
> (never the owner's live router), and the section says which steps were verified where. No
> machine paths of the owner's in the text. bd minirag-mcp-g7q.15.

**Files:**
- Modify: `README.md` (banner below the badges; new `## v2 (pre-release)` before `## Requirements`; Configuration, MCP Tools, CLI and Chunking sections; fixes from the sweep)

**Interfaces:**
- Consumes: plan 3's `minirag-mcp serve`, `minirag-mcp connect <index> [--allow a,b]`, `--index NAME`, `--config`, the `indexes` command, the `list_indexes` tool and the `index` argument, the `indexes.toml` schema, the URL forms, the token file and the LaunchAgent template `docs/launchd/local.minirag-mcp.plist` (label `local.minirag-mcp`, placeholders `__UVX__`, `__VERSION__`, `__HOME__`; Task 11 copies it). Plan 1's environment variables (`EMBED_URL`, `EMBED_ENDPOINT_MODEL`, `RERANK_URL`, `RERANK_MODEL`, `CLASSIFY_URL`, `CLASSIFY_MODEL`, `ENDPOINT_API_KEY_FILE`, `RAG_DOCUMENT_WEIGHT`, `RAG_RERANK_TOP`) and defaults (`MODEL_NAME`, `DB_PATH`, `CHUNK_TOKEN_BUDGET`). Plan 2 already documented `RAG_SYNC_WORKERS`, `RAG_OCR_LANG` and the OCR section.
- Produces: README anchor `#v2-pre-release`.

- [ ] **Step 1: Check the names against the code**

```bash
uv run minirag-mcp serve --help && uv run minirag-mcp connect --help && uv run minirag-mcp indexes --help
MINIRAG_CONFIG="$SCRATCH/none.toml" uv run minirag-mcp --index x status; echo "exit $?"
grep -n "indexes.toml\|minirag-mcp serve\|minirag-mcp connect\|RERANK_URL\|RAG_SYNC_WORKERS\|tesseract" README.md
sed -e "s|__UVX__|$(command -v uvx)|" -e "s|__VERSION__|2.0.0a1|" -e "s|__HOME__|$HOME|" docs/launchd/local.minirag-mcp.plist > "$SCRATCH/agent.plist" && plutil -lint "$SCRATCH/agent.plist"
```

Expected: the three help texts print. The `--index x` command exits 1 naming the missing `$SCRATCH/none.toml`, not an empty result. The grep shows only plan 2's lines (`RAG_SYNC_WORKERS`, `RAG_OCR_LANG`, the OCR section); plans 1 and 3 documented nothing in the README, so this task writes their parts. The filled template prints `OK`.

- [ ] **Step 2: Bring the reference sections up to v2**

- **Configuration** table: `MODEL_NAME` default `intfloat/multilingual-e5-large` (2.24 GB first download; MiniLM by setting it); `DB_PATH` default `<first root>/.minirag/v2` (a 1.x index there is refused with a message naming the v2 path); `CHUNK_TOKEN_BUDGET` default plan 1's measured value for e5, ceiling the model's window (e5 512, MiniLM 128, unknown 128); `CACHE_DIR` model size; new rows `RAG_DOCUMENT_WEIGHT` (0.5, 0..1, 0 disables), `RAG_RERANK_TOP` (30, 1..200), `EMBED_URL` + `EMBED_ENDPOINT_MODEL`, `RERANK_URL` + `RERANK_MODEL`, `CLASSIFY_URL` + `CLASSIFY_MODEL`, `ENDPOINT_API_KEY_FILE` (a URL without its model name is an error), `MINIRAG_CONFIG` (the `indexes.toml` path).
- **Search Tuning**: one subsection each for `RAG_DOCUMENT_WEIGHT` and `RAG_RERANK_TOP` (the fusion formula of spec section 4; what `reranked`, `rerankScore` and `warnings` in a result mean).
- **MCP Tools**: 13 tools; `list_indexes`; every tool's optional `index`; every result's `index`.
- **CLI**: `serve`, `connect`, `indexes`, and `--index` / `--config` before or after the command.
- **Chunking**: the window comes from the model, not a fixed 128.

Run: `uv run pytest tests/test_scaffold.py -q`
Expected: pass.

- [ ] **Step 3: Write `## v2 (pre-release)`**

The section has five subsections, with these facts:
1. **Installing.**
   - `uvx 'minirag-mcp>=2.0.0a1'` gets the newest pre-release. A specifier that names a pre-release admits pre-releases of minirag-mcp only, never of its dependencies.
   - Extras: `uvx --from 'minirag-mcp[ocr]>=2.0.0a1' minirag-mcp`.
   - pip: `pip install --pre minirag-mcp`.
   - One exact pre-release: `uvx minirag-mcp==<version>`, with the version taken from PyPI's release history.
   - Without any of these you get 1.1.1.
2. **Next to 1.x.**
   - The v2 index lives in `<first root>/.minirag/v2/`, and v2 never opens 1.x's `.minirag/lancedb` for writing.
   - A `DB_PATH` pointing at a 1.x index is refused.
   - The first sync builds a fresh index with `intfloat/multilingual-e5-large`: a 2.24 GB first download unless an endpoint serves it.
   - Going back means dropping the pin. The 1.x index is still there; a 1.x `sync` catches up what was added in between.
3. **One server, several indexes.**
   - A trimmed `indexes.toml` from spec section 1, `minirag-mcp serve`, and the URLs `/mcp/<index>`, `?allow=a,b` and `/mcp`. The token file is mode 0600.
   - Claude Code: `claude mcp add --transport http minirag http://127.0.0.1:8765/mcp/<index> --header "Authorization: Bearer $(cat <token file>)"`.
   - A project `.mcp.json` uses `"headers": {"Authorization": "Bearer ${MINIRAG_TOKEN}"}`, so the token never lands in a file that may be committed.
   - Codex connects over HTTP the same way (URL plus the bearer header); Claude Desktop and any stdio-only client use `minirag-mcp connect <index> [--allow a,b]` (spec section 1, Clients).
   - `list_indexes` shows what a connection may use; a refused index names the reachable ones.
4. **Model endpoints.**
   - The `[embedding."<fastembed id>"]`, `[rerank]` and `[classify]` tables, and their env-var equivalents for the single-index mode.
   - Fallbacks are loud, and the served embedding model is verified by the recorded probe.
   - The llama.cpp router presets: a reranker section (`rerank = true`, `pooling = rank`) and a classify section (at least 8192 context per slot), plus `--models-max 4` in the router's own arguments.
   - Validate an edited preset file on a throwaway router on a spare port with `--no-models-autoload` first: an unknown key makes the router exit at startup.
   - Reload with `launchctl bootout` and then `bootstrap`; `kickstart -k` does not re-read a plist.
5. **Running it at login.**
   - Copy `docs/launchd/local.minirag-mcp.plist` to `~/Library/LaunchAgents/`, fill `__UVX__`, `__VERSION__` and `__HOME__` with absolute values (launchd expands neither `~` nor `$HOME`), run `plutil -lint`, then `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.minirag-mcp.plist`.
   - Re-read `indexes.toml` with `launchctl kill HUP gui/$(id -u)/local.minirag-mcp`.
   - Upgrade by editing the pin, then bootout and bootstrap. Stop with bootout.

- [ ] **Step 4: Add the banner (exact copy) below the badges, keeping the `mcp-name` marker first**

```markdown
> **This page describes minirag v2, a test pre-release (`2.0.0aN`).** `pip install minirag-mcp`
> and `uvx minirag-mcp` still install **1.1.1**, documented in
> [its own README](https://github.com/sfrangulov/minirag-mcp/blob/v1.1.1/README.md).
> To try v2, see [v2 (pre-release)](#v2-pre-release).
```

- [ ] **Step 5: Sweep for 1.x facts and broken anchors**

```bash
grep -n "MiniLM\|paraphrase-multilingual\|\.minirag/lancedb\|128-token\|rapidocr\|eslav\|220 MB" README.md
uv run python - <<'EOF'
import re, pathlib
t = pathlib.Path("README.md").read_text(encoding="utf-8")
anchors = {re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-") for h in re.findall(r"^#+ (.+)$", t, re.M)}
anchors |= set(re.findall(r'<a id="([^"]+)"', t))
print(sorted(set(re.findall(r"\]\(#([^)]+)\)", t)) - anchors))
EOF
uv run pytest tests/test_scaffold.py -q
```

Expected: every remaining grep hit talks about 1.x or about MiniLM as an opt-in; fix any other hit in place. The anchor check prints `[]`, which is its output on today's README. The scaffold tests pass; the `mcp-name` marker is intact.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: v2 pre-release section and reference sections" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Joint check of the final pipeline on both stands; the English baseline re-taken

Spec section 6: before the re-index, one joint check of the final pipeline on both stands, and `evals/enterpriserag/baseline.json` re-taken after the retrieval changes (it predates 2ee521f's tie order and v2's model). Plan 1 Task 13 measured the chunk budgets before plan 2 changed OCR and sync, and plan 1 Task 12 left the baseline alone; this task checks what ships, on this branch with plans 1–3 merged.

**Files:**
- Modify: `evals/enterpriserag/baseline.json` (re-taken)
- Outside the repository (private): one run of plan 1 Task 13's `v2_measure.py` under the private ruler (`<ruler>`; its location is in the bd memory about the ruler, `bd memories ruler`)
- Record: bd notes on `minirag-mcp-g7q.1` and `minirag-mcp-g7q.5`

**Interfaces:**
- Consumes: plan 1 Task 12's `recall.py --subset` (a run is an experiment when a `CORE_ENV` variable is set), plan 1 Task 13's chosen budget and per-budget numbers in g7q.1's notes and its Step 1 environment; plan 2's parallel sync and Tesseract OCR through the CLI.
- Produces: `baseline.json` on the shipped defaults; the aggregate numbers Task 12 compares against.

- [ ] **Step 1: English, the defaults anyone gets**

Run, with no `CORE_ENV` variable exported (in-process e5, no reranker): `uv run python evals/enterpriserag/recall.py --subset`, then `uv run python evals/enterpriserag/recall.py --subset --update-baseline`, then `git diff --stat evals/enterpriserag/baseline.json`.
Expected: the first run prints its deltas against the old baseline (MiniLM, taken before 2ee521f) and a fingerprint warning; a drop beyond the gate's thresholds is explained (bd note on g7q.1) before the update. The second writes the file; the diff shows `baseline.json` changed, its fingerprint naming `intfloat/multilingual-e5-large` and plan 1's budget.

- [ ] **Step 2: English, the owner's configuration, and the keyword weight**

Export plan 1 Task 13 Step 1's environment (`EMBED_URL`, `EMBED_ENDPOINT_MODEL`, `RERANK_URL`, `RERANK_MODEL`, `ENDPOINT_API_KEY_FILE`) and run `recall.py --subset` once, then once each with `RAG_HYBRID_WEIGHT=0.45` and `0.75` (g7q.5: English was to be re-tuned with the rest).
Expected: at 0.6, recall@10 within 1.0 point and MRR within 0.02 of plan 1 Task 13's English numbers at the chosen budget (the regression gate's thresholds). A larger drop stops the task: find the cause with superpowers:systematic-debugging, fix it test-first, and repeat from Step 1.

- [ ] **Step 3: Russian, the final pipeline**

Run `v2_measure.py` (plan 1 Task 13 Step 3) at the chosen budget only, into a fresh `DB_PATH=<ruler>/v2-final`, with the Step 2 environment and this branch's `minirag-mcp sync` (parallel workers, Tesseract OCR); then pool, judge the new pairs and score as in plan 1 Task 13 Step 4.
Expected: query-field MRR within 0.02, default-window hit within 1 point and question-field MRR within 0.02 of plan 1's numbers at that budget. OCR changed between the two runs, so a difference is first looked for in OCR'd documents; it is reported by count only. A larger drop stops the task, as in Step 2.

- [ ] **Step 4: Record, and settle g7q.5**

```bash
bd note minirag-mcp-g7q.1 "Joint check of the final pipeline (<date>): English subset defaults recall@10 <..>, MRR <..>; with endpoint+reranker <..>/<..>; Russian ruler query MRR <..>, window <..>, question MRR <..>. baseline.json re-taken on the defaults."
bd note minirag-mcp-g7q.5 "English subset, final pipeline, keyword weight 0.45/0.6/0.75: recall@10 <..>, MRR <..>."
```

If neither 0.45 nor 0.75 beats 0.6 on English by more than the gate's thresholds, `bd close minirag-mcp-g7q.5 -r "0.6 kept: within noise on the Russian ruler and on the English subset with the final pipeline"`. Otherwise ask the owner: one default serves both stands, and the spec keeps 0.6 for the owner's corpus. Aggregates only; no document names, questions or paths.

- [ ] **Step 5: Gate and commit**

```bash
uv run pytest evals/enterpriserag -q && uv run pytest -q
git add evals/enterpriserag/baseline.json
git commit -m "chore(evals): re-take the English baseline on v2's defaults" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: all pass; one commit touching only `baseline.json`.

### Task 5: Pull request to `main`, CI green

**Files:** none new. The v2 plan files under `docs/superpowers/plans/` are committed if still untracked.

- [ ] **Step 1: Preconditions**

```bash
git status --short
git log --oneline main..HEAD | wc -l
bd list --parent minirag-mcp-g7q --status open
```

Expected: the plan and spec files are tracked (commit them as `docs: v2 implementation plans` if not), and nothing else is pending. Plan 2 closed g7q.6, .9 and .14, and Task 4 settled g7q.5; plans 1 and 3 add notes but close nothing, so close g7q.1, .2, .3, .4 and .7 now, each with `bd close <id> -r "<what landed, and in which plan>"` (g7q.4: document BM25 shipped, document vectors measured and dropped). Afterwards the only open children are g7q.8 and g7q.10 (deferred), g7q.12 (only its struck-through item shipped; the rest waits on the ruler) and g7q.13.

- [ ] **Step 2: Local gates**

```bash
uv sync --exact && uv run ruff check src tests && uv run ruff format --check src tests && uv run pytest -q
uv run pytest -m slow -q
uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['tool']['pytest']['ini_options']['markers'])"
```

Run every marker the last command lists, each with the extra and environment variable its marker text names (for example `uv run --extra classify env MINIRAG_TEST_GGUF=<gguf> pytest -m classify -q`). Then:

```bash
uv sync --exact && uv run pytest -q
V="$SCRATCH/smoke"; uv build -o "$SCRATCH/dist" && uv venv "$V" && VIRTUAL_ENV="$V" uv pip install "$SCRATCH"/dist/*.whl && "$V/bin/minirag-mcp" status --base-dir .
uv run python mcpb/build.py && uv run python mcpb/smoke.py dist/minirag-mcp.mcpb
actionlint .github/workflows/*.yml && echo actionlint-ok
```

Expected: every command exits 0. The second `pytest -q` is green with no extras installed.

- [ ] **Step 3: Push and open the PR**

Write the body to a scratch file. It lists what v2 is (spec Goals), states that 1.1.1 stays the default, links the spec and plans 0–4, and ends with the attribution line from Global Constraints.

```bash
git push -u origin feat/v2-retrieval-experiments
gh pr create --base main --title "minirag v2 (2.0.0a1)" --body-file "$SCRATCH/pr-body.md"
```

- [ ] **Step 4: Wait for CI**

```bash
gh pr checks --watch
gh run list --workflow bundle.yml --branch feat/v2-retrieval-experiments -L 1 --json conclusion -q '.[0].conclusion'
```

Expected: every check passes (8 test legs and build), then `success`. On red, fix on the branch with `superpowers:systematic-debugging`, push, and watch again.

- [ ] **Step 5: Owner checkpoint, then merge**

Ask the owner to review the PR. On approval, run `gh pr merge --merge`, then:

```bash
gh run list --workflow ci.yml --branch main -L 1 --json conclusion -q '.[0].conclusion'
```

Expected: `success`.

### Task 6: Cut 2.0.0a1

**Interfaces:**
- Produces: `minirag-mcp==2.0.0a1` on PyPI and the tag `v2.0.0a1`. Part B installs exactly this.

- [ ] **Step 1: Clean `main`**

Run: `git switch main && git pull --ff-only && git status --short`
Expected: no output from `status`.

- [ ] **Step 2: Dry run**

Run: `uv run bump-my-version bump --new-version 2.0.0a1 --dry-run --verbose`
Expected: changes to `pyproject.toml`, `uv.lock` and both `server.json` lines, a commit `chore: release v2.0.0a1` and a tag `v2.0.0a1`. Afterwards `git status --short` is empty.

- [ ] **Step 3: Bump**

```bash
uv run bump-my-version bump --new-version 2.0.0a1
git show --stat --format=%s HEAD
git tag --points-at HEAD
uv run pytest tests/test_scaffold.py -q
```

Expected: `chore: release v2.0.0a1` touching exactly `pyproject.toml`, `server.json` and `uv.lock`. Then `v2.0.0a1`. Then the tests pass.

- [ ] **Step 4: Owner checkpoint (irreversible from here), then push**

Run: `git push && git push origin v2.0.0a1`

- [ ] **Step 5: Publish the GitHub pre-release**

Write release notes to `$SCRATCH/notes-2.0.0a1.md`; the repository keeps no CHANGELOG, so these notes are the changelog. They cover:
- that this is a test release and 1.1.1 stays the default;
- the install lines from the README v2 section;
- what changed for a 1.x user: default model e5 (2.24 GB), index at `.minirag/v2/` (a 1.x index is never written), `[ocr]` is now Tesseract (`RAG_OCR_LANG` in Tesseract spelling, `eslav`/`en` still accepted), ~~struck-through `.docx` text no longer indexed~~ struck-through text still indexed but marked `~~...~~` in every chunk that holds it, double strikethrough included (owner, 2026-11-19), parallel sync (`RAG_SYNC_WORKERS`), `serve`, `connect`, `--index`, the endpoints, document BM25 and the reranker;
- deliberate departures from the spec's endpoint fallback (added 2026-11-20 by plan 2's final review; spec section 2 records them): the workers of a parallel sync never fall back to the in-process model, so while the embedding endpoint is unreachable each file they read fails, 3 such failures in a row stop the sync with `embedding: endpoint unreachable, N files not synced`, and the next sync reads those files (N workers each loading e5 would need ~3.3 GB apiece); and with an endpoint configured, the in-process fallback never downloads the model: without a local copy in the cache folder a search, an ingest or a sync run in one process fails, naming the endpoint and the way out, instead of fetching 2.2 GB;
- a rebuild stopped that way (the endpoint unreachable while a rebuild re-ingests) reports some files twice, as failed and as kept from their stored text (`kept from its stored text: ...`); no data is lost: those files are stored again from the text the index held, or, if that fails too, the rebuild is not swapped in and the previous index keeps serving;
- a link to `#v2-pre-release`.

Then run:

```bash
gh release create v2.0.0a1 --verify-tag --prerelease --title v2.0.0a1 --notes-file "$SCRATCH/notes-2.0.0a1.md"
```

- [ ] **Step 6: Watch both workflows**

```bash
gh run watch "$(gh run list --workflow release.yml -L 1 --json databaseId -q '.[0].databaseId')" --exit-status
id=$(gh run list --workflow publish-mcp.yml -L 1 --json databaseId -q '.[0].databaseId'); gh run view "$id" --json conclusion -q .conclusion; gh run view "$id" --log | grep -c "is not a final release"
```

Expected: release.yml succeeds. publish-mcp.yml prints `success` and a count of at least 1, and its publishing steps show as skipped.

- [ ] **Step 7: Verify the public state, and that the default did not move**

```bash
curl -s https://pypi.org/pypi/minirag-mcp/json | jq -r .info.version
curl -s https://pypi.org/pypi/minirag-mcp/2.0.0a1/json | jq -r '.info.version, (.urls | length)'
uvx --refresh-package minirag-mcp minirag-mcp --version
uvx --refresh-package minirag-mcp --from 'minirag-mcp[ocr,classify]' minirag-mcp --version
uvx --refresh-package minirag-mcp 'minirag-mcp==2.0.0a1' --version
uvx --refresh-package minirag-mcp 'minirag-mcp>=2.0.0a1' --version
python3 -m venv "$SCRATCH/pip" && "$SCRATCH/pip/bin/pip" install -q --dry-run --pre minirag-mcp 2>&1 | grep -o "minirag-mcp-[0-9a-z.]*"
curl -s "https://registry.modelcontextprotocol.io/v0.1/servers/io.github.sfrangulov%2Fminirag-mcp/versions/latest" | jq -r .server.version
gh release view --json tagName -q .tagName
```

Expected, in order: `1.1.1`; `2.0.0a1` and `2`; `1.1.1`; `1.1.1`; `2.0.0a1`; `2.0.0a1`; `minirag-mcp-2.0.0a1`; `1.1.1`; `v1.1.1`.

---

## Part B — the owner's migration (g7q.13; nothing here changes the repository)

**Conventions for every task below.**
- **Placeholders.** Values are written as exported shell variables. Task 7 resolves them once into an env file (`export NAME=…` lines) kept next to the private handover, and every command block assumes `source <that file>` has run in the same shell:
  - paths: `REPO`, `SCRATCH`, `BACKUPS` (a new dated folder in the owner's backup directory), `UVX`, `LLAMA_SERVER`;
  - router: `ROUTER_PLIST`, `ROUTER_LABEL`, `ROUTER_INI`, `ROUTER_KEY`, `MODELS_DIR`, `RERANK_GGUF`, `CLASSIFY_GGUF`;
  - constraints: `CONSTRAINTS_1X` (the existing 1.1.1 file) and `CONSTRAINTS_V2` (new, alongside it);
  - minirag: `MINIRAG_CONFIG` (the `indexes.toml` path) and `TOKEN_FILE`;
  - indexes: `IDX_RU` (the Russian working corpus, three roots) and `IDX_B`, `IDX_C` (one root each), with the names chosen by the owner.
- **Two helpers**, defined in the same env file:
  - `K=$(head -1 "$ROUTER_KEY" | tr -d '[:space:]')`;
  - `mr() { HF_HUB_OFFLINE=1 "$UVX" --constraints "$CONSTRAINTS_V2" --from 'minirag-mcp[ocr,classify]==2.0.0a1' minirag-mcp "$@"; }`.
- **Window rule.** From Task 7, Step 5 until Task 13 starts, no 1.x client runs a sync. The owner agrees to this before Step 5.
- **Long commands** (index builds) run detached with output to a log in `$SCRATCH`.

### Task 7: Inventory and backups

**Files (outside the repository):** the private handover gets a new section "v2 migration inventory", and the env file is written next to it.

**Interfaces:**
- Produces: every placeholder above, a 1.x baseline (rows and sources) for each index, `$BACKUPS/.marker`, and `$SCRATCH/indexcheck.py`.

- [ ] **Step 1: Discover every place that runs minirag**

```bash
python3 - <<'EOF'
import json, pathlib
d = json.loads((pathlib.Path.home() / ".claude.json").read_text())
for k, v in d.get("mcpServers", {}).items():
    if "minirag" in json.dumps(v): print("claude user:", k)
for p, pv in d.get("projects", {}).items():
    for k, v in (pv.get("mcpServers") or {}).items():
        if "minirag" in json.dumps(v): print("claude local:", p, k)
EOF
find ~ -maxdepth 4 -name .mcp.json -not -path '*/node_modules/*' 2>/dev/null | xargs grep -l minirag
grep -l minirag "$HOME/Library/Application Support/Claude/claude_desktop_config.json" ~/.codex/config.toml ~/.cursor/mcp.json 2>/dev/null
grep -rlI --exclude-dir=.git -e 'minirag-mcp' -e '\["minirag"\]' <each folder that holds the owner's scripts> 2>/dev/null
find ~ -maxdepth 4 -type d -name .minirag -not -path "<repository checkout>/*" 2>/dev/null
uvx --from 'minirag-mcp==2.0.0a1' python -c "from minirag_mcp.indexes import default_config_path; print(default_config_path())"
```

Expected: three `.minirag` directories, one per corpus. The output also lists every client entry and every script or constraints file that names minirag; this includes at least one helper script that builds its command from a client entry. The last line is the `indexes.toml` path, which sets `MINIRAG_CONFIG`. `TOKEN_FILE` is `token` in the same directory.

- [ ] **Step 2: Write the inventory into the private handover and resolve the placeholders**

For each index, record:
- its name (chosen with the owner: lowercase letters, digits and `-`);
- its roots;
- its 1.x DB path;
- its 1.x model and FTS language, read from the client entry that serves it;
- whether its first root holds `classify.toml`.

For each client entry, record its file, scope, server name, and the index it will point at. Also record the helper scripts, the router files, the GGUF paths, `UVX`, `LLAMA_SERVER` and `BACKUPS`. Write the env file.

Run: `source <env file> && for v in REPO SCRATCH BACKUPS UVX LLAMA_SERVER ROUTER_PLIST ROUTER_INI ROUTER_KEY MODELS_DIR RERANK_GGUF CLASSIFY_GGUF CONSTRAINTS_1X MINIRAG_CONFIG TOKEN_FILE; do [ -n "${(P)v}" ] || echo "unset $v"; done; ls "$RERANK_GGUF" "$CLASSIFY_GGUF" "$CONSTRAINTS_1X"`
Expected: no `unset` lines, and the three files are listed. `MINIRAG_CONFIG` and `TOKEN_FILE` may not exist yet.

- [ ] **Step 3: Check that the classify pins agree with the GGUF the router will serve**

Run: `grep -h -A4 '^\[model\]' <first root of each index that has one>/.minirag/classify.toml | grep -E '^(repo|file|revision)'` and `shasum -a 256 "$CLASSIFY_GGUF"`
Expected: identical `repo`, `file` and `revision` lines for every `classify.toml`, and `CLASSIFY_GGUF` is that file from the hub cache's snapshot for that revision. Record its sha256 in the handover. One router preset can serve only one classify model.

- [ ] **Step 4: Write the read-only index checker**

`$SCRATCH/indexcheck.py <db>` opens the LanceDB at `<db>` without writing and prints JSON. For every table it gives `rows` (`count_rows()`) and, per FTS index, `index_stats(name).num_unindexed_rows`. For `chunks` it also gives the number of distinct `source` values, from `to_lance().to_table(columns=["source"])`.

Run: `uv run --project "$REPO" python "$SCRATCH/indexcheck.py" <1.x db of each index>`
Expected: JSON for `chunks` and `signatures` with non-zero rows. Record each index's rows and sources in the handover as its 1.x baseline.

- [ ] **Step 5: APFS-clone each 1.x index under its own sync lock**

```bash
mkdir -p "$BACKUPS"
for db in <1.x db of each index>; do
  dest="$BACKUPS/$(basename "$(dirname "$(dirname "$db")")")-lancedb"
  uv run --project "$REPO" python - "$db" "$dest" <<'EOF'
import subprocess, sys
from pathlib import Path
from minirag_mcp.lock import sync_lock
with sync_lock(Path(sys.argv[1])):
    subprocess.run(["cp", "-c", "-R", sys.argv[1], sys.argv[2]], check=True)
EOF
  diff <(cd "$db" && find . -type f | sort) <(cd "$dest" && find . -type f | sort) && echo "clone ok: $dest"
done
touch "$BACKUPS/.marker"
```

Expected: one `clone ok` per index, and no `SyncLockBusy`. If the lock is busy, a 1.x sync is still running: wait for it and repeat.

- [ ] **Step 6: Back up every file Part B will edit**

```bash
for f in "$ROUTER_INI" "$ROUTER_PLIST" ~/.claude.json ~/.zshrc <each .mcp.json and helper script from Step 1> <Desktop config, if listed>; do
  cp -p "$f" "$BACKUPS/$(basename "$f").before-v2" && echo "saved $f"
done
```

Expected: one `saved` line per file. Each later edit step repeats its own backup line, so it stays safe to run on its own.

### Task 8: Router presets for the reranker and the classify model

**Files (outside the repository):** `$ROUTER_INI` (two new sections) and `$ROUTER_PLIST` (`--models-max 4`).

**Interfaces:**
- Produces: router model names `bge-reranker-v2-m3` and `plumb-4b` (Task 9 names them in `indexes.toml`), and four resident models.

- [ ] **Step 1: Back up, then append the two sections to `$ROUTER_INI`**

Back up first: `cp -p "$ROUTER_INI" "$BACKUPS/$(basename "$ROUTER_INI").before-v2-$(date +%Y%m%d-%H%M%S)"`. Then append:

```ini
[bge-reranker-v2-m3]
model              = <RERANK_GGUF>
rerank             = true
pooling            = rank
; 4 slots x 2048: a query plus one chunk fits a slot, and a non-causal model needs its whole input in one ubatch
ctx-size           = 8192
parallel           = 4
batch-size         = 2048
ubatch-size        = 2048
kv-unified         = false

[plumb-4b]
model              = <CLASSIFY_GGUF>
; one slot of 8192: the classify prompt and N_CTX guard need >= 8192 per slot
ctx-size           = 8192
parallel           = 1
kv-unified         = false
```

- [ ] **Step 2: Back up, then set `--models-max 4` in `$ROUTER_PLIST`**

```bash
cp -p "$ROUTER_PLIST" "$BACKUPS/$(basename "$ROUTER_PLIST").before-v2-$(date +%Y%m%d-%H%M%S)"
i=$(plutil -extract ProgramArguments json -o - "$ROUTER_PLIST" | jq 'index("--models-max") + 1')
plutil -replace "ProgramArguments.$i" -string 4 "$ROUTER_PLIST"
plutil -lint "$ROUTER_PLIST" && plutil -extract ProgramArguments json -o - "$ROUTER_PLIST" | jq -r 'index("--models-max") as $i | .[$i+1]'
```

Expected: `OK`, then `4`.

- [ ] **Step 3: Validate the preset file on a throwaway router**

```bash
"$LLAMA_SERVER" --models-dir "$MODELS_DIR" --models-preset "$ROUTER_INI" --models-max 4 --host 127.0.0.1 --port 18091 --api-key-file "$ROUTER_KEY" --no-models-autoload > "$SCRATCH/router-check.log" 2>&1 & pid=$!
# once /health answers:
curl -s -H "Authorization: Bearer $K" http://127.0.0.1:18091/v1/models | jq -r '.data[].id' | grep -cE '^(bge-reranker-v2-m3|plumb-4b|multilingual-e5-large)$'
kill "$pid"
```

Expected: `3`, with no error in the log. If the router exits on `rerank`, the key's other spelling is `reranking`: fix it and repeat this step. Never reload the live router with an unvalidated file.

- [ ] **Step 4: Reload the live router**

Run: `launchctl bootout "gui/$(id -u)/$ROUTER_LABEL"; launchctl bootstrap "gui/$(id -u)" "$ROUTER_PLIST"`, then wait until `curl -sf http://127.0.0.1:8080/health` answers.

- [ ] **Step 5: Check every model the router serves**

```bash
curl -s -H "Authorization: Bearer $K" -H 'Content-Type: application/json' http://127.0.0.1:8080/v1/rerank \
  -d '{"model":"bge-reranker-v2-m3","query":"when does the contract expire","documents":["The contract is valid until 31 December 2027.","A recipe for apple pie."]}' \
  | jq -r '.results | sort_by(-.relevance_score) | .[0].index'
curl -s -H "Authorization: Bearer $K" "http://127.0.0.1:8080/props?model=plumb-4b" | jq -r '.model_path, .default_generation_settings.n_ctx'
curl -s -H "Authorization: Bearer $K" -H 'Content-Type: application/json' http://127.0.0.1:8080/v1/embeddings \
  -d '{"model":"multilingual-e5-large","input":"query: test"}' | jq '.data[0].embedding | length'
```

Then send one short chat completion to the owner's usual chat model, and list the loaded models with `curl -s -H "Authorization: Bearer $K" http://127.0.0.1:8080/v1/models | jq -r '.data[] | "\(.id) \(.status.value // .status)"'`.

Expected:
- `0`, so the matching passage ranks first;
- `$CLASSIFY_GGUF` (or the blob it resolves to) and `8192`;
- `1024`;
- the chat model answers;
- e5, the reranker, `plumb-4b` and the chat model are all loaded at the same time.

Rollback for this task: copy both backups back, then run the bootout and bootstrap from Step 4.

### Task 9: v2 constraints file and `indexes.toml`

**Files (outside the repository):** `$CONSTRAINTS_V2` (new) and `$MINIRAG_CONFIG` (new).

**Interfaces:**
- Consumes: Task 8's model names and Task 7's index inventory.
- Produces: the `mr` helper works, and three configured indexes.

- [ ] **Step 1: Generate the v2 constraints file, keeping the 1.x file untouched for rollback**

```bash
cp -p "$CONSTRAINTS_1X" "$BACKUPS/$(basename "$CONSTRAINTS_1X").before-v2"
git -C "$REPO" worktree add "$SCRATCH/v2.0.0a1" v2.0.0a1
{ echo "# Tested dependency versions for minirag-mcp 2.0.0a1 (uv export of its uv.lock, extras ocr+classify)."
  echo "# uvx without this resolves the newest releases instead; regenerate when upgrading minirag."
  uv export --directory "$SCRATCH/v2.0.0a1" --frozen --no-hashes --no-dev --no-emit-project --extra ocr --extra classify; } > "$CONSTRAINTS_V2"
git -C "$REPO" worktree remove "$SCRATCH/v2.0.0a1"
grep -E '^(lancedb|fastmcp|tesserocr|llama-cpp-python)==' "$CONSTRAINTS_V2"
mr --version
```

Expected: four pinned lines matching the tag's `uv.lock`, then `2.0.0a1`.

- [ ] **Step 2: Write `$MINIRAG_CONFIG`**

Back up first, if a file is already there: `[ ! -e "$MINIRAG_CONFIG" ] || cp -p "$MINIRAG_CONFIG" "$BACKUPS/indexes.toml.before-v2"`. Then write the file with the spec section 1 layout:
- `[server]` with `host = "127.0.0.1"`, `port = 8765` and `api_key_file` set to `$ROUTER_KEY`;
- `[embedding."intfloat/multilingual-e5-large"]` with `url = "http://127.0.0.1:8080/v1/embeddings"` and `name = "multilingual-e5-large"`;
- `[rerank]` with `url = "http://127.0.0.1:8080/v1/rerank"` and `name = "bge-reranker-v2-m3"`;
- `[classify]` with `url = "http://127.0.0.1:8080"` and `name = "plumb-4b"`;
- one `[index.<name>]` per inventory entry, giving `description` (one line from the owner), `roots` and `fts_language`. The Russian index uses `"Russian"`; the others use what the inventory recorded.

Set neither `db` nor `model`: the defaults are `<first root>/.minirag/v2` and e5 with prefixes. Never copy a 1.x `DB_PATH` into this file.

- [ ] **Step 3: Check each index's resolved configuration before any build**

Run: `for i in "$IDX_RU" "$IDX_B" "$IDX_C"; do mr --index "$i" status --json | jq '{index, dbPath, embedding, ftsLanguage, embeddingBackend, reranker, classify: .labels.backend, indexState, indexError}'; done`
Expected, per index (plan 1's `status_payload` keys, plan 3's `index`):
- `index` the configured name; `dbPath` ends in `/.minirag/v2` (this first open writes its `index.json` with an empty probe; the first write records the probe);
- `embedding.model` `intfloat/multilingual-e5-large`, `embedding.prefixes` `e5`, `embedding.backend` `endpoint`;
- `ftsLanguage` as configured;
- `embeddingBackend` `endpoint`, `reranker` `endpoint`, and `classify` `endpoint` wherever a `classify.toml` exists (its served model checked against the pin);
- `indexState` `ready` and no `indexError`.

An unknown name must fail loudly: `mr --index nope status` exits 1 with `Unknown index 'nope'. Reachable indexes: <the three names>.`

### Task 10: Build each v2 index and verify it

**Interfaces:**
- Consumes: Task 7's baseline, `indexcheck.py` and `.marker`.
- Produces: three v2 indexes in state `ready`.

- [ ] **Step 1: Build, smallest corpus first**

For each index: `mr --index <name> sync > "$SCRATCH/sync-<name>.log" 2>&1` (run detached), then `tail -5` the log.
Expected: a final summary with ingested equal to the baseline's sources (give or take files added or deleted since), and every failure named. Compare the failures with the 1.x results for the same files. A summary of `ingested: 0` is a failure, not a success.

- [ ] **Step 2: Check the identity file**

Run: `jq '{format, minirag, generation, model: .embedding.model, dim: .embedding.dim, prefixes: .embedding.prefixes, probe: (.embedding.probe | length), chunking, fts_language, document_table}' "<first root>/.minirag/v2/index.json"`
Expected:
- `format` 2, `minirag` `"2.0.0a1"`;
- the e5 model, `dim` 1024, `prefixes` `"e5"`, `probe` 1024;
- `chunking.token_budget` equal to plan 1's measured default;
- the configured `fts_language`;
- `document_table` true.

- [ ] **Step 3: Check the tables and FTS coverage**

Run: `uv run --project "$REPO" python "$SCRATCH/indexcheck.py" "<first root>/.minirag/v2"`
Expected: `documents.rows` equals `chunks.sources`, and both are close to the 1.x baseline. Every `num_unindexed_rows` on `chunks` and on `documents` is 0. This closes the open half of minirag-mcp-aca for the owner's indexes.

- [ ] **Step 4: Check state, reranking and the probe**

Run, for each index: `mr --index <name> status --json | jq -r .indexState`, then `mr --index <name> query "<a distinctive phrase from a known document>" --json | jq '{index, reranked, embeddingBackend, top: .results[0].source}'`
Expected: `ready`, which means the probe matched through the live endpoint (plan 1 checks it at open), then the index name, `reranked: true`, `embeddingBackend: "endpoint"` and that known document on top.

- [ ] **Step 5: Classify parity through the endpoint, for each index that has `classify.toml`**

Run `mr --index <name> classify try <file>` on three documents. Run the 1.x CLI with the exact command and env of that index's 1.x client entry, plus `classify try <same file>`.
Expected: the same labels, file by file.

- [ ] **Step 6: The 1.x indexes are untouched**

Run: `for db in <1.x db of each index>; do test ! -e "$db/index.json" && find "$db" -newer "$BACKUPS/.marker" -not -name .sync.lock | head -3; ls "$(dirname "$db")"; done`
Expected: no file newer than the marker, and each `.minirag` holds only `lancedb`, `v2` and (if present) `classify.toml` and pending markers. There must be no `lancedb.next`, no `lancedb.previous` and no `index.json` in `lancedb`.

### Task 11: The server under launchd

**Files (outside the repository):** `~/Library/LaunchAgents/local.minirag-mcp.plist` (new, from the template).

**Interfaces:**
- Consumes: Task 3's template and Task 9's configuration.
- Produces: `http://127.0.0.1:8765/mcp/<index>` for every index, behind the token in `$TOKEN_FILE`.

- [ ] **Step 1: Install the LaunchAgent**

```bash
P=~/Library/LaunchAgents/local.minirag-mcp.plist
[ ! -e "$P" ] || cp -p "$P" "$BACKUPS/local.minirag-mcp.plist.before-v2"
cp "$REPO/docs/launchd/local.minirag-mcp.plist" "$P"
plutil -replace ProgramArguments -json "[\"$UVX\",\"--constraints\",\"$CONSTRAINTS_V2\",\"--from\",\"minirag-mcp[ocr,classify]==2.0.0a1\",\"minirag-mcp\",\"serve\"]" "$P"
plutil -replace EnvironmentVariables -json '{"HF_HUB_OFFLINE":"1"}' "$P"
plutil -replace StandardOutPath -string "$HOME/Library/Logs/minirag-mcp.log" "$P"
plutil -replace StandardErrorPath -string "$HOME/Library/Logs/minirag-mcp.log" "$P"
plutil -lint "$P" && launchctl bootstrap "gui/$(id -u)" "$P"
launchctl print "gui/$(id -u)/local.minirag-mcp" | grep -E '^\s+state = '
stat -f '%Lp' "$TOKEN_FILE"
```

Expected: `OK`, then `state = running`, then `600`.

- [ ] **Step 2: Transport refusals**

Send an MCP `initialize` POST to `http://127.0.0.1:8765/mcp/$IDX_RU` three ways: without `Authorization`; with the token and `Origin: http://evil.example`; and with the token and `Host: evil.example:8765`. Print each status with `curl -s -o /dev/null -w '%{http_code}\n'`.
Expected: `401`, then `403`, then `421` (plan 3's answer to a non-loopback `Host`). None of them may be `200`.

- [ ] **Step 3: Tools over HTTP**

```bash
uv run --project "$REPO" python - "$IDX_RU" "<a question the corpus answers>" <<'EOF'
import asyncio, json, os, sys
from fastmcp import Client
token = open(os.environ["TOKEN_FILE"]).read().strip()
async def main():
    async with Client(f"http://127.0.0.1:8765/mcp/{sys.argv[1]}", auth=token) as c:
        print(json.dumps((await c.call_tool("list_indexes", {})).data, ensure_ascii=False))
        q = (await c.call_tool("query_documents", {"query": sys.argv[2]})).data
        print(q["index"], q.get("reranked"))
        bad = await c.call_tool("query_documents", {"query": "x", "index": "nope"}, raise_on_error=False)
        print(bad.is_error, bad.content[0].text[:200])
asyncio.run(main())
EOF
```

Expected: three indexes, all `ready`, with `$IDX_RU` as the default. Then `$IDX_RU True`. Then `True` and a message that names `nope` and the reachable indexes.

### Task 12: Measure the owner's corpus on the private ruler through the v2 server

**Files (outside the repository):** the private ruler's measurement script, which gets a server mode. Its location is in the bd memory about the Russian ruler.

- [ ] **Step 1: Back up, then add a server mode to the measurement script**

Back up first: `cp -p <ruler>/scripts/measure.py "$BACKUPS/measure.py.before-v2"`. The new mode calls `query_documents` on `http://127.0.0.1:8765/mcp/$IDX_RU` through `fastmcp.Client(url, auth=token)` with the default `top_k`. It does this for every gold item, once with the agent-style `query` field and once with the natural `question` field. It maps result sources to gold by their path under the corpus root, and writes a run in the format `score.py` reads.

- [ ] **Step 2: Run, pool and judge, then score**

Run the new mode into a new run folder. Pool unjudged documents with the ruler's `pool.py`, judge the new pairs with the ruler's judging brief, and score with `score.py`.
Expected: Task 4 Step 3's numbers for the final pipeline (the same code and budget, now over HTTP and on the rebuilt production index), each within 0.02 MRR or 1 point. For reference, the spec's figures at a 110-token budget:
- agent-style queries: MRR about 0.825 (at least 0.80), a hit in the default 8-chunk window about 93% (at least 90%), hit@10 at least 93.7%;
- natural questions: MRR about 0.75 (at least 0.72).

**Stop rule:** if any number falls below the 1.x production figures (0.757, 90.0, 0.706), Task 13 does not run. File a bug on g7q with the aggregate numbers instead.

- [ ] **Step 3: Record the aggregate numbers only, then the owner's go-ahead**

Run: `bd note minirag-mcp-g7q.13 "v2 server on the Russian ruler (<date>): agent MRR …, window …, hit@10 …; questions MRR …"`
Then ask the owner to approve the client switch.

### Task 13: Switch the owner's clients to the server

**Files (outside the repository):** the client entries and helper scripts from Task 7's inventory, and `~/.zshrc`.

- [ ] **Step 1: Back up, then export the token for `.mcp.json` expansion**

```bash
cp -p ~/.zshrc "$BACKUPS/zshrc.before-v2-$(date +%Y%m%d-%H%M%S)"
echo 'export MINIRAG_TOKEN="$(cat '"$TOKEN_FILE"' 2>/dev/null)"' >> ~/.zshrc
zsh -ic 'test -n "$MINIRAG_TOKEN" && echo token-set'
```

Expected: `token-set`.

- [ ] **Step 2: Claude Code local-scope entries, through the CLI and never by editing `~/.claude.json`**

Back up first: `cp -p ~/.claude.json "$BACKUPS/claude.json.before-v2-$(date +%Y%m%d-%H%M%S)"`. Then, for each entry, from its project directory:

```bash
claude mcp remove minirag -s local
claude mcp add --transport http -s local minirag "http://127.0.0.1:8765/mcp/<its index>" --header "Authorization: Bearer $(cat "$TOKEN_FILE")"
claude mcp get minirag
```

Expected: the transport is `http` and the status is connected.

- [ ] **Step 3: Project `.mcp.json` files (every one the inventory lists)**

Back up first: `cp -p <file> "$BACKUPS/$(basename "$(dirname <file>)")-mcp.json.before-v2"`. Replace the `minirag` entry with `{"type": "http", "url": "http://127.0.0.1:8765/mcp/<its index>", "headers": {"Authorization": "Bearer ${MINIRAG_TOKEN}"}}`.

Run, from that directory: `python3 -m json.tool .mcp.json >/dev/null && grep -cF "$(cat "$TOKEN_FILE")" .mcp.json; claude mcp get minirag`
Expected: `0`, because the literal token is not in the file, and the server is connected.

- [ ] **Step 4: Helper scripts that derived their command from a client entry**

Back up first: `cp -p <script> "$BACKUPS/$(basename <script>).before-v2"`. Make each script run `"$UVX" --constraints "$CONSTRAINTS_V2" --from 'minirag-mcp[ocr,classify]==2.0.0a1' minirag-mcp --index <its index> "$@"` with literal paths, and stop reading the client entry.
Run: `<script> status`
Expected: the status of that index, `ready`. Scripts like this, the LaunchAgent and the Desktop bridge are the only setups that still pin a version, so they are the only users of `$CONSTRAINTS_V2`.

- [ ] **Step 5: Claude Desktop, only if the inventory lists an entry or the owner asks for one**

Back up first: `cp -p "$HOME/Library/Application Support/Claude/claude_desktop_config.json" "$BACKUPS/claude_desktop_config.json.before-v2"`. Set `minirag` to `{"command": "<UVX>", "args": ["--constraints", "<CONSTRAINTS_V2>", "--from", "minirag-mcp==2.0.0a1", "minirag-mcp", "connect", "<index>"]}`, merging into the file without replacing it. Check it with `python3 -m json.tool`, then quit Desktop with Cmd+Q and reopen it.
Expected: in a new Desktop chat, "list the minirag indexes" returns the three indexes.

- [ ] **Step 6: End-to-end, in every switched project**

Run: `claude -p "Call the minirag list_indexes tool and print only this connection's default index name" --allowedTools mcp__minirag__list_indexes`
Expected: the index that entry points at.

### Task 14: Rollback on record, issues closed

**Files (outside the repository):** the private handover (new section "v2 rollback").

- [ ] **Step 1: Write the exact rollback into the handover, with concrete paths**

The handover section lists these steps in order:
1. `launchctl bootout gui/$(id -u)/local.minirag-mcp`.
2. For each local-scope Claude Code entry, from its project: `claude mcp remove minirag -s local && claude mcp add-json minirag "$(jq -c --arg p <project> '.projects[$p].mcpServers.minirag' <claude.json backup>)" -s local`. Never restore `~/.claude.json` wholesale, because that would undo everything else Claude Code has written since.
3. Copy each project `.mcp.json`, helper script and the Desktop config back from `$BACKUPS`.
4. Optionally, restore the router files from their backups and reload them with bootout and bootstrap. The new presets are harmless to keep.
5. The 1.x indexes are untouched at `.minirag/lancedb`. If one is damaged: `mv <db> <db>.damaged && cp -c -R <its clone> <db>`.
6. Run one 1.x sync per index to catch up. Then heal its FTS coverage as recorded in minirag-mcp-aca, because 1.1.1 leaves rows it newly syncs unindexed.

- [ ] **Step 2: Check that the rollback material exists, without rolling back**

Run: `ls "$BACKUPS"`, and for each local-scope entry `jq -e -c --arg p <project> '.projects[$p].mcpServers.minirag.command' <claude.json backup>`
Expected: every backup named in Step 1 is listed, and each `jq` prints the 1.x `uvx` path and exits 0.

- [ ] **Step 3: Close the issues and file the follow-ups**

```bash
bd close minirag-mcp-g7q.13 -r "Owner's three corpora rebuilt with 2.0.0a1 (e5 + prefixes, document BM25, reranker, classify via the router), measured on the private ruler, clients on the HTTP server; 1.x indexes and configs kept as rollback (private handover)."
bd close minirag-mcp-aca -r "Superseded: the fixes shipped in 2.0.0a1 and the owner's indexes were rebuilt by v2 with every FTS row indexed; no 1.x patch release (owner's call)."
bd create "Before 2.0.0 final: server.json environment variables and README describe v2 as the default" -t task -p 2 --parent minirag-mcp-g7q
bd create "Remove the owner's 1.x indexes and v2 migration backups once v2 is confirmed (owner's call)" -t chore -p 4 --parent minirag-mcp-g7q
```

If `bd close` refuses because of the dependency on g7q.10 (deferred in the spec), run `bd dep remove minirag-mcp-g7q.13 minirag-mcp-g7q.10` and close again. The epic stays open for the deferred g7q.8 and g7q.10 and the ruler-gated rest of g7q.12.

- [ ] **Step 4: Memories, then the session-close protocol**

```bash
bd remember --key v2-migration "Owner moved to minirag 2.0.0a1 on <date>: one HTTP server (LaunchAgent local.minirag-mcp) serving three indexes; router presets for the reranker and classify. Inventory, backups and exact rollback live in the private handover (see enterpriserag-bench-handover memory for its location)."
```

Append the v2 server's ruler numbers to the Russian-ruler memory with `bd remember --key <that key> "<existing text> + one line"`. Then follow `bd prime`'s session-close protocol: `git pull --rebase && git push`, and `git status` must show "up to date with origin".
