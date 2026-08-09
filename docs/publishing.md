# Publishing to the MCP catalogues

Maintainers only, and a companion to [Releasing](../README.md#releasing) — that
section covers PyPI and the registry entry; this one covers everywhere else the
server can be listed, in the order the dependencies allow.

**Nothing in this document has been done.** It is a checklist to execute, not a
record of work. Every step that reaches outside this repository is marked, and
the ones that cannot be taken back are marked twice.

State below was verified on **2026-08-09**. Re-check before acting: catalogues
change, and a stale checklist is worse than none. How to re-check is in
[Verifying, without fooling yourself](#verifying-without-fooling-yourself).

## Where we stand

| Destination | State today | What it needs |
|---|---|---|
| [Official MCP Registry](#2-the-official-mcp-registry) | absent | a release carrying the ownership marker |
| [Glama](#4-glama) | listed, unclaimed, stale | a re-crawl; optionally a claim |
| [LobeHub](#5-lobehub) | listed, stale | a re-crawl |
| [PulseMCP](#3-pulsemcp) | absent | nothing — it ingests from the registry |
| [punkpeye/awesome-mcp-servers](#6-punkpeyeawesome-mcp-servers) | absent | a pull request |
| [TensorBlock/awesome-mcp-servers](#7-tensorblockawesome-mcp-servers) | absent | a pull request, or an issue form |
| [mcpservers.org](#8-mcpserversorg) | absent | a web form |
| [mcp.so](#9-mcpso-optional) | absent | a web form (optional) |
| everything else | absent | see [Deliberately not doing](#deliberately-not-doing) |

We are listed in exactly two places, and neither was submitted to — both
crawled us. Both render a README snapshot from before the PyPI release, so the
install command they show the world is `uvx --from git+https://…`, which is
wrong now. Fixing that is a side effect of step 1, not a separate task.

## The order, and why

The dependencies are real, not stylistic. Taking them out of order costs a
release you cannot take back.

1. **The release comes first.** The official registry proves you own the PyPI
   package by finding `mcp-name: io.github.sfrangulov/minirag-mcp` in the
   package description — which is the README *as published to PyPI*. A PyPI
   description is immutable per version, and 0.4.0 shipped without the marker.
   So the marker cannot be added to a release that already exists; it needs a
   new one. Nothing downstream of the registry can start until that release is
   on PyPI.
2. **The registry entry comes second**, and it is automatic — pushing the tag
   starts [`publish-mcp.yml`](../.github/workflows/publish-mcp.yml).
3. **PulseMCP comes third and costs nothing.** It aggregates the official
   registry, so it needs no submission at all — only patience and a check.
4. **Glama and LobeHub are already listed**, and both are showing the
   pre-PyPI README. Refreshing them is worth doing before pointing anyone at
   them, which is why they sit ahead of the lists rather than after.
5. **punkpeye comes after Glama** — not because its `CONTRIBUTING.md` demands
   a Glama score badge (it does not; see step 6), but because nearly every
   entry there carries one, and ours will link to a page still advertising an
   install command that no longer applies. Refresh first, then link.
6. **The forms come last.** They are independent of everything above and of
   each other; they are last only because they are the least valuable.

## Verifying, without fooling yourself

An HTTP 200 does not mean a page exists. Glama serves `200` with a **zero-byte
body** to any non-browser client, for URLs that certainly do not exist:

```console
$ curl -sL -o /dev/null -w '%{http_code} %{size_download}\n' \
    https://glama.ai/mcp/servers/zzzz-does-not-exist-99999
200 0
```

Our own page answers exactly the same way. So do not use status codes as
evidence. Use one of these instead:

| Catalogue | How to check |
|---|---|
| Official registry | `curl -s 'https://registry.modelcontextprotocol.io/v0/servers?search=minirag'` — empty today: `{"servers":[],"metadata":{"count":0}}` |
| PyPI | `curl -s https://pypi.org/pypi/minirag-mcp/json` and read `.info.version` and `.info.description` |
| Glama | `curl -s 'https://glama.ai/api/mcp/v1/servers?query=minirag'` — returns the record; the site's own search box misses the hyphenated `minirag-mcp` and finds `minirag` |
| PulseMCP | `curl -s 'https://api.pulsemcp.com/v0beta/servers?search=sfrangulov'` — but read the caveat in [step 3](#3-pulsemcp) first |
| LobeHub, awesome lists | `curl -sL <url>` and grep the body; both serve real bytes |

Two traps worth knowing. Searching PulseMCP for `minirag-mcp` already returns a
hit — `witwicki/minirag-mcp`, an unrelated project with a colliding name. Do
not read it as us; search `sfrangulov` instead. And Glama's API is
rate-limited: a request that returned JSON a minute ago can come back empty,
which looks identical to "not listed". Retry before concluding anything.

---

## 1. Cut the release

**Public. Irreversible.** A PyPI version number can never be reused, and
neither can a registry version.

Everything this depends on is already on `registry-prep`: the `mcp-name:`
marker in [`README.md`](../README.md), [`server.json`](../server.json), the
badge row, and [`publish-mcp.yml`](../.github/workflows/publish-mcp.yml).
None of it is on `main` yet.

1. Land `registry-prep` on `main`. **Public** — the branch has never been
   pushed, and `publish-mcp.yml` must exist on the default branch before any
   tag is cut, for the same reason [Releasing](../README.md#releasing) gives
   about `release.yml`.
2. Follow [Releasing](../README.md#releasing) unchanged: `bump-my-version`,
   push the commit and the tag, publish the GitHub release. A patch bump is
   right — nothing about the server's behaviour changed.

Then confirm the marker actually landed, because this is the one thing no
re-run can fix:

```bash
curl -s https://pypi.org/pypi/minirag-mcp/json \
  | jq '.info.description | contains("<!-- mcp-name: io.github.sfrangulov/minirag-mcp -->")'
```

`true` means every step below is unblocked. `false` means the next release has
to carry it instead.

*About 20 minutes, most of it waiting for CI.*

## 2. The official MCP Registry

**Public. Irreversible.** Automatic — you do not run anything.

Pushing the tag in step 1 starts
[`publish-mcp.yml`](../.github/workflows/publish-mcp.yml), which waits up to 30
minutes for the release to appear on PyPI, checks the ownership marker in the
description PyPI is serving, and publishes [`server.json`](../server.json) over
GitHub OIDC. [The MCP Registry entry](../README.md#the-mcp-registry-entry)
explains the sequencing.

Confirm with the API — an empty `servers` array means it did not land:

```bash
curl -s 'https://registry.modelcontextprotocol.io/v0/servers?search=minirag'
```

**Fallback only**, if the workflow is broken and you need to publish by hand.
Install `mcp-publisher` per the
[quickstart](https://github.com/modelcontextprotocol/registry/blob/main/docs/modelcontextprotocol-io/quickstart.mdx)
(`brew install mcp-publisher`, or the tarball from
[releases/latest](https://github.com/modelcontextprotocol/registry/releases/latest)),
then from the repository root:

```bash
mcp-publisher login github && mcp-publisher publish
```

`login github` is the interactive device flow, for a laptop. The workflow uses
`login github-oidc`, which only works inside Actions. Publishing by hand skips
both of the workflow's guards — the tag/`server.json` version check and the
PyPI marker check — so read them in the workflow file before you lean on this.

*Zero minutes of work; up to 30 of waiting.*

## 3. PulseMCP

**Nothing to submit.** PulseMCP ingests the official registry, so step 2 is the
whole action. Give it a few days, then check.

Checking is the awkward part. `www.pulsemcp.com` answers `403` to `curl`, so
the site itself needs a real browser. The open API still works:

```bash
curl -s 'https://api.pulsemcp.com/v0beta/servers?search=sfrangulov'
```

but it is on a published sunset ramp — it fails a rising share of requests on
purpose (10% since April 2026, 50% since June, 100% in September 2026) and
answers with an `API_SUNSET` error telling you so. A single empty response
therefore proves nothing; re-run it, and check the response is not an error
before believing a `total_count` of 0. Its replacement, `/v0.1/servers`,
requires an `X-API-Key` we do not have, so once the ramp closes, a browser is
the only free channel left.

If we are still missing a week after the registry entry is live, that is worth
an email — not a resubmission.

*Two minutes, later.*

## 4. Glama

Already listed, at
[glama.ai/mcp/servers/sfrangulov/minirag-mcp](https://glama.ai/mcp/servers/sfrangulov/minirag-mcp).
The id-based form <https://glama.ai/mcp/servers/btvcl5o1wx> redirects there.
The record was created by a crawl; nobody submitted it.

What is wrong with it: it renders the README from commit `8d2e47d`, which
predates the PyPI release. Every install command on the page is the
`uvx --from git+https://github.com/sfrangulov/minirag-mcp minirag-mcp` form,
including the Claude Code and Codex snippets. The
[score page](https://glama.ai/mcp/servers/sfrangulov/minirag-mcp/score) reports
`Latest release: v0.1.0`, and the API's `tools` array is empty although the
server exposes 11.

After step 1, the fix is a re-crawl. Glama re-indexes from the repository, so
the first thing to try is simply waiting a few days after `main` moves, then
re-reading the page in a browser (not `curl` — see
[above](#verifying-without-fooling-yourself)). If it stays stale, the
[Admin tab](https://glama.ai/mcp/servers/sfrangulov/minirag-mcp/admin) is where
a maintainer would ask for one.

Two things would raise the score, neither of them done here:

- **Claiming the listing.** The server page carries a claim control, and the
  score page grades an `Author verified` criterion we currently fail. Claiming
  means signing in to Glama as the GitHub account that owns the repository —
  an account action, so it is yours to take, not something a checklist should
  do on your behalf.
- **Adding `glama.json`.** The score page reports `No glama.json` and links to
  <https://glama.ai/blog/2025-07-08-what-is-glamajson>. It is a small metadata
  file in the repository root. Worth reading before deciding; not required for
  anything else in this document.

One detail for step 6: the badge row in [`README.md`](../README.md) points its
link at the id URL `btvcl5o1wx` while the image comes from the namespace URL.
Both resolve. Entries in the awesome lists conventionally use the namespace
form for both, so use that when you paste.

*Ten minutes, plus days of waiting for the crawler.*

## 5. LobeHub

Already listed, at
[lobehub.com/mcp/sfrangulov-minirag-mcp](https://lobehub.com/mcp/sfrangulov-minirag-mcp),
also without anyone submitting it, and also from the pre-PyPI snapshot — the
metadata block it serves still reads `Version 0.1.0`.

There is nothing to do but wait for the re-crawl after step 1, and check.
Fetch that page directly and read the `Version` row; do not try to judge
presence from LobeHub's catalogue search, which does not answer usefully to a
plain fetch.

No claim or verification control was visible on the page. If the listing never
refreshes, the "Request a Server" link their catalogue offers is the only
channel on offer.

*Five minutes, later.*

## 6. punkpeye/awesome-mcp-servers

**Public.** A pull request into somebody else's repository — 92k stars, so
assume it is read.

[`CONTRIBUTING.md`](https://github.com/punkpeye/awesome-mcp-servers/blob/main/CONTRIBUTING.md)
asks for: an edit to `README.md`, the entry categorised and in **alphabetical
order** within its category, one server per line, and consistency with the
surrounding format. It does **not** require a Glama score badge — that is a
convention almost every entry follows, not a documented rule.

Category: `### 🔎 end to end RAG platforms`. Alphabetical by owner puts us
between `poll-the-people/customgpt-mcp` and `vectara/vectara-mcp`.

The legend is defined at the top of that README. `🐍` Python, `🏠` local
service, `🍎` macOS, `🐧` Linux. Not `🪟` — [`ci.yml`](../.github/workflows/ci.yml)
tests `ubuntu-latest` and `macos-latest` only, so a Windows claim is one we
cannot back.

Paste this as a single line:

```markdown
- [sfrangulov/minirag-mcp](https://github.com/sfrangulov/minirag-mcp) [![sfrangulov/minirag-mcp MCP server](https://glama.ai/mcp/servers/sfrangulov/minirag-mcp/badges/score.svg)](https://glama.ai/mcp/servers/sfrangulov/minirag-mcp) 🐍 🏠 🍎 🐧 - Hybrid search over a folder of your own documents: vector similarity fused with BM25 by weighted RRF, so exact identifiers and error codes surface next to semantically similar passages. Filenames are searchable and become titles when a document's own heading is boilerplate. Multilingual out of the box; nothing leaves the machine except the one-time embedding-model download. `uvx minirag-mcp`
```

One thing to know before opening it: that `CONTRIBUTING.md` asks automated
agents to append `🤖🤖🤖` to the PR title to opt into a fast-track merge. A
human opening the PR should not use it.

*Fifteen minutes, plus however long review takes.*

## 7. TensorBlock/awesome-mcp-servers

**Public.** Correcting an earlier note: this is not PR-only. There are two
paths, and the repository says the pull request is the faster one.

**Path A — pull request** into
[`docs/search.md`](https://github.com/TensorBlock/awesome-mcp-servers/blob/main/docs/search.md).
The format there is plainer than punkpeye's: no badges, no legend emoji, just
`- [owner/repo](url): description.` New entries sit near the top of the list.

```markdown
- [sfrangulov/minirag-mcp](https://github.com/sfrangulov/minirag-mcp): Local-first RAG over a folder of your own documents — hybrid search fusing vector similarity with BM25 by weighted RRF, so exact identifiers and error codes surface next to semantically similar passages. Filenames are searchable and become titles when a heading is boilerplate. 11 tools over stdio, multilingual by default, no API key and no network at query time. `uvx minirag-mcp`, document root via `BASE_DIR`.
```

**Path B — issue form**, if you would rather they place it:
<https://github.com/TensorBlock/awesome-mcp-servers/issues/new?template=add-mcp-server.yml>.
Its required fields are Server name (`sfrangulov/minirag-mcp`), Project URL
(`https://github.com/sfrangulov/minirag-mcp`), Best category (pick **Search**
from the dropdown) and a description. Optional but easy: Install
(`uvx minirag-mcp`; env `BASE_DIR` or `BASE_DIRS`) and Transport (**stdio**).

They also offer a profile-claim form once the entry exists. Same reasoning as
Glama — an account action, left to you.

*Ten minutes either way.*

## 8. mcpservers.org

**Public.** A web form: <https://mcpservers.org/submit>.

This one submission covers two catalogues. `wong2/awesome-mcp-servers` is the
same directory's repository, and its README opens with "We do not accept PRs.
Please submit your MCP on the website" pointing at this form — so there is no
separate action for it, and a PR there would be rejected.

Fields, all required:

| Field | Paste |
|---|---|
| Server Name | `minirag-mcp` |
| Short Description | `Local-first RAG MCP server: hybrid search — vector similarity fused with BM25 — over a folder of your own documents, with nothing leaving the machine.` |
| Link (GitHub or docs) | `https://github.com/sfrangulov/minirag-mcp` |
| Category | **Search** (the dropdown also offers Development, File System and Memory; Search is the closest) |
| Contact Email | yours |

The page offers a **$39 "Premium Submit"** for faster review, a badge and a
dofollow link. Listing is free; skip it.

*Five minutes.*

## 9. mcp.so (optional)

**Public.** A web form: <https://mcp.so/submit>. Only two fields — Repository
URL (`https://github.com/sfrangulov/minirag-mcp`) and Name (`minirag-mcp`;
2–120 characters).

Judged marginal: high volume, little curation, no obvious traffic. It costs two
minutes, so do it if you are already in the browser, and skip it without regret
otherwise.

*Two minutes.*

---

## Deliberately not doing

Recorded so nobody has to rediscover the reasons.

**`modelcontextprotocol/servers` — closed.** Its
[`CONTRIBUTING.md`](https://github.com/modelcontextprotocol/servers/blob/main/CONTRIBUTING.md)
says under "We don't accept:" — "**New server implementations** — We encourage
you to publish them to the MCP Server Registry instead." That is step 2. Do not
open a PR there.

**Anthropic's Connectors Directory — ineligible.** Connectors are remote MCP
servers; Claude reaches them from Anthropic's cloud, not from your machine.
[About custom connectors](https://support.claude.com/en/articles/11175166-about-custom-connectors-remote-mcp-servers)
is explicit that locally-configured servers are a separate mechanism. This
server is stdio and local by design, and putting a document index behind a
hosted endpoint would undo the one property it is built around. Not a gap to
close.

**Claude Code plugin marketplaces — a repackage, not a submission.** There is
no central directory to submit to: a marketplace is a git repository holding
`.claude-plugin/marketplace.json`, and users add it by URL with
`/plugin marketplace add`. Shipping as a plugin means adding a plugin manifest
and either hosting a marketplace or getting into someone else's. See
<https://code.claude.com/docs/en/plugin-marketplaces>. Reasonable later; it is
its own piece of work, not a checklist item.

**Docker MCP Registry — needs a Dockerfile.** Its
[`CONTRIBUTING.md`](https://github.com/docker/mcp-registry/blob/main/CONTRIBUTING.md)
requires local servers to be containerised, with a Dockerfile in the source
repository, and the PR adds a `server.yaml` under `servers/`. We ship no
Dockerfile, and a container is an awkward fit for a server whose whole job is
reading a folder on your disk and caching a ~220 MB model. Skipped on cost, not
on principle.

**Smithery — needs an `.mcpb` bundle.** Smithery takes remote servers by URL;
a local stdio server has to be published as a prebuilt `.mcpb` bundle
(<https://smithery.ai/docs/build/publish>). That is a second distribution
artifact to build and keep in step with every release, for a catalogue we have
no evidence sends traffic to local Python servers. Revisit if `.mcpb` becomes
something we want for other reasons.

**`appcypher/awesome-mcp-servers` — archived.** The repository is archived on
GitHub, so it accepts nothing. Nothing to do, ever.

**`wong2/awesome-mcp-servers` — no PRs.** Covered by
[step 8](#8-mcpserversorg); see the note there.

## What is not wrong

Worth stating, because it is the kind of thing that gets re-investigated: there
are **no duplicate or incorrect entries** for this project anywhere that was
checked. The two listings that exist are both ours, both point at the right
repository, and both are merely out of date. There is no cleanup to do — only
the refresh in steps 4 and 5.
