---
name: openwiki-refresh
description: Use when refreshing the generated OpenWiki pages under docs/openwiki — a scheduled or manual session installs the pinned OpenWiki CLI, selects a provider key, runs scripts.update_openwiki, and opens a PR on branch openwiki/update (skips cleanly when no provider key exists)
---

# OpenWiki refresh

## Overview

`docs/openwiki/` is generated documentation. The old scheduled GitHub Actions
job was disabled to save hosted minutes
([`docs/design/ci-minutes-and-speed-plan-20260806.md`](../../../docs/design/ci-minutes-and-speed-plan-20260806.md));
`.github/workflows/openwiki-update.yml` is `workflow_dispatch`-only, so the
refresh is **manual today** and this skill is the agent-run replacement. It does
exactly what the workflow did: install the pinned CLI, pick a provider, run the
wrapper script, and open a PR on `openwiki/update`.

**Repo law:** generated OpenWiki pages are **never hand-edited**. Fix source
code/docs and regenerate. Do not patch `docs/openwiki/*.md` by hand (the only
exception is an explicit user request).

## 1. Check / install the CLI (pinned)

```bash
node --version                                     # Node ≥ 22 expected
npm ls -g --depth=0 openwiki@0.1.2 || npm install --global openwiki@0.1.2
```

Check the **pinned version explicitly** (`openwiki@0.1.2`, not a bare `openwiki`
presence check) — an unversioned check reports whatever global `openwiki` tree
exists, even a different version, and skips the install that would fix it.
Always pin `openwiki@0.1.2` — the wrapper and the disabled workflow assume this
version. Do not float to `latest`.

## 2. Select a provider (or SKIP — never fail)

Mirror the workflow's selection order exactly:

| Key present | Export | Model |
| --- | --- | --- |
| `OPENAI_API_KEY` | `OPENWIKI_PROVIDER=openai` | `OPENWIKI_MODEL_ID=gpt-5.6-terra` |
| else `OPENROUTER_API_KEY` | `OPENWIKI_PROVIDER=openrouter` | `OPENWIKI_MODEL_ID=z-ai/glm-5.2` |
| neither | **SKIP the run** | — |

When **neither** key is in the environment, the run is **skipped, not failed**:
print a clear message such as
`OpenWiki refresh SKIPPED: no OPENAI_API_KEY or OPENROUTER_API_KEY in this
session — nothing was generated or committed` and stop with success. A
scheduled session without keys must never register as a failure or open an
empty PR.

Optional: if `LANGSMITH_API_KEY` is present, set `LANGCHAIN_TRACING_V2=true`
and `LANGCHAIN_PROJECT=openwiki`; otherwise `LANGCHAIN_TRACING_V2=false`.

Never commit, echo into logs, or persist any of these keys — they live only in
the session environment.

## 3. Prepare the branch, then run the wrapper (never raw `openwiki`)

Prepare `openwiki/update` **before** generating anything — resetting it
*after* the wrapper runs would discard the wrapper's own uncommitted output
before `git add` ever sees it:

```bash
# git switch -c fails if the branch already exists locally; reset it instead.
git switch openwiki/update 2>/dev/null || git switch -c openwiki/update
git reset --hard main            # start the branch clean from main each run
python -m scripts.update_openwiki --update --print
```

`scripts/update_openwiki.py` is the only supported entry point: it symlinks
`openwiki -> docs/openwiki` for the CLI, snapshots `AGENTS.md`, `CLAUDE.md`,
and the workflow file so the generator cannot rewrite scaffold files, restores
them afterwards, and fails loudly if `docs/openwiki/` disappears. Running the
`openwiki` binary directly bypasses those protections.

## 4. Commit + PR (what the old workflow did)

Only `docs/openwiki/` may change. If `git status` shows no diff there, report
"no OpenWiki changes" and stop. Otherwise, make the PR step idempotent — a
rerun must never fail just because the PR already exists from a prior
refresh:

```bash
git add docs/openwiki
git commit -m "docs: update OpenWiki"
git push -u origin openwiki/update --force-with-lease

# Reuse an existing open PR instead of always creating a new one.
existing_pr=$(gh pr list --base main --head openwiki/update --state open --json number --jq '.[0].number')
if [ -n "$existing_pr" ]; then
  echo "Reusing existing PR #$existing_pr"
else
  gh pr create --base main --head openwiki/update \
    --title "docs: update OpenWiki" \
    --body "Automated OpenWiki documentation update (agent-run refresh; replaces the disabled scheduled workflow)."
fi
```

Never commit straight to `main`, and never add non-`docs/openwiki` paths to the
commit. If scaffold files (`AGENTS.md`, `CLAUDE.md`, workflow YAML) show a diff
after the run, the wrapper's restore failed — revert them and investigate; do
not include them in the PR.

## Red flags

- Hand-edited `docs/openwiki/*.md` in the diff
- Unpinned `openwiki` install (`latest` instead of `0.1.2`)
- A "failed" run whose only problem was a missing provider key (must be SKIP)
- Provider keys appearing in committed files or PR text
- Commit touching anything outside `docs/openwiki/`
