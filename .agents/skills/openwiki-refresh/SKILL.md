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
node --version                      # Node ≥ 22 expected
npm ls -g openwiki || npm install --global openwiki@0.1.2
```

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

## 3. Run the wrapper (never raw `openwiki`)

```bash
python -m scripts.update_openwiki --update --print
```

`scripts/update_openwiki.py` is the only supported entry point: it symlinks
`openwiki -> docs/openwiki` for the CLI, snapshots `AGENTS.md`, `CLAUDE.md`,
and the workflow file so the generator cannot rewrite scaffold files, restores
them afterwards, and fails loudly if `docs/openwiki/` disappears. Running the
`openwiki` binary directly bypasses those protections.

## 4. Commit + PR (what the old workflow did)

Only `docs/openwiki/` may change. If `git status` shows no diff there, report
"no OpenWiki changes" and stop. Otherwise:

```bash
git switch -c openwiki/update   # reuse/reset the branch if it already exists
git add docs/openwiki
git commit -m "docs: update OpenWiki"
git push -u origin openwiki/update --force-with-lease
gh pr create --base main --head openwiki/update \
  --title "docs: update OpenWiki" \
  --body "Automated OpenWiki documentation update (agent-run refresh; replaces the disabled scheduled workflow)."
```

Reuse an existing open `openwiki/update` PR instead of opening a duplicate.
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
