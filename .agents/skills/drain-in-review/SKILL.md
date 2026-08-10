---
name: drain-in-review
description: Use when the Linear In Review queue for team SLM is backed up and an agent session should claim the oldest issues, finish each in an isolated git worktree (reusing open PRs), fix CI in bounded rounds, and squash-merge once merge preflight passes
---

# Drain In Review

## Overview

Harness-neutral port of the Grok workflow
[`.grok/workflows/unblock-in-review.rhai`](../../../.grok/workflows/unblock-in-review.rhai)
(that file stays the Grok-native implementation — do not edit it from here).
Any agent harness can follow this skill; phases and defaults match the rhai
workflow: **Discover → Claim → Work → Heal → Report**.

Linear issues `SLM-N` map 1:1 to `docs/design/iter-*.md` — read the matching
iteration doc before implementing.

## Hard rules (non-negotiable)

1. **Merge preflight.** Before **any** merge, `python -m scripts.verify_merge_ready`
   must pass — run it and require exit 0. **When that script does not exist
   yet**, run the static-check list from `.github/workflows/ci.yml` job
   `python-static` instead, all green:
   `python -m scripts.repo_policy` ·
   `python -m scripts.verify_decode_invariants` ·
   `python -m scripts.verify_agent_surfaces` ·
   `python -m scripts.verify_ownership_map` ·
   `python -m scripts.extract_test_cases` then
   `python -m scripts.refresh_test_cases --check` ·
   `ruff check .` ·
   `python -m compileall -q src scripts tests` ·
   `python -m scripts.verify_checkpoint_references --check` ·
   `python -m scripts.verify_version_stamps --check`.
   Never bypass, stub, or "temporarily skip" any of these.
2. **Never weaken gates.** No lowering ship bars, deleting tests, loosening
   preflight, or editing gate configs to go green. Document a fail and stop the
   worker instead (`honest-ship-eval`).
3. **Bounded rounds.** CI-fix loops per issue are capped: `max_check_rounds`
   default **3**, hard max **6** — and this cap covers the Work phase's CI-fix
   loop **and** the single Heal re-run together, not each independently.
   Reserve the last round for Heal: the Work phase's own loop stops at
   `max_check_rounds - 1` rounds so Heal's one re-run is the round that
   fills the cap, keeping total attempts per issue at exactly
   `max_check_rounds`, never `max_check_rounds + 1`. When the cap is hit,
   report the issue as `blocked` with the failing check and next action —
   do not loop forever.
4. **Worktree isolation.** One isolated git worktree per issue under a sibling
   parent dir (default `../.worktrees-unblock/<id-lower>`). Never work on two
   issues in one checkout, never work in the primary clone.

## Config defaults (mirror the rhai args)

| Setting | Default |
| --- | --- |
| Linear team | `SLM` (rhai arg string: `slm-training` — use whichever your Linear workspace resolves) |
| State | `In Review`, oldest → newest |
| `max_issues` | 3 (hard max 8) |
| `base_branch` | `main` |
| Claim label | `claimed-by-<agent>` (rhai: `claimed-by-grok`) |
| Done state | `Done` |
| `allow_merge` | true (squash) |
| `max_check_rounds` | 3 (max 6) |
| Prefer existing PR | true |

## Linear MCP availability caveat

Linear MCP is configured **for Claude Code only** (`.mcp.json`). In other
harnesses, or in a session where the Linear tools are absent, do the
Discover/Claim/Done steps with whatever Linear access exists (CLI/API) or —
if there is none — run in **PR-only mode**: skip claims/state changes, still
babysit and merge PRs, and report which Linear mutations were skipped. Never
fabricate Linear state.

Without the Linear claim label, nothing else serializes two concurrent drain
sessions onto the same PR. Substitute a PR-level compare-and-swap before the
Work phase: check for an existing `drain-in-review: claimed by <agent>`
comment on the PR (`gh pr view <n> --json comments`); if one already exists
from a different, still-active agent, skip that PR this round instead of
babysitting it. Otherwise post that comment as this session's own claim
before starting Work.

## Phases

1. **Discover.** List team `In Review` issues oldest → newest (skip archived /
   canceled / already-claimed unless told otherwise). List open PRs
   (`gh pr list --state open --json number,title,headRefName,url`) and match by
   issue id in title or head branch (highest PR number wins). Mode per issue:
   `babysit` when an open PR exists, else `implement`. Cap at `max_issues`.
2. **Claim (idempotent).** Ensure the claim label exists; merge it into the
   issue's labels (read existing labels first — label saves replace the full
   set); comment once that this session claimed it. Create the worktree:
   - greenfield: `git worktree add -b <agent>/unblock-<id-lower> <path> origin/<base>`
   - babysit: `git worktree add -B <head> <path> origin/<head>` — **never**
     `git worktree add <path> origin/<head>` alone (detached HEAD).
   Reuse a valid existing worktree; recreate a broken one.
3. **Work (one worker per issue, parallel when the harness supports it).**
   - Read the issue + `docs/design/iter-*.md`; inspect git/PR state before any
     side effect (idempotency).
   - If the PR's base ≠ `base_branch`, retarget it
     (`gh api -X PATCH repos/<owner>/<repo>/pulls/<n> -f base=<base>`); if the
     PR is closed unmerged, open a new PR from the same head.
   - Rebase onto `origin/<base>`, resolve conflicts, `push --force-with-lease`.
   - On CI failure: `gh run view --log-failed`, fix root cause, push, re-watch
     — within the round cap. Required checks: `python`, `python-static`,
     `data-build`; Vercel / CodeRabbit alone must not block a merge. Bound
     every network/diagnostic command (`gh run view`, `gh pr view`, log
     fetches) with a wall-time limit and a log-size cap — a hung network call
     or an oversized CI log must not exhaust the round budget or the worker.
   - **Run the merge preflight (hard rule 1).** A push after preflight but
     before merge can change the PR head, so pin the exact commit it
     validated: capture `headRefOid` (`gh pr view <n> --json headRefOid`)
     immediately before or right after preflight succeeds, then merge with
     `gh pr merge <n> --squash --delete-branch --match-head-commit <sha>`.
     If the merge rejects the SHA (the head moved), fetch the new
     `headRefOid`, reset the isolated worktree to that exact commit (verify
     `git rev-parse HEAD` matches it), and re-run preflight against that
     refreshed head before merging — never merge a commit preflight never saw,
     and never rerun preflight in a worktree that's still on the stale head.
     Confirm MERGED via `gh pr view`.
   - Move the Linear issue to `Done` with a comment linking the PR; remove the
     claim label.
   - Status per worker: `ok` (merged) | `pr_ready` (green, `allow_merge=false`)
     | `blocked` | `failed`, with error + next action.
   - **Every worker's worktree is removed in every terminal state** — `ok`,
     `pr_ready`, `blocked`, and `failed` alike (`git worktree remove --force`).
     `blocked`/`failed` are terminal statuses too; leaving their worktrees
     behind "for diagnosis" is how repeated drains exhaust local disk with
     orphaned worktrees. If a worktree is deliberately kept for diagnosis,
     record that decision explicitly in the report — don't just skip cleanup.
4. **Heal (once).** Re-run each `failed`/`blocked` worker exactly one more
   time with the prior error as context — this is the round reserved by hard
   rule 3, not an extra attempt beyond `max_check_rounds`. One heal pass
   total — no heal loops. Whatever the worker's status after Heal, its
   worktree is removed per the cleanup rule above.
5. **Report.** Scoreboard to chat: merged / pr_ready / blocked / failed per
   issue, PR URLs, leftover queue depth. Best-effort durable copy under
   `outputs/unblock-in-review/` (gitignored; never commit it).

## Red flags

- Merging without a passing `verify_merge_ready` (or its documented fallback list)
- Merging a commit `verify_merge_ready` never validated (missing `--match-head-commit`)
- Editing a gate, test, or preflight script to make a check pass
- Unbounded fix loops past `max_check_rounds`, or Heal running as an extra
  attempt beyond the cap rather than the cap's reserved last round
- Two issues sharing a worktree, or edits in the primary clone
- An orphaned worktree left behind for a `blocked` or `failed` worker
- Claiming Linear state changes that were never made (MCP absent)
- `--admin` merges past **required** checks (only ever past non-required ones)
