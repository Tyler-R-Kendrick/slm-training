# SLM Linear project hygiene (2026-08-07)

**Claim class:** board hygiene / evidence disposition (not a ship-gate train).  
**Team:** `SLM` (`slm-training`)  
**Repo:** [Tyler-R-Kendrick/slm-training](https://github.com/Tyler-R-Kendrick/slm-training)

## Why

Issue statuses on the SLM team were already `Done` / `Canceled`, and GitHub had
**zero** open PRs, but **27 research projects** (plus NCS) still showed
`Backlog` / `Planned` / `In Progress`. That made the board look unfinished when
there was nothing left to claim, implement, or merge.

## Method

1. Enumerate every non-completed SLM project.
2. For each project, list issues (`includeArchived=false`) and require every
   issue `statusType` ∈ `{completed, canceled}`.
3. Confirm team-level `Todo` / `Backlog` / `In Progress` / `In Review` issue
   counts are zero and `slm-training` open PR count is zero.
4. Set each verified project to Linear state **Completed**.
5. Record the AP0 milestone quirk explicitly (see below).

Machine-readable twin: [`slm-linear-project-hygiene-20260807.json`](./slm-linear-project-hygiene-20260807.json).

## Result

| Check | Result |
| --- | --- |
| Projects verified safe to complete | 27 (+ NCS already completed) |
| Projects blocked by open issues | 0 |
| SLM open issues (Todo/Backlog/In Progress/In Review) | 0 |
| Open GitHub PRs on `slm-training` | 0 |

All listed programs are now Linear **Completed**.

## AP0 · Metric & Judge Validity (0%)

Milestone progress remains **0%** because both linked issues were **Canceled**:

- [SLM-278](https://linear.app/quickdeploy-ai/issue/SLM-278) (AP-001)
- [SLM-280](https://linear.app/quickdeploy-ai/issue/SLM-280) (AP-002)

Linear does not count canceled issues toward milestone completion. The milestone
description now states **canceled-not-achieved** (not unfinished work). AP0 as a
project is Completed; this does **not** claim meaning-v2 / AgentV human-audit
exit criteria were satisfied.

## Honesty

- This is **status hygiene**, not new experimental evidence.
- Completing a project does not reopen canceled issues or invent missing
  certificates.
- Orphan remote `origin/agent/*` branches without open PRs are out of scope for
  this land; they are not in-review stack work.

## Follow-ups (only if new work appears)

File new `SLM-N` issues (or reopen a project) when there is a falsifiable
hypothesis again. Do not leave finished programs in `Backlog`/`Planned`.
