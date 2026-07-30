---
name: sdlc
description: >
  How to do multi-step engineering work in this repo: parent agents plan and
  stack PRs with official GitHub Stacked PRs (`gh stack` / `gs`); subagents
  implement layers with incremental check-ins; parent closes every PR
  bottom-up with rubber-duck adversarial review, comment resolution, CI
  repair, and squash-merge. Also covers Scalar enlistments, sparse-checkout,
  worktrees/workspaces for humans and agents. Use for multi-phase tasks,
  stacked PR workflows, PR closeout, workspace setup, scalar/sparse checkout,
  or whenever work spans more than one reviewable layer.
---

# SDLC — how work ships in this repo

**Required for multi-step / multi-phase / multi-task work.** Single-file
hotfixes may stay one PR; everything larger uses this skill.

**Activation is automatic.** Load and follow this skill whenever you are
implementing engineering work that is meant to land in the repo — not only
when the user types `sdlc`. Creating the skill, fixing delivery process, or
being told to “check in / push / land / ship” still runs the **full**
lifecycle below.

Canonical laws still win: [`AGENTS.md`](../../../AGENTS.md),
[`docs/repository-organization.md`](../../../docs/repository-organization.md),
and the other skills in this directory. This skill is the **delivery process**
layer on top of those laws.

Load references on demand:

| When | Read |
| --- | --- |
| Creating or managing stacks | [`references/stacked-prs.md`](references/stacked-prs.md) |
| Clone / workspace / sparse / worktrees | [`references/scalar-sparse-workspaces.md`](references/scalar-sparse-workspaces.md) |
| Closing the stack (review → merge) | [`references/closeout-review.md`](references/closeout-review.md) |

## Roles

| Role | Owns | Must not |
| --- | --- | --- |
| **Parent agent** | Plan layers, open/maintain the stack **or PR**, spawn subagents, integrate mid-stack fixes, **bottom-up closeout through squash-merge** | Implement every layer itself when a subagent can; stop at push; ask the user whether to open/review/merge |
| **Subagent** | One stack layer (or one vertical slice), incremental commits, local checks for that layer | Open competing stacks; rewrite lower layers without parent coordination; skip check-ins |
| **Human** | Intent, merge policy exceptions, billing budget calls | Required as a rubber stamp for every commit, PR open, or CI fix — agents open PRs, fix CI, and closeout themselves |

## Non-negotiable process rules

1. **Subagents by default.** Multi-file, multi-phase, or multi-concern work is
   decomposed. Parent plans; subagents execute. Parent integrates.
2. **Incremental check-ins.** Subagents commit as soon as a coherent unit is
   green locally — not one giant end-of-task commit. Prefer small, reviewable
   commits on the layer branch.
3. **Stacked PRs for multi-layer work.** Parent uses official
   [`gh stack`](https://gh.io/stacks) (`gs` alias). One logical concern per
   layer. Dependency order: foundations at the **bottom**, dependents **above**.
   Single-concern work still opens **one** PR and still runs closeout.
4. **Parent owns the full lifecycle:** plan → implement/check-in → **open or
   update PRs** (`gh stack submit --open` or `gh pr create`) → mid-stack fixes
   with `rebase`/`push` → **closeout through squash-merge**.  
   **Push is not done.** A branch on the remote without an open PR (or with an
   open PR and no closeout) is incomplete work.
5. **Never ask permission for required delivery steps.** Do **not** say
   “want me to open a PR?”, “shall I create a PR?”, “let me know if you want
   me to merge”, or similar. User shorthand maps to the full lifecycle:

   | User says (examples) | Agent does |
   | --- | --- |
   | check in / commit / push | commit, push, **open/update PR**, start closeout |
   | land it / ship it / get this in | full lifecycle through **squash-merge** |
   | open a PR | open PR **and** run closeout (review → CI → merge) |
   | draft only / do not merge / PR only no merge | open draft or ready PR; stop before merge; still fix CI |

   Only an **explicit** human stop (“don’t merge”, “draft only”, “wait for
   me”) pauses merge. Silence after push is not a stop.
6. **Closeout is mandatory and bottom-up.** For **every PR the parent
   opened**, before the task is done:
   - Review that PR (and its diff vs its base) in **rubber-duck + adversarial**
     mode — see [`references/closeout-review.md`](references/closeout-review.md).
     Post the duck notes on the PR (comment), not only in chat.
   - Address **all** PR comments and review feedback.
   - Fix **all** relevant status checks (CI, required checks). The **only**
     allowed skip is an explicit **billing / budget exceeded** failure; document
     it in the PR and stop merging that layer until budget is restored or a
     human waives.
   - **Squash-merge** that PR (or land the approved prefix of the stack with
     `gh stack merge --yes --squash` when the whole stack is ready).
   - Then move **up** the stack and repeat until every opened PR is merged or
     intentionally closed with a written reason.
7. **Repo laws still apply on every layer.** Ship gates, docs-after-experiment,
   model cards, version stamps, `git mv` / `organize-repository`, decode
   invariants — none are waived by stacking.

## Decision: one PR vs stack

| Shape of work | Delivery |
| --- | --- |
| One concern, ≤ ~reviewable size, single owner | One branch + one PR is fine |
| 2+ sequential concerns, phases, or independent review units | **Stack** — one layer per concern |
| Parallel independent efforts | Separate stacks (or separate PRs), not one tangled stack |
| Experiment / matrix run | Follow `running-experiment-matrices` / `documenting-experiment-results`; stack the *code* changes, not the raw `outputs/` |

## Parent agent playbook

```text
1. Clarify goal + acceptance (what "done" means = merged unless human said stop).
2. Plan layers bottom→top (story a reviewer can read in order).
3. Prepare workspace (worktree preferred; Scalar/sparse as needed).
4. gh stack init <bottom-layer>   # or one feature branch for single-PR work
5. For each layer:
   a. Spawn a subagent with: layer goal, base branch, allowed paths,
      check-in cadence, local test commands, "do not touch lower layers".
   b. Subagent implements + incremental commits on that branch.
   c. Parent spot-checks; if mid-stack fix needed, parent (or a subagent)
      commits on the correct lower layer, then gh stack rebase --upstack
      and gh stack push.
   d. gh stack add <next-layer> when starting the next concern.
6. Immediately open/update reviewable PRs — do not wait for the user:
     gh stack submit --open
     # single PR: gh pr create (or gh pr edit if already open)
7. Closeout (required, no ask): follow references/closeout-review.md bottom→top
   — rubber-duck comment on each PR, fix CI, squash-merge each opened PR.
8. gh stack sync --prune after merges; report final PR URLs + merge SHAs.
```

**After the last intended commit:** open/update the PR in the **same turn**
as the push. **After PRs exist:** start closeout in the **same session** —
do not end with “branch pushed; say if you want a PR.”

### Planning layers (reviewer story)

Typical order (adapt to the change):

```text
main
 └── 01-contract-or-types
      └── 02-implementation
           └── 03-tests-harness
                └── 04-docs-model-card   # if checkpoint/docs required
```

One concern per layer. If a layer grows large, split with `gh stack modify`
or fold with `d`/`u` — do not ship a kitchen-sink PR "because the stack exists."

### Spawning subagents

Give each subagent a **closed brief**:

- Goal for **this layer only** and out-of-scope list
- Branch name / stack position (bottom = 1)
- Paths they may edit; paths they must not
- Required local checks (`python -m scripts.repo_policy`, targeted tests via
  `.githooks/check-changed` patterns, etc.)
- Check-in rule: commit after each coherent green unit; push when asked or
  after each check-in if the layer is already submitted
- Skills that apply (e.g. `ponytail`, `organize-repository`,
  `dashboard-openui-parity`, `honest-ship-eval`)
- Instruction: **do not** open a second stack; **do not** rewrite history of
  lower layers without parent approval

Parent re-reads subagent output, runs or assigns verification, and only then
advances the stack.

### Incremental check-ins (subagent)

- Commit early and often on the **layer branch**.
- Message: why, not noise (`caveman-commit` optional).
- After push to an open PR: paste a short status (what landed, what remains,
  risks) so the parent can replan without re-deriving state.
- Never accumulate a day of uncommitted work "for a clean history" — history
  is cleaned by layer boundaries and squash-merge, not by silent WIP.

## Tooling prerequisites

```bash
# Official Stacked PRs extension (GitHub)
gh extension install github/gh-stack
gh extension upgrade github/gh-stack
gh stack alias          # installs `gs` → gh stack (~/.local/bin)

# Auth
gh auth status          # need repo + workflow scopes for CI re-runs

# Scale-oriented Git (once per machine enlistment)
scalar register /path/to/slm-training   # or: cd enlistment && scalar register
scalar list
```

Docs: [gh-stack](https://gh.io/stacks) ·
[CLI reference](https://github.github.com/gh-stack/reference/cli/) ·
`man scalar` / `git sparse-checkout`.

Prefer `gs` when the alias is on `PATH`; otherwise `gh stack`. Non-interactive
automation: `gh stack submit --auto`, `gh stack merge --yes --squash`,
`gh stack sync --prune`.

## Workspace model (summary)

Full detail: [`references/scalar-sparse-workspaces.md`](references/scalar-sparse-workspaces.md).

| Mechanism | Use for |
| --- | --- |
| **Scalar register** | Background maintenance (commit-graph, multi-pack, prefetch), large-repo Git defaults on the enlistment |
| **Scalar clone** | New machines: partial clone + sparse by default (or `--full-clone` when you need everything) |
| **Sparse-checkout (cone)** | Restrict a worktree to the paths a human/agent needs for one task |
| **Git worktrees** | Parallel agents/layers without stashing; one worktree per active stack or long-running agent |
| **Agent isolation worktrees** | Subagent isolation when the harness supports it; still one stack owned by the parent |

Do **not** enable sparse-checkout on a shared multi-agent worktree without
coordinating — other agents may need paths you cone out. Prefer a **dedicated
worktree** per task with its own sparse cone.

## Closeout gate (parent cannot skip)

Before reporting the task complete:

```text
FOR each PR opened by this parent (bottom → top):
  [ ] Rubber-duck walkthrough of the PR diff (explain aloud / in notes)
  [ ] Adversarial pass (invariants, leakage, gate weakening, size growth, …)
  [ ] All review comments resolved or answered with code/docs
  [ ] All relevant status checks green OR documented billing-budget block
  [ ] Squash-merged (gh stack merge --yes --squash for a ready prefix,
      or per-PR squash when landing partially)
  [ ] Higher layers rebased/synced after lower merges (gh stack sync)
```

If closeout is blocked (human review required by policy, budget, or secrets),
state the block explicitly — **open PRs without a closeout plan are incomplete
work**, same class of failure as missing experiment docs.

## Interaction with other skills

| Concern | Skill |
| --- | --- |
| Minimal implementation | `ponytail` |
| Path placement / moves | `organize-repository` |
| Experiment / eval / matrix runs | `documenting-experiment-results`, `honest-ship-eval`, `running-experiment-matrices` |
| Data builds | `synthesis-feedback` |
| Dashboard page edits | `dashboard-openui-parity` |
| Verbose shell | `rtk` |
| Large tool output | `headroom` |

## Red flags

| Smell | Fix |
| --- | --- |
| Parent implements a 5-phase epic alone | Decompose; spawn subagents; stack |
| One mega-PR "to save time" | Split with `gh stack`; reviewer time is the bottleneck |
| Subagent never commits until the end | Enforce incremental check-ins |
| Fix for layer 1 committed on layer 3 | `gh stack down`, fix on the right layer, `rebase --upstack` |
| Stack submitted then abandoned | Closeout is part of the task |
| **Pushed branch, no PR; agent asks "want a PR?"** | **Open the PR immediately; start closeout — never ask** |
| **Agent stops after push/PR open and waits** | **Continue review → CI → squash-merge unless human said stop** |
| Status checks red, merged anyway | Only billing-budget exceed may pause; do not merge red otherwise |
| Merge commits / non-squash land | Use `--squash` for this repo's agent-landed stacks |
| Sparse-checkout on shared worktree blanks another agent | Separate worktree + cone per task |
| Skills only installed for one harness | Canonical copy under `.agents/skills/sdlc` + discovery symlinks |

## Done means

- Every planned layer has an opened PR that is **squash-merged** or explicitly
  dropped/closed with reason (human “don’t merge” counts as a documented stop)
- Rubber-duck + adversarial review posted and acted on for each PR
- Relevant status checks green (or documented billing-budget block)
- Stack synced/pruned locally
- Required docs / model card / version stamps from other skills are done
- Parent reported **merged** PR URLs + merge SHAs + residual risks — not just
  a branch name or “ready when you are”
