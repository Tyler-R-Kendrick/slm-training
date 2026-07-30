# Scalar, sparse-checkout, and workspaces

Goal: humans and agents get **fast, correct, isolated** working trees without
downloading or materializing the entire history/blob set when they do not need
it — and without stepping on each other.

## Scalar (`scalar`)

[Scalar](https://git-scm.com/docs/scalar) is Git's large-repo manager. When you
**register** a repository, it configures scale-oriented settings and schedules
background maintenance.

### What `scalar register` does

Behind the scenes (exact knobs evolve with Git version):

| Capability | Effect |
| --- | --- |
| **Background maintenance** | Automates commit-graph updates, multi-pack index / incremental repack, loose-object cleanup, scheduled prefetch so everyday `git fetch` / `git status` stay snappy |
| **Opinionated Git config** | Large-repo friendly index/pack settings (e.g. modern index version, pack bitmaps/sparse pack use, maintenance strategy) |
| **Enlistment tracking** | Adds the path to `maintenance.repo` (global) so cron/systemd maintenance can find it |

Register **once per machine enlistment** (the primary clone that owns
`.git` and hosts worktrees), not once per worktree:

```bash
# Preferred: register the main clone
scalar register /path/to/slm-training

# Or from inside that enlistment:
cd /path/to/slm-training && scalar register

scalar list
scalar reconfigure /path/to/slm-training   # after Git/Scalar upgrades
```

This repo's primary enlistment on the shared host is typically:

```text
/home/codex/repos/slm-training
```

Linked worktrees under e.g. `/home/codex/.herdr/worktrees/slm-training/*` share
that object database — registering the main clone covers them.

### What `scalar clone` does (new machines)

```bash
# Default: partial clone + sparse-checkout (top-level only), worktree at <enlistment>/src
scalar clone https://github.com/Tyler-R-Kendrick/slm-training.git ~/enlistments/slm-training

# Flat layout (no nested src/):
scalar clone --no-src https://github.com/Tyler-R-Kendrick/slm-training.git ~/src/slm-training

# Need the full tree immediately:
scalar clone --full-clone --no-src https://github.com/Tyler-R-Kendrick/slm-training.git ~/src/slm-training
```

| Mechanism | Meaning |
| --- | --- |
| **Partial clone** | Skips downloading large blobs until a checkout actually needs them |
| **Sparse-checkout** | Restricts the working directory to paths you opt into (default: top-level only after `scalar clone`) |
| **Background maintenance** | Same family of tasks as `register` |

Then expand what you need:

```bash
cd ~/enlistments/slm-training/src   # or enlistment root if --no-src
git sparse-checkout set src scripts docs tests .agents .github
# or disable sparse entirely:
git sparse-checkout disable
```

### Maintenance commands

```bash
scalar run all /path/to/slm-training
scalar run commit-graph /path/to/slm-training
scalar diagnose /path/to/slm-training    # zip for bug reports
```

Usually unnecessary day-to-day once registered (cron schedule owns it).

## Sparse-checkout (humans and agents)

Use **cone mode** (default in modern Git) so patterns are directory-based and
fast.

```bash
git sparse-checkout init --cone
git sparse-checkout set \
  src/slm_training \
  scripts \
  tests \
  docs \
  .agents \
  .github \
  src/apps/dashboard   # only if the task needs the dashboard
git sparse-checkout list
git sparse-checkout add docs/design   # grow the cone
git sparse-checkout disable           # full tree again
```

### Agent-oriented cones (examples)

| Task family | Start with |
| --- | --- |
| Python harness / model | `src/slm_training` `scripts` `tests` `docs` `.agents` |
| Dashboard OpenUI parity | `src/apps/dashboard` `src/slm_training/web` `scripts` `docs` `.agents` |
| Docs-only | `docs` `.agents` `README.md` `AGENTS.md` |
| Lean optimums | `src/leverproof_lean` `src/slm_training` `scripts` `tests` `docs` |

Always keep enough of the repo to run the checks you claim: policy scripts,
hooks, and version stamp registry often live at roots (`.githooks`,
`pyproject.toml`, `src/slm_training/resources/versions.json`). When unsure,
`git sparse-checkout add` the path rather than inventing files.

### Safety with multi-agent hosts

- Sparse-checkout is **per worktree** when `extensions.worktreeConfig=true`
  (this enlistment uses that).
- **Never** cone-out a shared worktree that other agents still use.
- Prefer: **new worktree + cone for this task** over mutating the shared full
  tree.

## Worktrees and agent workspaces

### Why worktrees

| Pattern | When |
| --- | --- |
| One worktree per stack | Parent owns stack; clean `main` left alone |
| One worktree per long-running agent | Avoids branch thrash and dirty-tree collisions |
| Isolated subagent worktree | When the harness supports worktree isolation; parent still owns `gh stack` |

```bash
# From main enlistment
git fetch origin
git worktree add ../slm-training-wt-feature -b agent/feature origin/main
cd ../slm-training-wt-feature
# optional cone for this worktree only:
git sparse-checkout init --cone
git sparse-checkout set src scripts tests docs .agents

# later
git worktree remove ../slm-training-wt-feature
```

### Parent vs subagent filesystem rules

1. Parent records the worktree path in the task brief.
2. Subagent works **only** in the assigned worktree/branch.
3. Subagent does not `git worktree remove` the parent's tree.
4. Large ignored artifacts stay under `outputs/` (never new sibling data roots).
5. Tracked moves still use `git mv` + `organize-repository`.

### Herdr / multi-worktree layout (example)

```text
/home/codex/repos/slm-training              # scalar-registered enlistment (.git)
/home/codex/.herdr/worktrees/slm-training/
  repo/                                     # agent workspace worktree
  lean4/
  …
```

Register Scalar on the **enlistment** (`repos/slm-training`), not on every
worktree path.

## Recommended setup checklist (human or bootstrap agent)

```text
[ ] gh auth login (repo, workflow)
[ ] gh extension install github/gh-stack && gh stack alias
[ ] scalar register <main-clone>
[ ] scalar list shows the enlistment
[ ] git maintenance / cron present (scalar register installs schedule)
[ ] For a new task: worktree add + optional sparse cone
[ ] For multi-layer work: gh stack init in that worktree
```

## What not to do

| Don't | Why |
| --- | --- |
| `scalar delete` casually on a shared enlistment | Removes enlistment tracking; disrupts other agents |
| Enable sparse on the only full worktree mid-flight | Breaks other sessions expecting full paths |
| Partial clone without understanding blob fetch | First checkout of large binaries may stall; plan cones |
| One dirty shared tree for three agents | Collisions, partial commits, lost work |
| Treat `outputs/` as a second git root | Violates repository organization |

## Quick diagnosis

```bash
scalar list
git worktree list
git sparse-checkout list
git rev-parse --is-inside-work-tree
git config --get remote.origin.partialclonefilter   # if partial clone
git maintenance run --auto
```
