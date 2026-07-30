# Stacked PRs — official `gh stack` / `gs`

Source of truth: [GitHub Stacked PRs](https://gh.io/stacks),
[CLI reference](https://github.github.com/gh-stack/reference/cli/),
[typical workflows](https://github.github.com/gh-stack/guides/workflows/).

This repo uses the **official** extension `github/gh-stack`, not Graphite or
hand-rolled base-branch chains (unless recovering a pre-existing stack).

## Install and alias

```bash
gh extension install github/gh-stack
gh extension upgrade github/gh-stack
gh stack alias          # default: gs → ~/.local/bin/gs
# ensure ~/.local/bin is on PATH
gs version              # or: gh stack --version
```

Requires `gh` authenticated (`gh auth login`) with rights to push and open PRs.

## Core commands (agents)

Prefer non-interactive flags in automation.

| Step | Command |
| --- | --- |
| Start stack | `gh stack init <bottom-branch>` |
| Next layer | `gh stack add <branch>` |
| Commit+add (abbrev) | `gh stack add -Am "msg" [branch]` |
| View | `gh stack view` / `gh stack view --json` |
| Push branches only | `gh stack push` |
| Create/update PRs + stack | `gh stack submit --auto` or `--open` |
| Cascading rebase | `gh stack rebase` / `--upstack` / `--downstack` |
| Fetch+rebase+push+PR sync | `gh stack sync` / `gh stack sync --prune` |
| Navigate | `gh stack bottom` / `up` / `down` / `top` / `trunk` |
| Restructure | `gh stack modify` (interactive TUI) |
| Land | `gh stack merge --yes --squash` |
| Drop local+remote stack meta | `gh stack unstack` |

**Do not** use plain `gh pr merge` for stacked PRs — it breaks the stack
contract. Use `gh stack merge`.

## Standard multi-layer flow

```bash
gh stack init 01-foundation
# ... commits on 01-foundation ...

gh stack add 02-impl
# ... commits ...

gh stack add 03-tests
# ... commits ...

gh stack submit --open
# review / CI / fixes ...

# After a mid-stack fix:
gh stack checkout 01-foundation   # or: gh stack bottom
# fix + commit
gh stack rebase --upstack
gh stack push

# Land when ready (this repo: squash)
gh stack merge --yes --squash
gh stack sync --prune
```

## Layering rules

1. **Bottom closest to trunk** — contracts, types, shared infra first.
2. **One concern per branch** — reviewer should understand the PR in isolation
   against its base (the layer below).
3. **No unrelated drive-bys** on a layer "while you're there."
4. **Mid-stack bugfix goes on the owning layer**, then rebase upstack — never
   paper over a lower bug in a higher PR.
5. **Linear history inside the stack** — rebases, not merge commits from trunk
   into mid-stack branches.

## Responding to review on a lower PR

```bash
gh stack checkout <branch-or-pr>
# implement feedback, commit
gh stack rebase --upstack    # or full: gh stack rebase
gh stack push
# reply to each comment on the PR
```

## Merge policy (this skill)

- Default land method: **squash** (`--squash` / `--merge-method squash`).
- Parent closeout may land the **whole ready stack** or a **bottom prefix**;
  after partial land, always `gh stack sync` so remaining PRs retarget correctly.
- Branch protection and required checks still apply; stack merge does **not**
  bypass them.
- If the repo uses a merge queue, `gh stack merge` queues the stack; the queue
  may choose the method — note that in the closeout summary.

## Exit codes worth knowing

| Code | Meaning |
| --- | --- |
| 2 | Not in a stack / stack not found |
| 3 | Rebase conflict — resolve, then `--continue` or `--abort` |
| 8 | Stack locked by another process |
| 9 | Stacked PRs not enabled for this repository (join waitlist / org enablement) |
| 10 | Modify session needs recovery |

## Agent notes

- Non-interactive `submit` uses `--auto` (drafts by default unless `--open`).
  Prefer **`--open`** when the parent is ready for review/CI on all layers.
- `gh stack view --json` is the structured status for parent orchestration.
- If exit code 9: stop and report that Stacked PRs are not enabled for the
  repo/org; fall back only with an explicit human decision (manual base-branch
  chaining is legacy and error-prone).
