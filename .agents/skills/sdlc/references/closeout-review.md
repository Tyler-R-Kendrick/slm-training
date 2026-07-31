# Closeout: bottom-up rubber-duck + adversarial review → squash-merge

**When:** Parent agent has finished implementing (via subagents) and opened
one or more PRs / a stack — **or** just pushed commits that are meant to land
— **or** an `autotrain` loop has **stopped** (user stop, hard block, session
end) and stacked iteration layers still need to land. Autotrain mid-loop uses
Phase A (open/update stack between runs); Phase B (this closeout) runs after
training stops — see
[autotrain-iteration-delivery.md](autotrain-iteration-delivery.md).

Closeout is **required** before the task is done. Opening the PR is part of
closeout prep if it is not open yet (`gh pr create` / `gh stack submit --open`).
Do not ask the user whether to open, review, or merge.

**Order:** Always **bottom → top** (closest to trunk first). Higher layers
assume lower layers; fixing a top PR while the bottom is wrong wastes work.

## Per-PR checklist (repeat for every PR the parent opened)

```text
1. Identify next unmerged PR from the bottom of the stack
2. gh stack checkout <that-pr-or-branch>
3. Rubber-duck review of the PR-only diff
4. Adversarial review pass
5. Address all PR comments + review threads
6. Fix all relevant status checks (unless billing budget exceeded)
7. Squash-merge that PR (or land approved bottom prefix via gh stack merge)
8. gh stack sync (rebase remaining layers onto new trunk/base)
9. Move up one layer; repeat until none remain
```

## 1. Rubber-duck mode

Explain the change as if to a skilled engineer who has never seen the PR:

- **Problem:** what user/system need does this layer solve?
- **Approach:** why this design, not the obvious alternative?
- **Diff tour:** file-by-file, what changed and why (use the PR diff against
  **its base**, not against trunk for mid-stack layers).
- **Risks:** what could break; what was deliberately not done.
- **Verification:** what was run; what CI will run; what is untested.

Write the duck notes in the PR comment or the parent handoff if they help the
next reviewer — do not only "think" them.

If you cannot explain a hunk, you do not understand it: re-read, or fix, or
split the layer.

## 2. Adversarial thinking

Attack your own PR. At minimum probe:

| Attack | Question |
| --- | --- |
| **Invariant break** | Does this weaken constrained decode, ship gates, honest eval, or any I* law? |
| **Gate gaming** | Did we weaken a threshold, shrink a suite, or swap fixture-demo for ship? |
| **Size cheat** | Did we grow params to buy quality without `EG_params` / size-matched arms? |
| **Leakage** | Train/test contamination, gold placeholders under honest contracts, eval in train? |
| **Sprawl** | New parallel harness, duplicate path, raw `mv` of tracked files? |
| **Silent fallback** | Unconstrained decode, missing dependency → widen behavior? |
| **Docs debt** | Experiment/checkpoint without `docs/design` / MODEL_CARD / version stamp? |
| **Security** | Secrets in diff, unsafe defaults, trust-boundary skips? |
| **API / contract** | Callers broken? Fingerprints/registries needing rebuild? |
| **Stack hygiene** | Does this layer include fixes that belong below? Unrelated drive-bys? |

Fail any serious hit → fix on **this** layer (or the correct lower layer),
rebase upstack, re-run checks. Do not merge and "follow up later" for
invariant or gate issues.

## 3. Comments and review feedback

```bash
# List review threads / comments (adjust as gh evolves)
gh api repos/{owner}/{repo}/pulls/<n>/comments
gh api graphql -f query='…reviewThreads…'   # when needed
gh pr view <n> --comments
```

Rules:

- **Every** unresolved review thread: either code/docs fix + reply, or a
  written non-action with rationale (and human agreement if it is a product call).
- Do not resolve a thread without addressing the substance.
- Nitpicks: fix if cheap; otherwise reply and leave consistent with repo style.
- After fixes: commit on the correct layer, `gh stack rebase --upstack`,
  `gh stack push`, then re-request review if required.

## 4. Status checks

```bash
gh pr checks <n>
gh pr view <n> --json statusCheckRollup,mergeable,reviewDecision
```

- Fix **all relevant** failing checks (lint, tests, policy, required CI).
- Re-run flaky checks once with evidence; fix if reproducible.
- **Billing / budget exceeded** (CI minutes, hosted runner quota, org spending
  limit): document on the PR, do **not** invent a merge bypass, do **not**
  treat as a green check. Stop land for that layer until budget is restored or
  a human explicitly waives with a durable note.
- Local reproduction when CI is opaque:
  `python -m scripts.repo_policy`,
  `.githooks/check-changed`, targeted pytest, etc.

## 5. Squash-merge

This skill standardizes agent-landed stacks on **squash**:

```bash
# Preferred: land entire ready stack or bottom prefix
gh stack merge --yes --squash

# Or merge up to a specific PR number
gh stack merge <pr-number> --yes --squash
```

After any land:

```bash
gh stack sync --prune
gh stack view
```

Do **not** use merge commits for agent closeout unless branch protection
forces it (then record that in the handoff).

## 6. Bottom-up discipline

```text
main ← PR#A (bottom) ← PR#B ← PR#C (top)

Closeout order: A → B → C
Never: C first, or B while A is still red/unreviewed
```

Why:

- Review of B assumes A is the base you will actually land.
- CI for upper layers is only honest once lower layers are final.
- Squash of A rewrites the base of B; `sync` must run before re-reviewing B.

If A needs changes after B was approved, re-approve B after rebase — do not
assume sticky approval across force-with-lease pushes if rules require re-review.

## 7. Definition of done (parent)

- [ ] PR(s) exist for every intended land (created without asking)
- [ ] Every PR opened by this parent is squash-merged **or** closed with reason
- [ ] Rubber-duck + adversarial notes posted on each PR
- [ ] No open review threads left dangling on merged PRs
- [ ] No red relevant checks ignored (except documented billing block)
- [ ] Stack synced/pruned; local branches cleaned
- [ ] Cross-skill obligations done (docs, model card, version stamps, …)
- [ ] Handoff lists **merged** PR URLs, merge commits, residual risks

Incomplete closeout = incomplete task. “Pushed; want a PR?” is a process bug.
