# Autotrain iteration delivery (SDLC × autotrain)

How `autotrain` must ship **code and durable docs** while training iterations
run, and how it must close the stack when training stops. Owner skills:
`autotrain` (run loop) + `sdlc` (delivery process). This reference is binding
for both.

## Scope split

| Artifact | Where it lives | Delivery |
| --- | --- | --- |
| Harness / model / eval / script code | tracked tree | Incremental commits always; **stacked PR only on positive runs** |
| Measured results under `docs/design/` | tracked tree | Always document (iron law); **stack with the positive layer** that earns them |
| Model card / version stamps | tracked tree | Same layer when a checkpoint/component change requires them |
| Raw `outputs/`, local checkpoints, OTLP, wall logs | gitignored / local | **Never** stack or PR these blobs |
| Paid GPU / HF write / remote train | external | Still needs **explicit** session authority |

Stack the *code and design docs*, not the raw run directory.

## Two phases

```text
PHASE A — TRAINING ACTIVE (continuous or multi-run finite)
  edit + incremental commit → train/eval → document
  if POSITIVE RESULT → stack layer PR (gh stack add/submit)
  always → get latest → next run

PHASE B — TRAINING STOPPED (user stop, hard block, session end of loop)
  bottom-up SDLC closeout for open stack layers (rubber-duck → CI → squash-merge)
```

Do **not** open a stacked PR for every cycle. **Stacked PRs are only for
positive-result runs** (definition below). Do **not** skip local incremental
commits or docs "until the loop ends." Do **not** full-stack squash-merge
mid-loop unless a lower positive layer is independently shippable and CI-green;
default is land the open stack in Phase B.

## Positive result (required gate for a stack layer)

A run is **positive** only when at least one of the following is true on
honest, documented evidence for that cycle:

1. **Primary metric win** — the campaign `primary_metric` moves in the
   declared beneficial direction vs the matched control or locked predecessor
   under the same wall/recipe (size-matched when comparative), with the scoreboard
   and recipe recorded under `docs/design/`.
2. **Ship-quality win** — a multi-suite scoreboard clears its declared gates
   for a non-fixture / ship-claim suite (not wiring-only smoke), without gate
   weakening.
3. **Executable unblocking** — a harness/path/code fix removes a prior hard
   path error or unrecoverable blocker and the **identical arm** then completes
   with a usable scoreboard (replay-proven). Knob thrash that still fails the
   same way is **not** positive.

**Not positive** (never open a new stack layer for these alone):

- Fixture `insufficient_n` / expected ship-gate fails on smoke-scale data
- Null lever deltas, wall timeouts with no metric win
- Soft diagnosis-only cycles that only restate known residuals
- "It ran" without metric or unblocking evidence

When uncertain, treat as **not positive** and keep going (docs + local commits
only). Do not ask the user.

## Phase A — every training cycle

Required whenever the agent is:

- working on a training iteration (recipe, knobs, harness, data, docs for a run), or
- fixing code so the next training iteration can execute honestly.

### A1. Incremental commits (during the iteration)

While editing for the current run:

1. Commit as soon as a coherent unit is green locally (tests/policy for touched
   paths). Prefer small commits on a **local iteration branch** (or the open
   stack tip if one already exists).
2. Message: why this change exists for the iteration (not "wip").
3. Never hold a day of uncommitted harness/docs "for a clean history."
4. Do not mix unrelated refactors into the iteration.

Local commits apply to **every** cycle (positive or not). They are how WIP stays
recoverable; they are **not** the same as opening a stacked PR.

### A2. After every run (always)

```text
1. Document the run (documenting-experiment-results) when the iron law triggers
   — JSON + markdown under docs/design/. Negative runs still get docs.
2. Stage only iteration code + design docs (never outputs/).
3. Incremental commit(s) for anything still uncommitted.
4. Classify the run: positive? (see gate above)
5. IF POSITIVE:
     a. Ensure a gh stack exists for this autotrain campaign:
          gh stack init autotrain/<loop-id>-L01-<slug>   # first positive layer
          # later positive iterations:
          gh stack add autotrain/<loop-id>-L<nn>-<slug>
     b. Include the code + design docs that produced the win on that layer.
     c. Push + open/update PRs:
          gh stack submit --open
        (or gh pr create if still single-layer)
     d. PR body must cite: campaign/cycle id, primary metric delta or unblock
        proof, measured-results path, why the run is positive.
6. IF NOT POSITIVE:
     - Do **not** gh stack add / submit a new layer for this cycle.
     - Keep local commits; carry fixes into the next cycle.
     - Optional: note "no stack layer (non-positive)" in the cycle matrix line.
7. git fetch origin main and integrate (see A3).
8. Start the next train/eval cycle.
```

One **stack layer per positive iteration** (or per positive concern). If the
same positive concern continues (tiny follow-up that still improves the metric),
commit on that open layer and `gh stack push` — do not invent empty layers.

### A3. Get latest / conflicts (every decision-bearing run)

Before **every** decision-bearing train or eval:

```bash
git fetch origin main
# Prefer stack-aware sync when a stack is active:
gh stack sync
# Or on a non-stack delivery branch:
git merge --no-edit origin/main
git status --short
git rev-parse origin/main HEAD
```

- Resolve conflicts; repair failing repo checks.
- Never rebase away experiment provenance or drop measured-results docs.
- Never begin a run with unresolved conflicts or tracked dirt on the loop tree
  (stash or isolate unrelated WIP).

### A4. Authority during Phase A

| Action | Allowed during active training? |
| --- | --- |
| Local commit | Yes (required for code/docs units) |
| Push stack branches + open/update stacked PRs | **Only after a positive run** (or to update an existing positive layer) |
| New stack layer for a non-positive cycle | **No** |
| Squash-merge entire stack to main | **No** by default — wait for Phase B |
| Squash-merge a green bottom positive layer that unblocks others | Yes if CI green and independently shippable |
| Paid remote GPU / HF bucket write | Only with prior user authority |
| Stopping to ask "continue?" | Never |

Opening a stacked PR after a **positive** run is delivery process, not a user
confirmation step. Do not ask permission to open that layer. Do not open layers
for noise cycles.

## Phase B — training stopped → full SDLC closeout

Triggers (any one):

- User says stop / pause training / end autotrain
- Hard block after three identical unrecoverable failures
- Session is ending and the loop will not continue
- Agent switches from continuous train to "land the work"

Then the parent **must** run the existing SDLC closeout, **bottom → top**, for
**open stack layers** (which should only be positive-result layers):

1. Inventory the stack: `gh stack view` / `gh stack view --json`
2. For each unmerged PR from the **bottom**:
   - Rubber-duck + adversarial review
     ([closeout-review.md](closeout-review.md))
   - Post duck notes on the PR
   - Resolve all review comments
   - Fix all relevant CI (billing-budget block is the only pause)
   - `gh stack merge --yes --squash` for that PR or ready bottom prefix
   - `gh stack sync` so upper layers retarget
3. Repeat until every autotrain stack PR is merged or explicitly closed with
   reason
4. `gh stack sync --prune`; report merged PR URLs + SHAs + residual risks
5. Confirm iron-law docs / model card / version stamps for landed layers

If training stops with **only local commits** and no positive stack layers,
either (a) leave non-positive WIP unpushed / on a private branch with an
explicit note, or (b) open a single intentional PR only if the remaining
code is a pure infra fix that is independently valuable — still not a stack
of failed experiment noise.

**Incomplete closeout of opened (positive) layers = incomplete autotrain
delivery.** Do not end with only a resume recipe.

## Naming conventions (recommended)

```text
autotrain/<loop-id>-L01-<short-slug>   # first positive layer (bottom)
autotrain/<loop-id>-L02-<short-slug>
autotrain/<loop-id>-L03-<short-slug>
```

PR titles: `autotrain(<loop-id>): L0N <one-line positive why>`  
Body: campaign/cycle id, primary metric delta or unblock proof, link to
measured-results path, what the next run will use.

## Anti-patterns

| Smell | Fix |
| --- | --- |
| Days of uncommitted harness fixes while cycles run | Commit each green unit locally |
| Stacked PR for every fixture-fail smoke cycle | Stack **only** on positive results |
| One mega-PR at the end of continuous training | Layer per **positive** iteration as you go |
| Stacking `outputs/` or `.pt` checkpoints | Keep local / bucket; never PR blobs |
| Merge main by rewriting measured-results history | Resolve conflicts; keep provenance |
| Full closeout mid-loop for every tiny knob | Positive layers open mid-loop; land in Phase B |
| "Local-only" used to skip PRs after real wins | Positive runs still ship via stack |
| Stop training and only paste a resume recipe | Phase B closeout for open positive layers |

## Pointers

- Stack commands: [stacked-prs.md](stacked-prs.md)
- Closeout checklist: [closeout-review.md](closeout-review.md)
- Continuous loop body: [`../../autotrain/references/continuous.md`](../../autotrain/references/continuous.md)
- Autotrain facade: [`../../autotrain/SKILL.md`](../../autotrain/SKILL.md)
