# Autotrain iteration delivery (SDLC × autotrain)

How `autotrain` must ship **code and durable docs** while training iterations
run, and how it must close the stack when training stops. Owner skills:
`autotrain` (run loop) + `sdlc` (delivery process). This reference is binding
for both.

## Scope split

| Artifact | Where it lives | Delivery |
| --- | --- | --- |
| Harness / model / eval / script code | tracked tree | **Incremental commits + stacked PR layers** |
| Measured results under `docs/design/` | tracked tree | Same layer as the code that produced them (iron law) |
| Model card / version stamps | tracked tree | Same layer when a checkpoint/component change requires them |
| Raw `outputs/`, local checkpoints, OTLP, wall logs | gitignored / local | **Never** stack or PR these blobs |
| Paid GPU / HF write / remote train | external | Still needs **explicit** session authority |

Stack the *code and design docs*, not the raw run directory.

## Two phases

```text
PHASE A — TRAINING ACTIVE (continuous or multi-run finite)
  incremental commit → train/eval → stack layer PR → get latest → next run

PHASE B — TRAINING STOPPED (user stop, hard block, session end of loop)
  bottom-up SDLC closeout for the whole stack (rubber-duck → CI → squash-merge)
```

Do **not** skip Phase A check-ins "until the loop ends." Do **not** run full
stack squash-merge mid-loop unless a lower layer is independently shippable
and CI-green (rare); default is open/update the stack between runs and land
everything in Phase B.

## Phase A — between every training run

Required whenever the agent is:

- working on a training iteration (recipe, knobs, harness, data, docs for a run), or
- fixing code so the next training iteration can execute honestly.

### A1. Incremental commits (during the iteration)

While editing for the current run:

1. Commit as soon as a coherent unit is green locally (tests/policy for touched
   paths). Prefer small commits on the **current stack layer branch**.
2. Message: why this change exists for the iteration (not "wip").
3. Never hold a day of uncommitted harness/docs "for a clean history."
4. Do not mix unrelated refactors into the iteration layer.

### A2. After the run (or after a blocking code fix)

Before starting the **next** training run:

```text
1. Document the run (documenting-experiment-results) — JSON + markdown under
   docs/design/ when the iron law triggers.
2. git status — stage only iteration code + design docs (never outputs/).
3. Incremental commit(s) for anything still uncommitted.
4. Ensure a gh stack exists for this autotrain campaign:
     gh stack init autotrain/<loop-id>-L01-<slug>   # first code layer
     # later iterations:
     gh stack add autotrain/<loop-id>-L<nn>-<slug>
5. Push + open/update PRs for the new/updated layer:
     gh stack submit --open
   (or gh pr create if this is still a single-layer stack)
6. git fetch origin main
   Integrate trunk into the stack without destroying provenance:
     gh stack sync          # preferred
     # or: merge origin/main into bottom, then gh stack rebase --upstack
7. Resolve merge conflicts on the owning layer; re-run local checks.
8. Only then start the next train/eval cycle.
```

One **stack layer per training iteration's code+docs** is the default when
iterations change different concerns (harness fix vs recipe vs docs). If the
same concern continues (tiny knob follow-up on the same PR), amend with
incremental commits on that layer and `gh stack push` — do not invent empty
layers.

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
| Local commit | Yes (required) |
| Push stack branches + open/update stacked PRs | Yes (required for code/docs) |
| Squash-merge entire stack to main | **No** by default — wait for Phase B |
| Squash-merge a green bottom layer that is pure infra and unblocks others | Yes if CI green and layer is independently shippable |
| Paid remote GPU / HF bucket write | Only with prior user authority |
| Stopping to ask "continue?" | Never |

Opening stacked PRs between runs is **not** a user-confirmation step; it is
delivery process. Do not ask permission to open the next layer PR.

## Phase B — training stopped → full SDLC closeout

Triggers (any one):

- User says stop / pause training / end autotrain
- Hard block after three identical unrecoverable failures
- Session is ending and the loop will not continue
- Agent switches from continuous train to "land the work"

Then the parent **must** run the existing SDLC closeout, **bottom → top**:

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

**Incomplete closeout = incomplete autotrain delivery**, even if the loop
printed matrices for days.

Do **not** treat "training stopped" as "report status and wait for the user
to open PRs." Open anything still missing, then closeout.

## Naming conventions (recommended)

```text
autotrain/<loop-id>-L01-<short-slug>   # bottom
autotrain/<loop-id>-L02-<short-slug>
autotrain/<loop-id>-L03-<short-slug>
```

PR titles: `autotrain(<loop-id>): L0N <one-line why>`  
Body: campaign/cycle id, primary metric, link to measured-results path,
what the next run will use.

## Anti-patterns

| Smell | Fix |
| --- | --- |
| Days of uncommitted harness fixes while cycles run | Commit each green unit |
| One mega-PR at the end of continuous training | Layer per iteration (or per concern) as you go |
| Stacking `outputs/` or `.pt` checkpoints | Keep local / bucket; never PR blobs |
| Merge main by rewriting measured-results history | Resolve conflicts; keep provenance |
| Full closeout mid-loop for every tiny knob | Open/update layer between runs; land in Phase B |
| "Local-only" used to skip all PRs for code fixes | Local-only = no paid remote compute; code still ships via stack |
| Stop training and only paste a resume recipe | Phase B closeout is mandatory |

## Pointers

- Stack commands: [stacked-prs.md](stacked-prs.md)
- Closeout checklist: [closeout-review.md](closeout-review.md)
- Continuous loop body: [`../../autotrain/references/continuous.md`](../../autotrain/references/continuous.md)
- Autotrain facade: [`../../autotrain/SKILL.md`](../../autotrain/SKILL.md)
