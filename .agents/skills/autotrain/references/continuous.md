# Continuous model and harness improvement

Bare `/autotrain` is a **hands-off, non-terminating** objective: a sequence of
bounded `ExperimentCampaignV1` cycles that the agent keeps chaining until the
session itself ends or a **repeated hard block** is hit. Do not run one
unbounded shell process. Every command still obeys `MAX_RUN_MINUTES`;
persistence is the host goal and the append-only campaign event chains.

## Absolute loop law (never violate)

1. **Never stop for user confirmation** between cycles. Never ask "continue?",
   never wait for another `/autotrain`, never treat a finished cycle as done.
2. **Never end a bare `/autotrain` turn with only a resume command.** After
   every cycle: **print the four-table liveness/result/diagnostic/priority view to the user**, then
   **immediately start the next cycle** in the same turn (or the next agent
   step without user input when the host supports autonomous continuation /
   persistent goals).
3. **Self-heal failures.** Path errors, missing suites, dirty trees, merge
   conflicts, bad matrices, and failed gates are inputs to the next cycle — not
   reasons to yield. Repair when evidence names a canonical harness family;
   otherwise change knobs and re-run.
4. **Stop only when blocked.** Report `blocked` only after the same hard
   blocker has failed **three consecutive cycles** with no new information
   (e.g. missing credentials the agent cannot obtain, theorem contradiction
   unrepaired after three formal attempts). Soft failures (ship gates fail on
   fixture n, null lever deltas, timeouts) **never** stop the loop.
5. **Remote compute is opt-in.** No paid GPU, remote job, or HF write unless
   the user already granted that authority in this session.
6. **Code delivery is not local-only, but stack layers are selective.** While
   the loop is active, follow `sdlc` Phase A: incremental commits every cycle,
   document every run, **open/update a stacked PR only after a positive-result
   run**, get latest, resolve conflicts. Non-positive cycles (fixture gate
   fails, null deltas, timeouts without a win) stay local commits + docs.
   When the loop **stops**, run `sdlc` Phase B bottom-up closeout on open
   positive layers. Binding reference:
   [`../../sdlc/references/autotrain-iteration-delivery.md`](../../sdlc/references/autotrain-iteration-delivery.md).
7. **Never kill a live continuous driver or its train/eval children** to ship
   docs, fix CI, open PRs, “clean the worktree,” or restart for convenience.
   Use a separate worktree/branch for side work. The only legal stop is the
   user stop / hard-block / session-end Phase B path.
8. **Prove liveness from process truth, not the host UI.** Grok/Codex
   “background operations” panels are **not** the continuous loop. Always
   report from `/tmp/autotrain-loop-status.txt` or `bash /tmp/autotrain-report.sh`
   (driver_state, PID, top child, latest campaign) plus the matrix.

## User-facing report (mandatory whenever you talk about the loop)

Agents **must** paste this into chat after each cycle and whenever the user
asks if training is running:

```bash
bash /tmp/autotrain-report.sh
# writes:
#   /tmp/autotrain-loop-status.txt      # liveness
#   /tmp/autotrain-loop-matrix.md       # compact four-table view
#   /tmp/autotrain-loop-dashboard.md    # combined dashboard
```

Minimum paste: **Run Results** table (last 5 cycles) + one-line liveness
(`driver_state=RUNNING pid=… latest=… top_child=…`). Diagnostic/priorities
tables follow when non-empty and truncated if huge.

Do **not** say “it’s running” with only a remembered PID or a Grok UI icon.

## Preferred hands-off supervisor

The host agent owns an **unbudgeted persistent goal** and runs one bounded cycle
at a time. The agent must regain control between cycles to repair canonical
harnesses, handle Lean dispositions, commit durable docs, and perform delivery:

```bash
# local-only continuous loop worktree, clean tree required
python -m scripts.run_autotrain_continuous \
  --loop-id continuous-openui-local \
  --supervised --max-cycles 1 \
  --train-version wf_smoke_v2 \
  --steps 20
```

The invocation writes `<campaign>/cycle_handoff.json` and refreshes
`loops/<loop-id>/state.json`. Validate the handoff, execute every required
owner skill, print the compact matrix, commit the cycle, get latest, and start
the successor. `--max-cycles 0` remains a legacy unbounded executor; bare
`/autotrain` does not use it because a blocking process cannot close repairs or
delivery between cycles.

Every prerequisite action needs an append-only receipt before the successor can
start. After executing action index `<i>` from the handoff, bind its evidence:

```bash
python -m scripts.autoresearch --root outputs/autoresearch ack-action \
  --loop-id <loop-id> --campaign-id <campaign-id> --action-index <i> \
  --status completed --evidence <durable-path-or-commit>
```

Use `--status blocked` only with evidence of the real external blocker. Receipts
are action-content-bound; editing/reordering a handoff cannot satisfy an old action.
The driver enforces receipts for harness, Lean, data, docs, and delivery actions.
`next_experiment`, `retry_measurement`, and `monitor` are execution/steering actions,
so they are not predecessor prerequisites.

When `checkpoint_documentation_required` is true, update
`docs/MODEL_CARD.md` and the README model-card summary for every path listed in
`checkpoint_paths` before the next cycle.

After each cycle the driver runs **SDLC Phase A classification** and writes:

- `<campaign>/sdlc_delivery.json` — positive?, stack_layer?, reasons, metrics
- `<campaign>/cycle_handoff.json` — climb vs ship state, formal status, ranked
  priorities, and evidence-bound supervisor actions
- `outputs/autoresearch/sdlc_delivery_ledger.jsonl` — append-only ledger
- `outputs/autoresearch/loops/<loop-id>/state.json` — heartbeat and resumable phase
- log lines `SDLC_PHASE_A POSITIVE|NON_POSITIVE …`

**Stacked PRs only for positive evidence after documentation creates a reviewable
delta.** The driver does not
open PRs itself; the agent must `gh stack` positive layers. Non-positive
cycles stay local (no new stack layer). Positive is **not** “cand latency
lower”; it is a quality-aware win (see classification above).

## Start or resume (automatic)

1. Prefer a dedicated local branch/worktree. Keep the continuous loop tree
   clean of unrelated WIP (stash or isolate).
2. Before **every** decision-bearing run:

   ```bash
   git fetch origin main
   git merge --no-edit origin/main
   git status --short
   git rev-parse origin/main HEAD
   ```

   Resolve conflicts; repair failing repo checks. Never rebase away experiment
   provenance. Never begin a run with unresolved conflicts or tracked dirt on
   the loop worktree.
3. Create or resume an **unbudgeted** goal whose only exit is explicit user stop
   or the repeated-blocker rule. Do **not** mark the goal complete when a cycle
   finishes. Do not replace a missing goal with a blocking multi-cycle process.
4. If no host goal API exists, **emulate it in-session**: after cycle N
   closeout, start cycle N+1 without user text. Keep going until the session
   is preempted or the repeated-blocker rule fires.
5. Initialize each cycle with loop lineage:

   ```bash
   python -m scripts.autoresearch init --campaign-id <cycle-id> --loop-id <loop-id> \
     --cycle-index <n> [--predecessor-campaign-id <prior-cycle>] \
     --upstream-commit <fetched-origin-main-sha> \
     --integration-commit <merged-head-sha> \
     --objective "<falsifiable objective>" --primary-metric <metric>
   ```

   `upstream_commit` must be current `origin/main`; `integration_commit` must
   be clean `HEAD` containing that upstream. Locked
   `ExperimentCampaignV1.source_commit` equals integration and
   `source_dirty=false`.

## Iterate forever (one cycle body)

For each cycle, run the full body without pausing:

1. **Research** — offline-safe when network is unavailable.
2. **Hypothesize** — ≥5 candidates; continuous matrices need ranked
   `NextRunPriorityV1`. Bind real `train_version` / `eval_version` knobs (never
   both `data_source` and `train_version`; never leave eval on a missing suite).
3. **Validate** recommended experiment file.
4. **Lock + execute** with preregistered `ExperimentCampaignV1` manifests under
   the wall cap. Budget ≥1 control and ≥1 candidate when the primary metric is
   comparative.
5. **Diagnose** outcomes; write hypothesizer feedback.
6. **Document** JSON + markdown under `docs/design/` (`documenting-experiment-results`).
   Acknowledge the matching `document` action with the durable doc path; if a
   checkpoint exists, the evidence must also cover MODEL_CARD + README.
7. **Print** the compact four-table view:

   ```bash
   python -m scripts.autoresearch status --loop-id <loop-id> --matrix --last 5
   ```

8. **Audit** model vs harness. Open a harness lane only for a typed
   `HarnessSignalV1` reproduced on the frozen input naming exactly one family:
   `autoresearch`, `annotations`, `distill`, `experiments`, `model_build`,
   `preference`, `quality`, `rl`, `test_data`, or `train_data`. Route through
   `improve-openui-harnesses`, repair, replay the identical arm. Never mix
   harness and model changes in one attribution arm.
   A `repair_harness` handoff action is executable: invoke its owner skill,
   change the canonical owner, add a regression test, and replay the frozen arm
   before any new model hypothesis. Acknowledge the repair action with the commit
   or regression artifact; a recommendation without a receipt is still pending.
9. **SDLC Phase A — iteration delivery.**
   While fixing for this cycle and before the next train:
   - Incremental commits of green units (never leave harness WIP uncommitted).
   - Classify **positive** vs not with **quality-aware tradeoffs** (see
     `scripts/run_autotrain_continuous._classify_metric_tradeoff`): a pure
     latency blip with empty meaning is **not** positive; latency wins require
     held parse/mpr and mpr ≥ ~1/3; quality/efficiency wins may spend a
     bounded latency budget. Fixture `insufficient_n` / null deltas alone are
     **not** positive.
   - **If positive:** after documentation, stack layer for that iteration's tracked code +
     `docs/design/` results (`gh stack add` / `gh stack submit --open`, or
     push to the open positive layer if the same concern continues).
     Acknowledge `deliver_stack` with the merged commit/PR evidence.
   - **If not positive:** no new stack layer; keep local commits and continue.
   - Never PR raw `outputs/` or weight blobs.
   Full checklist:
   [`../../sdlc/references/autotrain-iteration-delivery.md`](../../sdlc/references/autotrain-iteration-delivery.md).
10. **Immediately** get latest and start the next cycle with incremented
    `cycle_index` and `predecessor_campaign_id`:

    ```bash
    git fetch origin main
    gh stack sync    # when a stack is active; else: git merge --no-edit origin/main
    # resolve conflicts on the owning layer; re-check policy/tests
    ```

    The supervised driver verifies this integrated clean tree but does not
    fetch, merge, commit, push, or open PRs itself.

Owner skills (invoke; do not reimplement):

| Signal or stage | Required owner skill |
| --- | --- |
| Campaign execution and typed feedback | `openui-autoresearch` |
| Canonical harness diagnosis/repair | `improve-openui-harnesses` |
| Lean metric certificate / band miss | `improve-lean-optimums` |
| Data synthesis feedback | `synthesis-feedback` |
| E*/matrix execution | `running-experiment-matrices` |
| Evaluation/readiness interpretation | `honest-ship-eval` |
| Run evidence and closeout | `documenting-experiment-results` |
| Prior-work/knowledge refresh when needed | `autoresearch` |
| Iteration commits, positive-only stacked PRs, get-latest, stop closeout | `sdlc` (+ autotrain-iteration-delivery) |

### Continuous recipe defaults (fail closed)

- Prefer published fixture train data (`train_version`, e.g. `wf_smoke_v2`) for
  wall-capped cycles unless the objective requires a new build.
- Prefer a published eval snapshot that has a `smoke` suite (e.g.
  `e938_role_safe_all_targets_v2`). If the compiler default would point at a
  missing suite, set `eval_version` explicitly.
- Ship gates stay on for honesty. Fixture quality/volume gate fails are
  **expected diagnostics**, not loop terminators.
- Size-match comparative arms; charge capacity growth with `EG_params`.

### Lean / formal

When a campaign carries `metric_expectations_sha256`, replay the v2 certificate
before promotion or successor selection. In-band → continue. Assumption-backed
miss → block promotion and cover five lanes in matrix + priorities. Theorem
contradiction → formal repair path; ordinary training waits until repaired
(still loop-active on formal work; not a user prompt).

## Automated promotion and judge changes

No human approval for ordinary local promotion. Researcher/hypothesizer or
harness changes promote only after frozen benchmarks pass. Judge/threshold
changes need a separate preregistered meta-campaign with unchanged held-out
controls. Never lower gates, train on frozen cases, or weaken decode invariants.

## When training stops (SDLC Phase B — mandatory)

Any of: user stop/pause, thrice-repeated hard block, session end of the loop.

1. Stop starting new train/eval cycles.
2. Inventory open autotrain stack layers (`gh stack view`) — these should only
   be positive-result layers.
3. Run **full** `sdlc` closeout **bottom → top** — rubber-duck + adversarial
   review, resolve comments, fix CI, squash-merge, `gh stack sync` — exactly
   as documented in
   [`../../sdlc/references/closeout-review.md`](../../sdlc/references/closeout-review.md)
   and
   [`../../sdlc/references/autotrain-iteration-delivery.md`](../../sdlc/references/autotrain-iteration-delivery.md)
   Phase B.
4. Report merged PR URLs + SHAs + residual risks.

Do **not** end with only a resume command or “branch is ready when you are.”

## Forbidden continuous-mode behaviors

- Stopping after one cycle to “report status and wait”
- Asking the user to re-invoke `/autotrain`
- Treating ship-gate fails on fixture `n` as terminal
- Leaving the loop because an experiment failed once
- Skipping docs closeout to go faster
- Remote/HF/paid compute without prior authority
- Days of uncommitted harness/docs fixes while cycles run
- Opening a stacked PR for a non-positive cycle (fixture fail / null delta)
- Skipping stacked PR after a **positive** run that changed code/docs
- Stacking or committing raw `outputs/` / checkpoint blobs
- Training stopped without bottom-up `sdlc` closeout of open positive layers
- **Killing** `run_autotrain_continuous` / train / eval to land unrelated work
- Claiming the loop is running from Grok/Codex background UI alone
- Reporting cycle progress **without** pasting the skill matrix tables
