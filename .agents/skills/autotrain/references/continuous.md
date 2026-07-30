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
   every cycle: print the matrix, then **immediately start the next cycle** in
   the same turn (or the next agent step without user input when the host
   supports autonomous continuation / persistent goals).
3. **Self-heal failures.** Path errors, missing suites, dirty trees, merge
   conflicts, bad matrices, and failed gates are inputs to the next cycle — not
   reasons to yield. Repair when evidence names a canonical harness family;
   otherwise change knobs and re-run.
4. **Stop only when blocked.** Report `blocked` only after the same hard
   blocker has failed **three consecutive cycles** with no new information
   (e.g. missing credentials the agent cannot obtain, theorem contradiction
   unrepaired after three formal attempts). Soft failures (ship gates fail on
   fixture n, null lever deltas, timeouts) **never** stop the loop.
5. **Local-only by default.** No push, PR, merge to shared main, paid GPU,
   remote job, or HF write unless the user already granted that authority in
   this session.

## Preferred hands-off driver

When available, run the bounded multi-cycle driver instead of hand-rolling
each CLI step (it still uses the same `scripts.autoresearch` contracts):

```bash
# local-only continuous loop worktree, clean tree required
python -m scripts.run_autotrain_continuous \
  --loop-id continuous-openui-local \
  --max-cycles 0 \
  --train-version wf_smoke_v2 \
  --steps 20
```

`--max-cycles 0` means keep going (cap 1024). Child train/eval still obey
`MAX_RUN_MINUTES`. Soft failures (ship-gate fails, wall timeout on full
suites, null deltas) do **not** stop the driver. Agents may also chain cycles
manually, but must still obey the Absolute loop law above.

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
3. If the host supports persistent goals / autonomous continuation, create or
   resume an **unbudgeted** goal whose only exit is the repeated-blocker rule.
   Do **not** mark the goal complete when a cycle finishes.
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
7. **Print** the three-table matrix:

   ```bash
   python -m scripts.autoresearch status --loop-id <loop-id> --matrix --last 5
   ```

8. **Audit** model vs harness. Open a harness lane only for a typed
   `HarnessSignalV1` reproduced on the frozen input naming exactly one family:
   `autoresearch`, `annotations`, `distill`, `experiments`, `model_build`,
   `preference`, `quality`, `rl`, `test_data`, or `train_data`. Route through
   `improve-openui-harnesses`, repair, replay the identical arm. Never mix
   harness and model changes in one attribution arm.
9. **Immediately** fetch/merge main again and start the next cycle with
   incremented `cycle_index` and `predecessor_campaign_id`.

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

## Forbidden continuous-mode behaviors

- Stopping after one cycle to “report status and wait”
- Asking the user to re-invoke `/autotrain`
- Treating ship-gate fails on fixture `n` as terminal
- Leaving the loop because an experiment failed once
- Skipping docs closeout to go faster
- Remote/HF/paid actions without prior authority
