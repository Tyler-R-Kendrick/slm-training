# Autotrain continuous-openui-p33lme c4: component-edge screen — measurement incomplete (wall timeout)

**Outcome:** non-positive, **not a model measurement** — the queued next
model hypothesis from cycle 3
(`c20260802-continuous-openui-p33lme-489d3aa7-c3-component-edge`), executed
as this loop's cycle 4, campaign
`continuous-loop-20260802-continuous-openui-p33lme-489d3aa7-c4`, integration
commit `5441b2037451127d52967db8a98d6af09a99176a`. Both arms (CPU scratch
TwoTower, `wf_smoke_v2`, lexer output, 21 steps, batch 2, seed 100004,
size-matched 1,766,987 trainable params each) **trained to completion** —
checkpoints and tokenizer sidecars were written for both — but the
`evaluate_model --ship-gates` stage for this cycle's `smoke,held_out` suite
pair exceeded its cycle stage wall-time budget before either arm produced a
complete `scoreboard.json`.

## What happened

This cycle's role is `promotion` (not `screening` like c1-c3), which widens
the eval suite set from `smoke` alone to `smoke,held_out`. That doubles the
evaluated record count and decode work per arm. The evaluation stage for
both `control` and `component-edge` hit
`stage exceeded wall-time limit` (`evaluate_model --test-dir
e938_role_safe_all_targets_v2 ... --suites smoke,held_out
--decode-timeout-seconds 24.0`), so neither arm has an honest scoreboard.
Per this repository's evaluation-stage-recovery rule, a missing
`scoreboard.json` means the run is not a completed measurement — the
decode-progress telemetry captured before the timeout is diagnostic-only
and is explicitly **not treated as model evidence**:

| Arm | smoke.latency p50 (ms, partial) | smoke.structural_similarity (partial) | smoke.meaningful_program_rate (partial) | smoke.binder_reference_f1 (partial) | held_out |
| --- | ---: | ---: | ---: | ---: | --- |
| control | 8869.91 | 0.4167 | 0.3333 | 0.9524 | timed out, no scoreboard |
| candidate (component-edge) | 8931.45 | 0.4167 | 0.3333 | 0.9524 | timed out, no scoreboard |

The partial `smoke` numbers are an exact tie between arms (as reported by
telemetry, not the ship-gate scoreboard) and `held_out` never produced a
scoreboard for either arm at all.

## Why this is non-positive (and not "blocked")

Per `autotrain-iteration-delivery.md`'s positive-result gate: no primary
metric win is possible without a complete measurement (`held_out
.structural_similarity` shows `improvement=0.0` only because the driver
fell back to the tied partial smoke number, not a real held_out result), no
ship-quality win (no scoreboard at all), and no executable-unblocking (both
arms already ran to completion pre-fix; nothing was broken-then-fixed). The
driver's own classifier reports `positive=false`,
`stack_action=no_stack_layer_non_positive`, `measurement_complete=false`.
Per repository law ("A timed out, interrupted, or killed run is never
evidence"), this cycle is **not** treated as a model result in either
direction — it is an infrastructure/measurement-incomplete outcome. No new
stack layer is opened; this cycle stays local commits + docs only.

## Self-heal action taken

Per the continuous loop's `retry_measurement` handling (frozen replay
resumes at evaluation, reusing the existing checkpoints rather than
retraining), the driver queued action index 0,
`retry_measurement`, on this campaign's frozen manifest
(`3c927a29572f8650f17092112aa129b7159602a11e6f421c8dc7ae5919e1de8e`). This
document (action index 1) is being acknowledged so the supervised driver can
consume the queued retry against the identical frozen arm before any new
model hypothesis is considered, per this loop's rules ("the exact
`retry_measurement` remains queued behind [the document] prerequisite").

## Checkpoints

Both arms' scratch checkpoints
(`runs/c20260802-continuous-openui-p33lme-489d3aa7-c4-{control,component-edge}/checkpoints/last.pt`)
are local-only (`outputs/autoresearch/.../runs/`, explicit no-sync) and are
**unproven** — training completed but evaluation did not. They are **never
reused, promoted, synced, or shipped** until a complete frozen-replay
measurement exists. Recorded in `docs/MODEL_CARD.md` and the README
model-card summary per the model-card duty
(`checkpoint_documentation_required=true` in this cycle's handoff).

## Replay attempt: a real harness bug found and fixed, one budget limit remains

Consuming the queued `retry_measurement` action surfaced a genuine,
reproducible bug in the driver itself, not the model or eval harness:
`_apply_frozen_replay` derived the arm slug from the frozen candidate's
experiment id via `old_candidate_id.rsplit("-", 1)[-1]` — splitting on only
the *last* hyphen. Every multi-word arm in `_SCREENING_ARM_BANK`
(`component-plan`, `component-edge`, `component-inventory`,
`binder-topology`, `component-structure`) contains a hyphen itself, so
`...-c4-component-edge` truncated to `edge` and the automatic frozen-replay
path failed closed immediately with `RuntimeError: unsupported automatic
frozen replay arm: edge` — the queued retry could not even attempt
evaluation. Fixed in commit `359b01c7d85eeaf3961d96d0d89cbf9f7731b907`
(`scripts/run_autotrain_continuous.py`, `harness.autoresearch
.experiment_campaign` v59→v60) by matching the full candidate id against
known arm-bank suffixes (longest first) instead of blindly splitting on the
last hyphen. Regression test
`test_apply_frozen_replay_supports_hyphenated_arm_slugs` reproduces the
exact original error on the pre-fix code and passes after the fix; the
full `test_run_autotrain_continuous.py` suite (97 cases) still passes.

Replaying the identical frozen arm after the fix (successor campaign
`continuous-loop-20260802-continuous-openui-p33lme-489d3aa7-c6`, predecessor
chain c4→c5→c6, reusing both c4 checkpoints via `--reuse-train-run` /
`FROZEN_TRAIN_REUSE`, no retraining) confirms the fix works — the driver no
longer crashes and correctly resumes at evaluation. However, both arms'
`evaluate_model --ship-gates` stage hit the **same wall-time budget**
(`max_wall_minutes=1.1666666666666667`, i.e. ~70s per arm) again and still
produced no `scoreboard.json`. This is a **separate, distinct** issue from
the arm-slug bug: the `smoke,held_out` two-suite decode workload for this
`promotion`-role cycle (8 records total at up to 24s decode timeout each) is
inherently too large for the ~70s per-arm stage budget the continuous driver
currently allocates, independent of whether training is skipped. Per
repository law, a timed-out run is never evidence, so this remains
**measurement-incomplete / non-positive** even after the fix — the arm-slug
bug is real and worth fixing on its own merits (and unblocks the
`retry_measurement` mechanism generally for any future hyphenated-slug arm),
but it does not, by itself, make this cycle's model comparison positive.

This exhausts one of `measurement.max_consecutive_frozen_replays`
(`retry_measurement (1/2)` per c6's handoff); a further identical retry is
still available automatically but is expected to hit the same per-arm
budget deterministically. Flagged for a future session as a distinct
infra follow-up: either raise the per-arm evaluation wall-minutes budget
for `promotion`-role two-suite cycles, or split `smoke`/`held_out` into
separately budgeted stages, without weakening `MAX_RUN_MINUTES` or any ship
gate.

## Next step

Rank-1 `NextRunPriorityV1` (per c6's handoff): replay the exact frozen `c4`
control and `component-edge` arms at the evaluation stage again, or route
the wall-timeout pattern through `improve-openui-harnesses` as a typed
`HarnessSignalV1` (family `autoresearch`) for a budget/staging fix, before
testing a new hypothesis. This is queued as the immediate successor action
for the next supervised invocation of this loop, not a new c5 model
hypothesis — c4's model hypothesis (`component-edge`) itself remains
unexecuted evidence-wise and is still the next thing to measure once the
budget issue is addressed.

Machine-readable evidence is in
[`autotrain-cycle-p33lme-c4-component-edge-screen.json`](autotrain-cycle-p33lme-c4-component-edge-screen.json).
