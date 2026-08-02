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

## Next step

Rank-1 `NextRunPriorityV1`: replay the exact frozen `c4` control and
`component-edge` arms at the evaluation stage (resume-at-evaluation, not
retrain) before testing a new hypothesis. This is queued as the immediate
successor action for the next supervised invocation of this loop, not a new
c5 model hypothesis.

Machine-readable evidence is in
[`autotrain-cycle-p33lme-c4-component-edge-screen.json`](autotrain-cycle-p33lme-c4-component-edge-screen.json).
