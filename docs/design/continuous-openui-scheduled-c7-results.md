# Continuous autotrain: 2026-08-05 (scheduled loop `continuous-openui-scheduled`) cycle 7 — `component-edge` hypothesis exact tie, rejected

**Loop:** `continuous-openui-scheduled`
**Campaign:** `continuous-loop-20260805-continuous-openui-schedu-1e62ecf9-c7`
**Integration commit:** `78b5b67b`

**Verdict:** non-positive. Another exact tie between candidate and control on
every measured smoke metric.

| Arm | primary (`smoke.structural_similarity`) | `meaningful_program_rate` | `binder_reference_f1` | `latency_ms_p50` |
| --- | ---: | ---: | ---: | ---: |
| control | 0.0575 | 0.0 | 0.633333 | 3,972.84 |
| component-edge (candidate) | 0.0575 | 0.0 | 0.633333 | 3,857.25 |

`improvement=0.0`.

## Worth watching (not yet escalated)

This is the **second consecutive** exact-tie null in this loop for a
`structural_aux_head_profile`-family hypothesis: c6 tested `component-plan`
and also tied exactly; c7 tests `component-edge` and ties exactly again. Two
data points is suggestive but not a reproduced, single-family, frozen-input
`HarnessSignalV1` yet — it could honestly be that these aux-head knobs have
zero measurable effect at this tiny fixture scale (n=3, steps=20), or there
could be a wiring gap where the knob is accepted but never reaches the
model/training path. **Not escalated this cycle.** Flagged in
`next_priorities` (rank 2, `monitor`): if a third `structural_aux_head_profile`
hypothesis also ties exactly, the next session should open a
`repair_harness` investigation via `improve-openui-harnesses` rather than
continuing to treat each as an independent model null.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse` with `improvement=0.0`,
`fixture_insufficient_n` on both arms). No stack layer; local-commit-only
record.

## Next priorities (ranked, from the driver)

1. Test the distinct size-matched `component-plan` quality hypothesis next
   (`experiment_next`, confidence 0.90).
2. *(This session's addition, not the driver's)* Watch for a third
   consecutive `structural_aux_head_profile`-family exact tie before
   escalating a harness investigation (confidence 0.55, `monitor`).
