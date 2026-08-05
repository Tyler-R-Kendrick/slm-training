# Continuous autotrain: 2026-08-05 (loop `continuous-openui-c3d4f8`) cycle 5 — frozen retry_measurement, decode timeout (soft, resolved)

**Loop:** `continuous-openui-c3d4f8`
**Campaign:** `continuous-loop-20260805-continuous-openui-c3d4f8-986c6dc3-c5`
**Integration commit:** `a08ffbc3`

## What happened

This cycle auto-consumed cycle 4's queued `retry_measurement`: it reused the
frozen `component-plan` train checkpoint (`FROZEN_TRAIN_REUSE`) rather than
retraining, and re-ran evaluation only. The control repeated cleanly
(structure `.41667`, matching cycle 4). The `component-plan` candidate's
evaluation hit `decode_timeout` on all 3 smoke documents inside the wall
cap — a soft CPU decode-cost timeout, the same class `continuous.md`
documents (per-record compiler+decode cost outrunning the nominal per-record
wall share even with the JS grammar bridges installed), not a harness
defect.

Crucially, the driver's disposition here is **terminal**
(`candidate_runtime_rejected_after_frozen_replay`), not another open retry —
it closes this measurement attempt and ranks a new, distinct hypothesis
(`component-edge`) for the next cycle instead of looping the same retry.

## SDLC Phase A

**Non-positive** (`measurement_incomplete` + `fixture_insufficient_n_alone`).
No stack layer.

## Next priorities

1. Screen the `component-edge` hypothesis next (rank 1, confidence 0.9).
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).

Machine evidence:
[`continuous-openui-c3d4f8-c5-results.json`](continuous-openui-c3d4f8-c5-results.json).
