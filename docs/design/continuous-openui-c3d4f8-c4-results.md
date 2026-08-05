# Continuous autotrain: 2026-08-05 (loop `continuous-openui-c3d4f8`) cycle 4 — measurement incomplete (wall timeout)

**Loop:** `continuous-openui-c3d4f8`
**Campaign:** `continuous-loop-20260805-continuous-openui-c3d4f8-986c6dc3-c4`
**Integration commit:** `0050ebf2` (`origin/main` tip at cycle start still `34111e6e`; local-only harness repair + docs from cycle 3 ahead of upstream)

## What happened

Cycle 4 preregistered the rank-1 successor priority from cycle 3 — a fresh
size-matched `component-plan` candidate (1,755,764 params) against a fresh
control — but the campaign's `MAX_RUN_MINUTES` (3-minute) wall cap was
exhausted by the control arm's train+eval alone; the candidate never started
(no scoreboard). This is an expected soft timeout on CPU-only fixture-scale
hardware (`continuous.md`: "wall timeouts... never stop the loop"), not a
harness defect.

| Arm | Status | structural_similarity | parse_rate | binder_reference_f1 | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| control | completed, gates rejected | .41667 | 1.0 | .95238 | 32,592.7 |
| component-plan | **not executed** (wall exhausted) | — | — | — | — |

Control's own ship gates fail as expected on fixture scale (`insufficient_n`
n=3 vs 20, `meaningful_program_rate` .333 < .66, `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate` all below threshold).

## SDLC Phase A

**Non-positive** (`measurement_incomplete` + `primary_metric_unavailable` —
no candidate to compare). No stack layer. The driver queued a typed
`retry_measurement` action bound to the frozen manifest
(`frozen_manifest_sha256 1eabc67edc5bb64d8effce6cc662e79384dc66e108139bf345e60dbb51caed05`).
Per `continuous.md`, this must be consumed — replayed to completion — before
any new model hypothesis; the loop's next supervised invocation auto-detects
and replays the identical frozen recipe rather than starting a new arm.

## Next priorities

1. Replay the exact frozen control + `component-plan` recipe
   (`retry_measurement`) before any new hypothesis (rank 1, confidence 0.95).

Machine evidence:
[`continuous-openui-c3d4f8-c4-results.json`](continuous-openui-c3d4f8-c4-results.json).
