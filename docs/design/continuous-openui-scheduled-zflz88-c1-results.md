# Continuous autotrain: 2026-08-04 (session zflz88, scheduled) cycle 1 — null delta on knob-rotation arm (screening)

**Loop:** `continuous-openui-scheduled-zflz88`
**Campaign:** `continuous-loop-20260804-continuous-openui-schedu-486913c8-c1`
**Integration commit:** `eba6db30` (`origin/main` tip at cycle start)

**Verdict:** the size-matched `bounds` (knob-rotation) arm ties its control
exactly on the declared primary at this seed — a null delta, not positive.

| Arm | Seed | structural_similarity | parse_rate | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .05750 | 1.0 | .63333 | 4248.82 |
| bounds | 100001 | .05750 | 1.0 | .63333 | 3826.30 |

Primary delta `0.0` — `meaningful_program_rate` stays 0 on both arms. Ship
gates fail as expected: `fixture_insufficient_n` (n=3, need 20).

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). Per `sdlc`
autotrain-iteration-delivery, no new stack layer opens for this cycle; docs
land and the loop continues into cycle 2 using the driver's ranked
successor priority — the `component-plan` hypothesis, which has
independently reproduced a structural_similarity win across multiple prior
sessions.

## Next priorities

1. Screen the `component-plan` hypothesis next (rank 1, confidence 0.9) —
   distinct from the exhausted null `bounds` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
3. Rotate thrash recommendation across the lever bank, not bounds-only
   (rank 3, confidence 0.65).

Machine evidence:
[`continuous-openui-scheduled-zflz88-c1-results.json`](continuous-openui-scheduled-zflz88-c1-results.json).
