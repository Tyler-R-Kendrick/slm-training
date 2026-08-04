# Continuous autotrain: 2026-08-04 (session gd6j83) cycle 1 — null delta on knob-rotation arm (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `4e137321` (`origin/main` tip at cycle start)

**Verdict:** the size-matched `bounds` (knob-rotation) arm ties its control
exactly on the declared primary at this seed — a null delta, not positive.

| Arm | Seed | structural_similarity | parse_rate | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .05750 | 1.0 | .63333 | 4319.10 |
| bounds | 100001 | .05750 | 1.0 | .63333 | 4167.49 |

Primary delta `0.0` — `meaningful_program_rate` stays 0 on both arms. Ship
gates fail as expected: `fixture_insufficient_n` (n=3, need 20).

## Repeats the prior session's null result

Same outcome as
[`continuous-openui-local-ts5ofk-c1-results.md`](continuous-openui-local-ts5ofk-c1-results.md):
the `bounds` knob-rotation arm is exhausted as a screening hypothesis at this
recipe. This session does not re-attempt it a third time; the driver's ranked
successor priority routes cycle 2 to the independently-reproduced
`component-plan` hypothesis instead
([`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
previously measured a `+0.056` structural_similarity win there).

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). Per `sdlc`
autotrain-iteration-delivery, no new stack layer opens for this cycle; docs
land locally and the loop continues into cycle 2.

## Next priorities

1. Screen the `component-plan` hypothesis next (rank 1, confidence 0.9) —
   distinct from the exhausted null `bounds` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
3. Rotate thrash recommendation across the lever bank rather than repeating
   `bounds`-only (rank 3, confidence 0.65).

Machine evidence:
[`continuous-openui-local-gd6j83-c1-results.json`](continuous-openui-local-gd6j83-c1-results.json).
