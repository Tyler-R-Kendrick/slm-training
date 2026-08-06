# Continuous autotrain: 2026-08-05 (session 8c0b60dd, scheduled run) cycle 1 — third reproduction of the null delta on the bounds arm (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `bdf143cd` (`origin/main` tip at cycle start)

**Verdict:** the size-matched `bounds` (knob-rotation) arm ties its control
exactly on the declared primary at this seed — a null delta, not positive.
Byte-identical to the two prior sessions that measured this same recipe/seed.

| Arm | Seed | structural_similarity | parse_rate | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .05750 | 1.0 | .63333 | 4142.26 |
| bounds | 100001 | .05750 | 1.0 | .63333 | 3758.47 |

Primary delta `0.0` — `meaningful_program_rate`, `ast_beq_rate`,
`canonical_beq_rate`, and `reward_score` all stay 0 on both arms. Ship gates
fail as expected: `fixture_insufficient_n` (n=3, need 20) plus the missing
`held_out`/`adversarial`/`ood`/`rico_held` suites.

## Third reproduction of the prior sessions' null result

Same outcome as
[`continuous-openui-local-gd6j83-c1-results.md`](continuous-openui-local-gd6j83-c1-results.md)
and `continuous-openui-local-ts5ofk-c1-results.md`: the `bounds`
knob-rotation arm is exhausted as a screening hypothesis at this recipe. This
session does not re-attempt it a third time; the driver's ranked successor
priority routes cycle 2 to the independently-reproduced `component-plan`
hypothesis instead
([`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
previously measured a `+0.056` structural_similarity win there).

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). Per `sdlc`
autotrain-iteration-delivery, no new stack layer opens for this cycle; docs
land locally and the loop continues into cycle 2.

## Next priorities

1. Screen the `component-plan` hypothesis next (rank 1, confidence 0.9) —
   distinct from the now 3x-exhausted null `bounds` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
3. Rotate thrash recommendation across the lever bank rather than repeating
   `bounds`-only (rank 3, confidence 0.65).

Machine evidence:
[`continuous-openui-local-8c0b60dd-c1-results.json`](continuous-openui-local-8c0b60dd-c1-results.json).
