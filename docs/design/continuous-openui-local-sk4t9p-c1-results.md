# Continuous autotrain: 2026-08-03 (scheduled session sk4t9p) cycle 1 — null delta on knob-rotation arm (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `089b1649` (`origin/main` tip at cycle start)

**Verdict:** the size-matched `bounds` (`grammar_completion_bounds=true`) arm
ties its control exactly on the declared primary at this seed — a null delta,
not positive.

| Arm | Seed | structural_similarity | parse_rate | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .05750 | 1.0 | .63333 | 1890.62 |
| bounds | 100001 | .05750 | 1.0 | .63333 | 1533.82 |

Primary delta `0.0` — `meaningful_program_rate` stays 0 on both arms. The
`bounds` arm's lower p50 latency does not count as a win: the primary metric
is flat and `meaningful_program_rate` is zero on both arms, so this is not a
quality-aware latency tradeoff under the SDLC Phase A classifier. Ship gates
fail as expected: `fixture_insufficient_n` (n=3, need 20); `held_out`,
`adversarial`, `ood`, `rico_held` are all `missing_suite` at smoke scale.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). Per `sdlc`
autotrain-iteration-delivery, no new stack layer opens for this cycle; docs
land and the loop continues into cycle 2 using the driver's ranked successor
priority (the `component-plan` hypothesis, which has independently
reproduced a `+0.056` structural_similarity win across four prior sessions:
PR #1369, PR #1376, PR #1378,
[`continuous-openui-local-ts5ofk-c2-results.md`](continuous-openui-local-ts5ofk-c2-results.md)).

## Next priorities

1. Screen the `component-plan` hypothesis next (rank 1, confidence 0.9) —
   distinct from the exhausted null `bounds` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
3. Rotate the thrash recommendation across the lever bank rather than
   re-running the exhausted `bounds` arm (rank 3, confidence 0.65).

Machine evidence:
[`continuous-openui-local-sk4t9p-c1-results.json`](continuous-openui-local-sk4t9p-c1-results.json).
