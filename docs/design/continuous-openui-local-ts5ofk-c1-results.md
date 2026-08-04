# Continuous autotrain: 2026-08-03 (session ts5ofk) cycle 1 — null delta on knob-rotation arm (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `d5b0c318` (`origin/main` tip at cycle start)

**Verdict:** the size-matched `bounds` (knob-rotation) arm ties its control
exactly on the declared primary at this seed — a null delta, not positive.

| Arm | Seed | structural_similarity | parse_rate | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .05750 | 1.0 | .63333 | 1142.13 |
| bounds | 100001 | .05750 | 1.0 | .63333 | 1119.14 |

Primary delta `0.0` — `meaningful_program_rate` stays 0 on both arms. Ship
gates fail as expected: `fixture_insufficient_n` (n=3, need 20).

## This is not a confirmation attempt

Same as the prior session's cycle 2
([`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)),
this cycle does not attempt the blocked `-confirm` fresh-seed path or the
seed-`100005` dual-arm decode timeout — both remain routed to a dedicated
`improve-openui-harnesses` session per
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md).

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). Per `sdlc`
autotrain-iteration-delivery, no new stack layer opens for this cycle; docs
land and the loop continues into cycle 2 using the driver's ranked
successor priority (the `component-plan` hypothesis, which has independently
reproduced a `+0.056` structural_similarity win across three prior sessions).

## Next priorities

1. Screen the `component-plan` hypothesis next (rank 1, confidence 0.9) —
   distinct from the exhausted null `bounds` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
3. Do not speculatively re-attempt the blocked `-confirm` slug bug or
   seed-`100005` decode timeout; that stays a dedicated harness-repair task.

Machine evidence:
[`continuous-openui-local-ts5ofk-c1-results.json`](continuous-openui-local-ts5ofk-c1-results.json).
