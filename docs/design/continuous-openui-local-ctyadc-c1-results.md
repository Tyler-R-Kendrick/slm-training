# Continuous autotrain: 2026-08-03 (session ctyadc, scheduled) cycle 1 — null delta on knob-rotation arm (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c1`
**Integration commit:** `45d78cb4` (`origin/main` tip at cycle start)

**Verdict:** the size-matched `bounds` (knob-rotation) arm ties its control
exactly on the declared primary at this seed — a null delta, not positive.
This is a **scheduled-routine** invocation of the same continuous loop that
sessions `ts5ofk` and `j48f8u` ran earlier; the local `outputs/` state resets
per session, so cycle 1 reruns the same screening arm and reproduces the
same result.

| Arm | Seed | structural_similarity | parse_rate | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .05750 | 1.0 | .63333 | 3145.09 |
| bounds | 100001 | .05750 | 1.0 | .63333 | 3186.42 |

Primary delta `0.0` — `meaningful_program_rate` stays 0 on both arms. Ship
gates fail as expected: `fixture_insufficient_n` (n=3, need 20); `held_out`,
`adversarial`, `ood`, `rico_held` suites missing at smoke scale.

## Third reproduction of the null bounds delta

Same outcome as
[`continuous-openui-local-ts5ofk-c1-results.md`](continuous-openui-local-ts5ofk-c1-results.md)
and the earlier `j48f8u` session — the `bounds` knob-rotation arm is now
exhausted as a screening candidate. Per the driver's own ranked successor
priority, cycle 2 of this loop should test the `component-plan` hypothesis
instead of repeating `bounds`.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). Per `sdlc`
autotrain-iteration-delivery, no new stack layer opens for this cycle; docs
land and the loop continues into cycle 2 using the driver's ranked successor
priority (`component-plan`, `proposed_experiment_id`
`c20260803-continuous-openui-local-8c0b60dd-c1-component-plan`).

## Next priorities

1. Run the driver's ranked successor experiment next (`component-plan`,
   rank 1, confidence 0.9) instead of re-running the exhausted `bounds` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
3. Do not re-select the completed non-positive `bounds` candidate again
   without a new preregistered hypothesis (rank 3, confidence 0.65).

Machine evidence:
[`continuous-openui-local-ctyadc-c1-results.json`](continuous-openui-local-ctyadc-c1-results.json).
