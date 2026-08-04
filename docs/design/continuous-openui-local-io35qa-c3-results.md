# Continuous autotrain: 2026-08-04 (session io35qa) cycle 3 — component-plan regresses at this seed, non-positive

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `b464c7ae` (this session's cycle-2 docs commit merged
onto `origin/main` tip `eba6db30`)

**Verdict:** `component-plan` **regresses** the size-matched control on the
declared primary at this seed — the opposite of the four independent
byte-identical reproductions of a `component-plan` win recorded in prior
sessions (see
[`continuous-openui-local-ts5ofk-c2-results.md`](continuous-openui-local-ts5ofk-c2-results.md)).
This is a **different** configuration (fresh seed / control state after
several intervening harness commits), not a contradiction of the earlier
result — flagging it explicitly rather than silently discarding it.

| Arm | structural_similarity | binder_reference_f1 | meaningful_program_rate | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| control | .23083 | .73333 | .33333 | 9487.45 |
| component-plan (candidate) | .17250 | .63333 | 0 | 9145.70 |

Primary delta `-.05833` (worse); `binder_reference_f1` also regresses
(`non_regression_fail`). Ship gates fail as expected on fixture scale
(`insufficient_n`, n=3).

## SDLC Phase A

**Non-positive** — `primary_metric_null_or_worse` (candidate worse than
control) and `fixture_insufficient_n_alone`. No stack layer.

## Next priorities (from the driver)

1. Rank 1 (confidence 0.90): this arm is exhausted; test the distinct
   size-matched `component-edge` quality hypothesis next.
2. Keep the matched control as the size-matched baseline every cycle.
3. Soft ship-gate fails on fixture `n` never stop the continuous loop.

Machine evidence:
[`continuous-openui-local-io35qa-c3-results.json`](continuous-openui-local-io35qa-c3-results.json).
