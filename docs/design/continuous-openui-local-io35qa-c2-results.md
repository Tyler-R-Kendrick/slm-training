# Continuous autotrain: 2026-08-04 (session io35qa) cycle 2 — frozen-replay executable unblock proven, non-positive fixture

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `d306a026` (this session's cycle-1 diagnosis-docs commit,
on top of `main` tip `f4949582`)

**Verdict:** the identical frozen arm pair that failed in cycle 1
(`ModuleNotFoundError: No module named 'torch'`) now **completes end to end**
after the environment fix documented in
[`continuous-openui-local-io35qa-c1-results.md`](continuous-openui-local-io35qa-c1-results.md) —
this is the replay proof for that cycle's `repair_harness` action.

| Arm | Seed | structural_similarity | binder_reference_f1 | parse_rate | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100001 | .0575 | .63333 | 1.0 | 3715.94 |
| bounds (candidate) | 100001 | .0575 | .63333 | 1.0 | 4536.84 |

Primary metric (`smoke.structural_similarity`) improvement `0.0` (tie) —
`bounds` does not beat `control` at this seed.

Ship gates fail as expected on fixture scale: `insufficient_n` (n=3, need 20),
plus the usual smoke-scale quality thresholds
(`meaningful_program_rate`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, `reward_score`). `placeholder_fidelity` and
`certified_fallback` pass. Not a ship or promotion claim.

## SDLC Phase A

**Non-positive** — `primary_metric_null_or_worse` (control == candidate) plus
`fixture_insufficient_n` on both arms. Per `sdlc` autotrain-iteration-delivery,
"it ran" without a metric win or unblocking evidence beyond what cycle 1
already earned is not itself a fresh positive; no new stack layer for this
cycle's own reasons list.

However, this cycle **is** the receipt for cycle 1's `repair_harness` action
(env-provisioning fix, no code defect) — the identical frozen arm ran and
produced a real, honest scoreboard instead of an infrastructure crash. That
proof lives in cycle 1's doc, cross-linked above; this cycle's own record
stays non-positive per the fixture/null-delta gate.

## Next priorities (from the driver)

1. Rank 1 (confidence 0.90): the completed frozen replay rejects the prior
   `bounds` arm at this seed; test the distinct size-matched `component-plan`
   quality hypothesis next (`c20260804-continuous-openui-local-8c0b60dd-c2-component-plan`).
2. Keep the matched control as the size-matched baseline every cycle.
3. Soft ship-gate fails on fixture `n` never stop the continuous loop.

Machine evidence:
[`continuous-openui-local-io35qa-c2-results.json`](continuous-openui-local-io35qa-c2-results.json).
