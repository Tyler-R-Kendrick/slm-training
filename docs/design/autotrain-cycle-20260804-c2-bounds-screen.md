# Autotrain cycle 2 (continuous-openui-local, 2026-08-04) — frozen replay, bounds screen

`FROZEN_REPLAY_ACK` of cycle 1's frozen `wf_smoke_v2` control/bounds arms,
run after `node_modules/@agentv/core` became available (see
[`autotrain-cycle-20260804-c1-agentv-missing-infra-failure.md`](autotrain-cycle-20260804-c1-agentv-missing-infra-failure.md)).
Both arms exited normally and completed a full ship-gate scoreboard this
time; this is a real measurement, not an infrastructure failure — and a
fixture result, not a production-learning claim.

| Arm | parse | meaningful | structure | binder F1 | p50 latency | gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Control | 1.000 | 0.000 | 0.0575 | 0.6333 | 4471.63 ms | fail |
| Bounds candidate | 1.000 | 0.000 | 0.0575 | 0.6333 | 7894.27 ms | fail |

Recipe: CPU scratch TwoTower, 20 steps, strict grammar-constrained
compiler-tree decode, smoke `n=3`; held-out, adversarial, OOD, and
`rico_held` suites were not run. AgentV completed with 0 execution errors.
The candidate is a quality null (identical `structural_similarity`,
`meaningful_program_rate`, `binder_reference_f1` to control) and slower, so
it is rejected and neither checkpoint is promotable, reusable, synced, or
ship-eligible.

What prevents a learning claim is capability/evidence, not Lean execution:
meaningful-program rate and exact AST/canonical agreement are zero, and the
smoke sample is below the required `n>=20`. Per
`sdlc` autotrain-iteration-delivery, this is a **non-positive** cycle
(`fixture_insufficient_n` + null primary-metric delta) — local commit and
docs only, no stacked PR.

Next priority (rank 1, `observed_result` confidence 0.90): the distinct
size-matched `component-plan` quality arm
(`c20260804-continuous-openui-local-8c0b60dd-c2-component-plan`), retaining
the unchanged control.

JSON twin: [autotrain-cycle-20260804-c2-bounds-screen.json](autotrain-cycle-20260804-c2-bounds-screen.json)
