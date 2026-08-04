# Autotrain cycle 3 (continuous-openui-local, 2026-08-04) — component-plan screen, regression

New hypothesis per cycle 2's rank-1 priority: a distinct size-matched
`component-plan` quality arm against a freshly matched control (both
1.7558M trainable params — size-matched to each other, not to the cycle
1/2 arms).

| Arm | parse | meaningful | structure | binder F1 | p50 latency | gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Control | 1.000 | 0.333 | 0.2308 | 0.7333 | 10896.6 ms | fail |
| Component-plan candidate | 1.000 | 0.000 | 0.1725 | 0.6333 | 14912.8 ms | fail |

Recipe: CPU scratch TwoTower, 20 steps, strict grammar-constrained
compiler-tree decode, smoke `n=3`; held-out, adversarial, OOD, and
`rico_held` suites were not run. AgentV completed with 0 execution errors.

This control is meaningfully stronger than the cycle 1/2 arms
(`meaningful_program_rate` 0.333 vs 0.0, `structural_similarity` 0.231 vs
0.058) — a reminder that these are independent fixture-scale screens, not a
single continuous baseline. Against **this** control, `component-plan`
regresses: `structural_similarity` -0.058 absolute, `binder_reference_f1`
0.733 -> 0.633, `meaningful_program_rate` 0.333 -> 0.0, and it is slower.
Ship gates reject on both `insufficient_n` (n=3 < 20) and the quality
regression. Per `sdlc` autotrain-iteration-delivery this is **non-positive**
(regression, not a null) — local commit and docs only, no stacked PR.

Next priority (rank 1, `observed_result` confidence 0.90): the distinct
size-matched `component-edge` quality arm
(`c20260804-continuous-openui-local-8c0b60dd-c3-component-edge`), retaining
this cycle's matched control.

JSON twin: [autotrain-cycle-20260804-c3-component-plan-screen.json](autotrain-cycle-20260804-c3-component-plan-screen.json)
