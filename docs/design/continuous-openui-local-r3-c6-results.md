# Continuous autotrain: 2026-08-04 (loop `continuous-openui-local-r3`) cycle 6 — component-plan, exact tie

**Loop:** `continuous-openui-local-r3`
**Campaign:** `continuous-loop-20260804-continuous-openui-local--c8650581-c6`
**Integration commit:** `9f24bd3e` (`origin/main` tip `34111e6e`)

**Verdict:** non-positive. Ran the size-matched `component-plan` hypothesis
flagged as [cycle 5](continuous-openui-local-r3-c5-results.md) priority rank
1, against a fresh matched control.

## Results

| Arm | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- |
| control | 0.0 | 0.0964 | 0.82222 | 5671.8 | fail (gate reject) |
| component_plan | 0.0 | 0.0964 | 0.82222 | 5167.7 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie on both the primary metric and `binder_reference_f1`. `component_plan`
is not distinguishable from control at this fixture scale (`n=3`).

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` with `improvement=0.0`). No stacked PR layer;
docs + local commit only.

## Next priorities

1. (rank 1, confidence 0.90, model) The completed non-positive arm is
   exhausted; test the distinct size-matched `component-edge` quality
   hypothesis next
   (`c20260804-continuous-openui-local--c8650581-c6-component-edge`).
2. (rank 2, confidence 0.70, evaluation) Keep the matched control as the
   size-matched baseline every cycle.
