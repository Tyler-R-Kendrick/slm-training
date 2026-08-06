# Continuous autotrain: 2026-08-05 (scheduled session `ttvqzi`) cycle 4 — component-edge exact tie, slower

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `1b613728`
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`
(component-plan regressed at seed 100001 — see
[c3 results](continuous-openui-local-ttvqzi-c3-results.md))

**Verdict:** non-positive — exact quality tie, candidate is slower.

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.3333 | 0.4167 | 0.9524 | 27792.2 | fail (gate reject) |
| component-edge | 1.0 | 0.3333 | 0.4167 | 0.9524 | 30898.2 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie on every smoke quality metric. `component-edge` is **slower**
(+11.2% p50, 30898.2ms vs 27792.2ms) with `1,766,990` trainable params in
both arms (matched capacity). Auxiliary component-edge decode coupling buys
no quality signal at additional latency cost. Both arms fail every
evidence-volume/quality-threshold gate at fixture `n=3`, as expected.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`, `primary_metric_null_or_worse`
with `improvement=0.0`). No stacked PR layer.

## Next priorities

1. (rank 1, confidence 0.90) Test the distinct size-matched `component-plan`
   quality hypothesis next
   (`c20260805-continuous-openui-local-8c0b60dd-c4-component-plan`) — a
   fourth independent measurement of this hypothesis identity, given the
   seed-sensitivity finding from c3.

Machine evidence:
[`continuous-openui-local-ttvqzi-c4-results.json`](continuous-openui-local-ttvqzi-c4-results.json).
