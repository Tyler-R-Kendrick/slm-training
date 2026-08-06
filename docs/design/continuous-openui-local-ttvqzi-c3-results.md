# Continuous autotrain: 2026-08-05 (scheduled session `ttvqzi`) cycle 3 — component-plan regresses at seed 100001 (opposite of the seed-100002 win)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `7630868f`
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`
(bounds hypothesis closed out as an exact tie — see
[c2 results](continuous-openui-local-ttvqzi-c2-results.md))

**Verdict:** non-positive — a real regression, cleanly measured (both arms
`exit=0`).

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.3333 | 0.2308 | 0.7333 | 20301.3 | fail (gate reject) |
| component-plan | 1.0 | 0.0 | 0.1725 | 0.6333 | 12259.6 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) delta: **-0.0583**
(`0.2308 → 0.1725`), and `meaningful_program_rate` drops to 0, and
`binder_reference_f1` regresses (`0.7333 → 0.6333`) — a genuine three-way
quality regression, not a measurement artifact (`non_regression_fail`
triggered on `binder_reference_f1`).

## This is the opposite of an earlier session's result for the same hypothesis identity

Session `j48f8u` (an earlier scheduled loop) measured `component-plan` beating
control by **+0.05613** structural similarity at **seed 100002**, and that
same win reproduced three independent times across sessions (PR #1369,
[`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)).
This cycle used **seed 100001** (this session's default) and gets a clean,
opposite result: `component-plan` regresses on every quality metric. Both
measurements are honest completed runs (`exit=0`, no timeouts, no harness
signal). Taken together this is evidence that the `component-plan`
hypothesis's effect is **seed-sensitive**, not a uniform win — the seed-100002
reproduction and this seed-100001 regression cannot both generalize to "always
positive."

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`, `non_regression_fail`,
`fixture_insufficient_n_alone`). No stacked PR layer.

## Next priorities

1. (rank 1, confidence 0.90) Test the distinct size-matched `component-edge`
   quality hypothesis next
   (`c20260805-continuous-openui-local-8c0b60dd-c3-component-edge`).
2. A dedicated seed-sweep confirmation of `component-plan` (100001 vs 100002,
   matched steps/params) would resolve whether this is genuine seed
   sensitivity or a confound in one of the two measurement lineages — flagged
   as a candidate for a future cycle, not attempted here.

Machine evidence:
[`continuous-openui-local-ttvqzi-c3-results.json`](continuous-openui-local-ttvqzi-c3-results.json).
