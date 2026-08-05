# Continuous autotrain: 2026-08-05 (scheduled loop `a08cs6`) cycle 5 — component-edge exact tie

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c5`
**Integration commit:** `5ff54717` (`origin/main` tip `bdf143cd`, i.e. after PR #1444 merged)
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c4`

**Verdict:** non-positive. Ran the `component-edge` arm flagged as priority
rank 1 from [cycle 4](continuous-openui-local-a08cs6-c4-results.md).

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.0 | 0.04307 | 0.82222 | 15454.79 | fail (gate reject) |
| component-edge | 1.0 | 0.0 | 0.04307 | 0.82222 | 15915.40 | fail (gate reject) |

Primary metric improvement: **0.0** — exact tie. This seed's control baseline
(0.04307) is much weaker than cycle 4's control (0.41667) — fixture n=3 seed
variance, not a regression, since both arms degraded identically.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` with `improvement=0.0`). No stacked PR layer;
local commit only.

## Next priorities

1. (rank 1) Test the distinct size-matched `component-plan` quality
   hypothesis next (`c20260805-continuous-openui-local-8c0b60dd-c5-component-plan`).
2. (rank 2) Keep the matched control as the size-matched baseline every cycle.
