# Continuous autotrain: 2026-08-05 (scheduled loop `0gz1bq`) cycle 2 — frozen replay, exact tie confirmed

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `76304a69` (`origin/main` tip `bdf143cd` + docs commit for cycle 1's infra self-heal)

**Verdict:** non-positive, exact tie. This is the frozen replay of cycle 1's
`control`/`bounds` arms (`grammar-completion-bounds` hypothesis), now that
the missing-torch infrastructure gap from
[cycle 1](continuous-openui-local-0gz1bq-c1-results.md) is repaired.

## Results

| Arm | meaningful_program_rate | structural_similarity | binder_reference_f1 | placeholder_fidelity | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- | --- |
| control | 0.0 | 0.0575 | 0.63333 | 0.52778 | 4164.45 | fail (gate reject, n=3) |
| bounds | 0.0 | 0.0575 | 0.63333 | 0.52778 | 4255.73 | fail (gate reject, n=3) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie. `bounds` is marginally slower (+91ms p50) with no quality delta,
reproducing the same null result found for `grammar-completion-bounds` in
three prior independent sessions (`continuous-openui-local-j48f8u-c1`,
`continuous-openui-local-peuum8-c1`, `autotrain-cycle-c3-bounds-quality-neutral`).

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` with `improvement=0.0`). No stacked PR layer;
docs + local commit only.

## Next priorities

1. (rank 1, confidence 0.90) `grammar-completion-bounds` is exhausted across
   four independent sessions now; test the distinct size-matched
   `component-plan` quality hypothesis next
   (`c20260805-continuous-openui-local-8c0b60dd-c2-component-plan`) — this
   hypothesis previously won `+0.05613` on `structural_similarity`
   (session `j48f8u`, cycle 2) but did not survive fresh-seed confirmation
   in session `peuum8` (cycle 3); this run is a fourth, fresh-seed attempt.
