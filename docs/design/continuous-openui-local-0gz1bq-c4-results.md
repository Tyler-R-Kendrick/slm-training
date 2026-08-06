# Continuous autotrain: 2026-08-05 (scheduled loop `0gz1bq`) cycle 4 — component-edge exact tie, slower

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `3dbde7cc` (`origin/main` tip `bdf143cd` + cycles 1-3 docs commits)

**Verdict:** non-positive, exact tie. Fresh test of the `component-edge`
hypothesis (rank-1 priority from [cycle 3](continuous-openui-local-0gz1bq-c3-results.md)).

## Results

| Arm | meaningful_program_rate | structural_similarity | binder_reference_f1 | component_type_recall | placeholder_fidelity | reward_score | latency p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 0.33333 | 0.41667 | 0.95238 | 0.25 | 0.91667 | 0.93600 | 27697.6 |
| component-edge | 0.33333 | 0.41667 | 0.95238 | 0.25 | 0.91667 | 0.93600 | 32093.2 |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — every
quality metric is byte-identical between arms. `component-edge` is markedly
slower (+15.9% p50, 32093.17ms vs 27697.6ms) with zero quality delta: a pure
cost regression at this fixture scale, no lever effect.

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` with `improvement=0.0`). No stacked PR layer;
docs + local commit only.

## Next priorities

1. (rank 1, confidence 0.90) `component-edge` shows no lever effect at this
   scale; cycle back to a fresh-seed `component-plan` re-test or rotate to a
   distinct lever from the thrash bank next
   (`c20260805-continuous-openui-local-8c0b60dd-c4-component-plan`).

## Session summary (cycles 1-4, loop `continuous-openui-local`, session `0gz1bq`)

| Cycle | Experiment focus | Primary metric delta | Outcome |
| --- | --- | --- | --- |
| 1 | grammar-completion-bounds vs control | infra failure (torch missing) | self-healed, no evidence |
| 2 | grammar-completion-bounds vs control (frozen replay) | 0.0575 → 0.0575 (Δ0.0) | non-positive, exact tie (4th confirmation) |
| 3 | component-plan vs control (fresh seed) | 0.23083 → 0.1725 (Δ-0.05833) | non-positive, regression (5th measurement, sign disagrees with prior sessions) |
| 4 | component-edge vs control | 0.41667 → 0.41667 (Δ0.0) | non-positive, exact tie, slower |

No cycle in this session produced a stacked PR: all four cycles are
fixture (`n=3`) screening evidence only, none clears the honest ship gates.
Per SDLC Phase A, this session's work stays as local commits with docs.
