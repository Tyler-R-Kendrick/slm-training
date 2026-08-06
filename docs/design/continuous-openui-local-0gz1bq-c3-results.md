# Continuous autotrain: 2026-08-05 (scheduled loop `0gz1bq`) cycle 3 — component-plan regresses (5th measurement)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `9b79c04a` (`origin/main` tip `bdf143cd` + cycle 1/2 docs commits)

**Verdict:** non-positive, regression. Fresh-seed test of the `component-plan`
hypothesis (previously a positive fixture win in session `j48f8u` cycle 2,
then an exact-tie non-reproduction in session `peuum8` cycle 3).

## Results

| Arm | meaningful_program_rate | structural_similarity | binder_reference_f1 | component_type_recall | placeholder_fidelity | reward_score | latency p50 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 0.33333 | 0.23083 | 0.73333 | 0.16667 | 0.63889 | 0.84067 | 9744.79 |
| component-plan | 0.0 | 0.1725 | 0.63333 | 0.0 | 0.52778 | 0.80733 | 8818.61 |

Primary metric (`smoke.structural_similarity`) improvement: **-0.05833**
(control beats candidate). `binder_reference_f1` also regresses
(`0.7333 -> 0.6333`), and both `meaningful_program_rate` and
`component_type_recall` drop to zero on the candidate. `component-plan` is
faster (-926ms p50) but that is not a valid tradeoff here: quality itself
regresses, not just latency vs a held quality floor.

## Cross-session history of this hypothesis

| Session | Cycle | Primary delta | Outcome |
| --- | --- | --- | --- |
| `j48f8u` | 2 | `.32667 → .38280` (**+0.05613**) | positive, champion queued |
| `peuum8` | 3 (fresh-seed confirmation) | `.23083 → .23083` (0.0) | exact tie, non-reproduction |
| `0gz1bq` | 3 (this cycle, fresh seed) | `.23083 → .1725` (**-0.05833**) | regression |

Three independent sessions now disagree on this hypothesis's sign: one win,
one tie, one regression. Treat `component-plan` as an unreliable/exhausted
lever at this fixture scale rather than a confirmed positive.

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse` with `improvement=-0.05833`,
`non_regression_fail:binder_reference_f1`). No stacked PR layer; docs + local
commit only.

## Next priorities

1. (rank 1, confidence 0.90) `component-plan` is exhausted (3-session
   disagreement, net regressing on fresh seed); test the distinct
   size-matched `component-edge` quality hypothesis next
   (`c20260805-continuous-openui-local-8c0b60dd-c3-component-edge`).
