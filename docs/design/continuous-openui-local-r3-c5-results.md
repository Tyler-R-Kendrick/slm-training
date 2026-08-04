# Continuous autotrain: 2026-08-04 (loop `continuous-openui-local-r3`) cycle 5 — frozen-arm replay, complete but non-positive

**Loop:** `continuous-openui-local-r3`
**Campaign:** `continuous-loop-20260804-continuous-openui-local--c8650581-c5`
**Integration commit:** `1825fb54` (`origin/main` tip `34111e6e`)

**Verdict:** non-positive. This is the identical frozen arm from
[cycle 4](continuous-openui-local-r3-c4-results.md)
(`frozen_manifest_sha256` `ec197e91...651b4`), replayed after the cycle-4
`repair_harness` action was acknowledged. Measurement is now complete and
honest for both arms.

## Results

| Arm | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- |
| control | 0.3333 | 0.41667 | 0.95238 | 34049.7 | fail (gate reject) |
| steps | 0.3333 | 0.51 | 0.82222 | 19160.7 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **+0.0933**
(control 0.41667 → steps 0.51). Both arms fail ship gates on
`insufficient_n` (actual=3, need>=20) plus the usual smoke-scale quality
thresholds (`meaningful_program_rate`, `ast_beq_rate`, `canonical_beq_rate`)
— expected at fixture scale, not a promotable result.

## Why this is non-positive despite a primary-metric improvement

`structural_similarity` moved in the beneficial direction, but
`binder_reference_f1` regressed 0.95238 → 0.82222 in the same arm — a mixed
outcome, not a clean quality win. Per the quality-aware tradeoff classifier
(`SDLC_PHASE_A`), a lone primary-metric delta on fixture `n=3` with a
same-arm quality regression elsewhere does not clear the positive-result
gate (`fixture_insufficient_n_alone`, `non_regression_fail:binder_reference_f1`).

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` despite the raw improvement number, because
volume/regression gates dominate). No stacked PR layer; docs + local commit
only.

## Next priorities

1. (rank 1, confidence 0.90, model) The completed frozen replay rejects the
   prior arm; test the distinct size-matched `component-plan` quality
   hypothesis next
   (`c20260804-continuous-openui-local--c8650581-c5-component-plan`).
2. (rank 2, confidence 0.70, evaluation) Keep the matched control as the
   size-matched baseline every cycle.

## Session summary (cycles 1–5, loop `continuous-openui-local-r3`)

| Cycle | Focus | Outcome |
| --- | --- | --- |
| 1–3 | schema/allowlist harness bugs blocked hypothesize/validate | fixed by commit (`harness.autoresearch.experiment_campaign` v180→v182); no experiment ran |
| 4 | first experiment attempt (control vs. steps) | blocked by a stale-`node_modules` harness bug mid-eval; repaired locally, no tracked change; measurement incomplete on both arms |
| 5 | frozen replay of cycle 4's arm | complete, honest scoreboards for both arms; non-positive (`fixture_insufficient_n_alone`, mixed metric outcome) |

No cycle in this session produced a stacked PR: cycles 1–4 never reached a
completed model measurement, and cycle 5's completed measurement is
fixture-scale (`n=3`) screening evidence that does not clear the honest ship
gates. Per SDLC Phase A, all work this session stays as local, incremental
commits with docs.
