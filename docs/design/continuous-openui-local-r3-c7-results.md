# Continuous autotrain: 2026-08-04 (loop `continuous-openui-local-r3`) cycle 7 — component-edge, exact tie

**Loop:** `continuous-openui-local-r3`
**Campaign:** `continuous-loop-20260804-continuous-openui-local--c8650581-c7`
**Integration commit:** `92b44294` (`origin/main` tip `34111e6e`)

**Verdict:** non-positive. Ran the size-matched `component-edge` hypothesis
flagged as [cycle 6](continuous-openui-local-r3-c6-results.md) priority rank
1, against a fresh matched control.

## Results

| Arm | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- |
| control | 0.0 | 0.0575 | 0.63333 | 5184.0 | fail (gate reject) |
| component_edge | 0.0 | 0.0575 | 0.63333 | 4954.4 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **0.0** — exact
tie on both the primary metric and `binder_reference_f1`. `component_edge`
is not distinguishable from control at this fixture scale (`n=3`).

## SDLC Phase A

**Non-positive** (`fixture_insufficient_n_alone`,
`primary_metric_null_or_worse` with `improvement=0.0`). No stacked PR layer;
docs + local commit only.

## Next priorities

1. (rank 1, confidence 0.90, model) The completed non-positive arm is
   exhausted; test the distinct size-matched `component-plan` quality
   hypothesis next
   (`c20260804-continuous-openui-local--c8650581-c7-component-plan`).
2. (rank 2, confidence 0.70, evaluation) Keep the matched control as the
   size-matched baseline every cycle.

## Session summary (cycles 1–7, loop `continuous-openui-local-r3`)

| Cycle | Focus | Outcome |
| --- | --- | --- |
| 1–3 | schema/allowlist harness bugs blocked hypothesize/validate | fixed by commit (`harness.autoresearch.experiment_campaign` v180→v182); no experiment ran |
| 4 | first experiment attempt (control vs. steps) | blocked by a stale-`node_modules` harness bug mid-eval; repaired locally, no tracked change |
| 5 | frozen replay of cycle 4's arm | complete, honest scoreboards; non-positive (mixed metric outcome) |
| 6 | component-plan vs. control | non-positive, exact tie |
| 7 | component-edge vs. control | non-positive, exact tie |

This session's one positive delivery is the infrastructure unblock
(commits `174b0f1`/`78cf357`, PR #1441): the continuous loop went from
completely non-executable (every screening hypothesis failed schema
validation) to running real, honest, gate-checked screening cycles. No
model-quality lever has produced a distinguishable win yet at fixture scale;
all four completed screening cycles (5–7 model comparisons) are exact ties
or mixed-metric non-wins. Per SDLC Phase A, this stays local commits + docs
on the existing open PR; the loop continues on its next scheduled firing.
