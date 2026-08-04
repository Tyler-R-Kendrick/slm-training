# Continuous autotrain: 2026-08-04 (session 8c0b60dd) cycle 3 — component-plan regresses structural similarity

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `8ebf6fcc` (`origin/main` tip after merging PR #1423's harness repair)

**Verdict:** the size-matched `component-plan` arm **regresses** the primary
metric relative to its matched control — not a null delta, an actual loss.

| Arm | Params | structural_similarity | meaningful_program_rate | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1,755,760 | .23083 | .33333 | .73333 | 11886.1 |
| component-plan | 1,755,760 | .17250 | 0 | .63333 | 12734.3 |

Primary delta `-0.0583` (`.2308 → .1725`). `binder_reference_f1` also
regresses (`.7333 → .6333`), and `meaningful_program_rate` drops to 0. Ship
gates fail as expected on the smoke fixture (`insufficient_n`, n=3 vs
need 20), independent of the regression.

## SDLC Phase A

**Non-positive** — `primary_metric_null_or_worse` (in this case, worse) plus
`non_regression_fail:binder_reference_f1`. No new stack layer opens. Per
`sdlc` autotrain-iteration-delivery, the loop continues into the driver's
ranked successor priority: the distinct `component-edge` hypothesis is
screened next rather than re-attempting the now-falsified `component-plan`
knob at this size.

## Next priorities

1. Screen the `component-edge` hypothesis next (rank 1, confidence 0.9) —
   distinct from the falsified `component-plan` arm.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).
