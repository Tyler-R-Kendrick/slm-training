# Continuous autotrain: 2026-08-05 (scheduled loop `continuous-openui-scheduled`) cycle 6 — `component-plan` hypothesis exact tie, rejected

**Loop:** `continuous-openui-scheduled`
**Campaign:** `continuous-loop-20260805-continuous-openui-schedu-1e62ecf9-c6`
**Integration commit:** `e0014796`

**Verdict:** non-positive. c5's top-ranked priority (test the `component-plan`
quality hypothesis) produced an **exact tie** with its matched control on
every measured smoke metric.

| Arm | primary (`smoke.structural_similarity`) | `meaningful_program_rate` | `binder_reference_f1` | `latency_ms_p50` | `forwards_count_mean` |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.0964 | 0.0 | 0.822222 | 3,554.0 | 5 |
| component-plan (candidate) | 0.0964 | 0.0 | 0.822222 | 3,611.6 | 5 |

`improvement=0.0` on the primary metric. Both arms produce a smaller/faster
model than c4/c5 (5 forwards vs. 21, ~3.6s p50 vs. ~30-35s), but
`meaningful_program_rate` dropped to `0` for both (vs. `0.333` in c4/c5) —
consistent with a distinct, not-yet-competitive configuration rather than a
harness problem (both arms behave identically, ruling out an arm-specific
bug).

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse` with `improvement=0.0`,
`fixture_insufficient_n` on both arms). No stack layer; local-commit-only
record.

## Next priorities (ranked, from the driver)

1. The completed non-positive `component-plan` arm is exhausted; test the
   distinct size-matched `component-edge` quality hypothesis next
   (`experiment_next`, confidence 0.90).
