# Continuous autotrain: 2026-08-05 (scheduled loop `continuous-openui-scheduled`) cycle 5 — frozen replay complete, candidate rejected on non-regression

**Loop:** `continuous-openui-scheduled`
**Campaign:** `continuous-loop-20260805-continuous-openui-schedu-1e62ecf9-c5`
**Replay of:** [`continuous-loop-20260805-continuous-openui-schedu-1e62ecf9-c4`](continuous-openui-scheduled-c4-results.md)
**Integration commit:** `eaf8e16f`

**Verdict:** non-positive. The driver auto-consumed c4's queued `retry_measurement`
action and replayed the identical frozen control/candidate manifest. Both
arms now ran to a **complete** honest measurement (c4's candidate arm never
executed at all). The candidate wins on raw primary metric and latency, but
**regresses `binder_reference_f1`**, so this does not qualify as a positive
result.

| Arm | primary (`smoke.structural_similarity`) | `meaningful_program_rate` | `binder_reference_f1` | `latency_ms_p50` | Ship gates |
| --- | ---: | ---: | ---: | ---: | --- |
| control | 0.416667 | 0.333333 | **0.952381** | 34,482.6 | fail (expected, fixture n=3) |
| steps (candidate) | **0.51** | 0.333333 | **0.822222** | 14,581.2 | fail (expected, fixture n=3) |

## Why this is not a positive result

- `primary_metric` improves (+0.0933) and latency drops (~58% faster,
  `mpr_per_ms` +136%), which alone might look like a win.
- But `binder_reference_f1` regresses **0.952381 → 0.822222** — a real
  quality loss on structural correctness, not noise. This repo's
  quality-aware tradeoff classification
  (`_classify_metric_tradeoff`) correctly treats a latency/primary win that
  trades away a tracked non-regression quality signal as **not positive**.
- Both arms remain `fixture_insufficient_n` (n=3, need ≥20) — fixture-scale
  evidence is wiring only, never a ship claim, independent of the
  non-regression finding above.

## SDLC Phase A

**Non-positive** (`non_regression_fail`, `primary_metric_null_or_worse`
under the quality-aware classifier, `fixture_insufficient_n` on both arms).
No stack layer. This doc is a local-commit-only record, per
`autotrain-iteration-delivery`.

## Next priorities (ranked, from the driver)

1. Test the distinct size-matched `component-plan` quality hypothesis next
   (`experiment_next`, confidence 0.90) — the completed candidate is
   exhausted and cannot be re-selected without a new preregistered
   hypothesis.
2. Keep the matched control as the size-matched baseline every cycle
   (`experiment_next`, confidence 0.70).
