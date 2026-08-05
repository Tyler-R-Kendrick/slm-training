# Continuous autotrain: 2026-08-05 (loop `continuous-openui-c3d4f8`) cycle 6 — null delta on component-edge screen

**Loop:** `continuous-openui-c3d4f8`
**Campaign:** `continuous-loop-20260805-continuous-openui-c3d4f8-986c6dc3-c6`
**Integration commit:** `28c432e4`

**Verdict:** the size-matched `component-edge` arm ties its control exactly
on the declared primary — a null delta, not positive.

| Arm | structural_similarity | parse_rate | binder_reference_f1 | meaningful_program_rate | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | .0964 | 1.0 | .82222 | 0 | 4343.63 |
| component-edge | .0964 | 1.0 | .82222 | 0 | 5007.87 |

Ship gates fail as expected on fixture scale (`insufficient_n`, `n=3` vs 20;
`meaningful_program_rate` 0 < .66; `structural_similarity` .0964 < .35).

## SDLC Phase A

**Non-positive** (`primary_metric_null_or_worse`). No new stack layer; docs
land and the loop continues with the driver's ranked successor priority
(`component-plan`, distinct from the exhausted `component-edge` null).

## Next priorities

1. Screen the `component-plan` hypothesis next (rank 1, confidence 0.9).
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).

Machine evidence:
[`continuous-openui-c3d4f8-c6-results.json`](continuous-openui-c3d4f8-c6-results.json).
