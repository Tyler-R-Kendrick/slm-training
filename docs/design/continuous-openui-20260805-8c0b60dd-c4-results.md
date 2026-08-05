# Continuous autotrain cycle 4 results (2026-08-05, `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c4` |
| Source | `81df7d8e` |
| Train | `wf_smoke_v2`, 20 steps |
| Eval | `e938_role_safe_all_targets_v2` |
| Params | 1,766,987 each (size-matched) |

## Run matrix

| Arm | Levers | smoke n | parse | MPR | structural_similarity | binder F1 | comp. recall | fidelity | reward | latency_ms_p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | component-edge off | 3 | 1.0 | 0.3333 | 0.4167 | 0.9524 | 0.25 | 0.9167 | 0.936 | 28,544.2 |
| component-edge | component-edge **on** | 3 | 1.0 | 0.3333 | 0.4167 | 0.9524 | 0.25 | 0.9167 | 0.936 | 26,376.7 |

Every guarded quality metric is an **exact tie**. Primary metric
(`smoke.structural_similarity`) delta: **0.0**.

## Diagnostics

1. The driver's SDLC Phase A classification flags an `efficiency_win`
   (`mpr_per_ms` gain_fraction `0.0822`, above the `0.05` floor — candidate
   is 7.6% faster p50: 26,376.7 vs 28,544.2 ms) with `quality_held`.
2. This does **not** clear the quality-aware positive bar: latency-only
   wins require a held **or positive** primary-metric move, not merely a
   favorable efficiency ratio, per `_classify_metric_tradeoff`. With
   `primary_metric_null_or_worse` (structural_similarity improvement
   `0.0`), the cycle is correctly classified `NON_POSITIVE`.
3. `fixture_insufficient_n_alone` also applies (smoke `n=3`).

## Next-run priorities

1. `component-edge` vs matched control is a quality tie with a
   latency-only signal — not sufficient for a positive classification on
   its own.
2. Test the distinct size-matched `component-plan` quality hypothesis next
   per the driver's ranked priorities.
3. Do not promote or ship either checkpoint.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260805-continuous-openui-local-8c0b60dd-c4/`
- JSON twin: `continuous-openui-20260805-8c0b60dd-c4-results.json`
