# Continuous autotrain cycle — 2026-08-01, campaign `continuous-loop-20260801-c3`

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-c3` (cycle 3, predecessor `continuous-loop-20260801-c2`) |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | mpr | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | --- |
| c20260801-c3-control | bounds/canvas off | 3 | 0.0 | 6512.57 | eval completed; gates fail |
| c20260801-c3-both | bounds **on**, canvas **on** | 3 | 0.0 | 6442.73 | eval completed; gates fail |

Primary delta (both − control) p50 latency: **-69.84 ms** (both faster).

## SDLC Phase A classification

`positive: false`, `stack_layer: false` — same rejection class as
`continuous-loop-20260801-c2`: `fixture_insufficient_n` (n=3 < 20) on both
arms plus `latency_win_rejected_low_mpr` (mpr 0.0 < 0.333 floor).

## Diagnostics

Third consecutive screening cycle at steps=20/batch=2 where
`meaningful_program_rate` stays at 0.0 for every lever combination tried so
far (bounds, canvas, both). The fixture is simply too small to produce a
non-zero mpr; latency deltas at this scale are screening-only signals, not
evidence of a real quality/latency tradeoff.

## Next-run priorities

1. **model:** none of bounds/canvas/both has moved mpr off 0.0 — either the
   fixture needs to grow or these latency deltas should be treated as
   screening-only until a larger `train_version` is used.
2. **process:** thrash rotation correctly advanced from `canvas` (c2) to
   `both` (c3) per the climb policy bank order.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c3/`
- JSON twin: `continuous-openui-20260801-c3-results.json`
