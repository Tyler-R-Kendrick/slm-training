# Continuous autotrain cycle 4 results (2026-08-01, loop `continuous-openui-bqw0tb`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-bqw0tb` |
| Campaign | `continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c4` |
| Source | `81677cdaf2610e8491307c35f9894f71e1d70eb3` |
| Device | CPU |
| Steps | 21 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Wall cap | 3 minutes |

This cycle replays the frozen `c3-control` / `c3-both` arm manifests
(`retry_measurement`, see
[c3 results](continuous-openui-bqw0tb-c3-results.md)) now that
`node_modules/@agentv/core` is installed and `NODE_OPTIONS` is sanitized.
Both arms completed train + eval + AgentV publication this time.

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | binder_reference_f1 | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c4-control | bounds/canvas off | 3 | 1.0 | 0.0 | 0.7222 | 9624.39 | eval completed; ship gates fail |
| c4-both | bounds **+** canvas **on** | 3 | 1.0 | 0.0 | 0.7222 | 10286.11 | eval completed; ship gates fail |

Primary metric delta (both − control): **0.0** (bit-identical). Latency
delta: **+661.72 ms** (both slower).

## Diagnostics

1. The AgentV infra fix from c3 holds — both arms ran end-to-end including
   AgentV publication.
2. The combined `grammar_completion_bounds` + `compact_active_canvas` lever
   pair produced no change in `binder_reference_f1` versus the matched
   control, and cost meaningfully more decode latency, mirroring the c2
   finding for `bounds` alone.
3. Ship gates fail as expected at fixture `n=3` (`insufficient_n` plus
   quality-threshold failures on `meaningful_program_rate`,
   `structural_similarity`, etc.) — not a quality claim either way.

## Classification (SDLC Phase A)

**Non-positive.** Primary metric delta is exactly 0.0
(`primary_metric_null_or_worse`); fixture `insufficient_n` alone; no
ship-quality win. No new stack layer opened for this cycle.

## Next-run priorities

1. **model:** neither `bounds` alone (c2) nor `bounds+canvas` (c4) moved
   `binder_reference_f1` off the matched-control value; try a different
   lever bank (e.g. `steps`, `batch_size`) per the driver's rotation
   priority instead of re-testing the same pair.
2. **evaluation:** keep the matched control as the size-matched baseline
   every cycle.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   continuous loop.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c4/`
- JSON twin: `continuous-openui-bqw0tb-c4-results.json`
