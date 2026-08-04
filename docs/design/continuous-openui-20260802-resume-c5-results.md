# Continuous autotrain resumed-session cycle 5 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260802` |
| Campaign | `continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c5` |
| Source | `e09ea3daa73c4b8dffe20dff947a60b4c227c045` |
| Device | CPU |
| Steps | 20 |
| Params (both arms) | 1,766,992 |
| Intent | `retry_measurement` (replay of cycle 4's frozen pair) |

## Run matrix

| Arm | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c5-control | 3 | 1.0 | 0.333 | 0.4167 | 0.952 | 7600.20 | ship gates fail (insufficient n) |
| c5-component-edge | 3 | 1.0 | 0.333 | 0.4167 | 0.952 | 7347.03 | ship gates fail (insufficient n) |

## What this confirms

Replay of cycle 4's frozen `c4-control` / `c4-component-edge` pair. The cycle-4 control-only
decode timeout **did not reproduce** — both arms completed cleanly this time with **identical**
quality metrics. This resolves cycle 4's open question: `component-edge`'s apparently strong
numbers there were an artifact of comparing against a broken (timed-out) control, not a real
quality win — with a valid matched control in the same cycle, the primary-metric delta is exactly
`0.0`.

A small latency improvement (`7600.20ms → 7347.03ms`, ≈3.4%) was evaluated and rejected under
`efficiency_win_rejected_min_effect` (`gain_fraction=0.0345` < the 0.05 minimum-effect floor).

Correctly classified **non-positive**: `component-edge` does not beat the matched control on any
axis that clears the minimum-effect bar. No stack layer.

## Next-run priorities

1. **model:** test the distinct size-matched `component-inventory` lever next
   (`c5-component-inventory`) — `component-edge` is now a confirmed null result and exhausted.
2. **evaluation:** keep the matched control as the size-matched baseline every cycle.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the continuous loop.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c5/`
- JSON twin: `continuous-openui-20260802-resume-c5-results.json`
