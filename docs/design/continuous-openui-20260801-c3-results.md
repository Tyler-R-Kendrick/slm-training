# Continuous autotrain cycle 3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260801-c3` |
| Source | `24c20769c366aeb9e9f7a98eb72089b3a97859c7` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c3-control | bounds off, canvas off | 3 | 1.0 | 0.0 | 6984.26 | eval completed; ship gates fail (insufficient n + quality) |
| c20260801-c3-both | bounds **on**, canvas **on** | 3 | 1.0 | 0.0 | 7034.74 | eval completed; ship gates fail (same) |

Primary delta (both − control) p50 latency: **+50.48 ms** (positive = combined arm slower).

## Diagnostics

1. Third consecutive cycle in this loop-id: c1 tested `grammar_completion_bounds`
   alone (+47.59ms), c2 tested `compact_active_canvas` alone (+2325.22ms), c3
   tests both together (+50.48ms). All three regress smoke p50 latency with no
   `meaningful_program_rate` gain.
2. `smoke n=3` stays below the `insufficient_n` ship-gate threshold
   (`need>=20`) at this step budget, so these deltas are wiring-scale signal,
   not a promotion-grade quality comparison.
3. Working hypothesis: at n=3/20-step fixture scale, per-call constrained-decode
   bookkeeping overhead dominates the latency signal rather than either lever
   causing a genuine regression — recommend not screening these two levers on
   latency alone until n and step budget both clear the fixture floor.

## Next-run priorities

1. **model:** stop screening `grammar_completion_bounds` /
   `compact_active_canvas` latency deltas at this fixture scale; re-test only
   once a higher-step/higher-n budget clears fixture `insufficient_n` and
   `meaningful_program_rate > 0`.
2. **evaluation:** keep ship gates honest; do not weaken for continuous smoke.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c3/`
- JSON twin: `continuous-openui-20260801-c3-results.json`
