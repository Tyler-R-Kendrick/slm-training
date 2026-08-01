# Continuous autotrain cycle 3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c3` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps | 20 / batch 2 / seed 7 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c3-control | bounds off, canvas off | 3 | 1.0 | 0.0 | 10111.73 | eval completed; ship gates fail (insufficient n + quality) |
| c3-both | bounds **on**, canvas **on** | 3 | 1.0 | 0.0 | 9514.05 | eval completed; ship gates fail (same) |

Primary delta (both − control) p50 latency: **-597.68 ms** (combined levers
faster).

## Diagnostics

1. With the AgentV SDK fixed (see cycle 2), both arms completed all 3 smoke
   documents.
2. Combined `grammar_completion_bounds` + `compact_active_canvas` again
   reduced p50 latency, consistent in direction with the single-lever c1
   result, but `meaningful_program_rate` again held at 0.0 — rejected by the
   same `latency_win_rejected_low_mpr` gate (`mpr=0.0 < 0.333333`).
3. Both arms also hit `fixture_insufficient_n` (smoke n=3). **Non-positive.**

## SDLC Phase A

`positive=False`, `stack_layer=False`, `action=no_stack_layer_non_positive` —
local commit only, no stacked PR for this cycle.

## Next-run priorities

1. Bounds+canvas now has two consistent latency-only wins (c1 single-lever,
   c3 combined); still gated on `mpr=0` until a quality-moving lever is
   found (see c4).
2. Raise fixture `n` above the insufficient-n floor before treating any
   latency delta here as more than a screening signal.

## Artifacts

- Campaign (ephemeral, not committed): `continuous-loop-20260801-c3/`
- JSON twin: `continuous-openui-20260801-c3-results.json`
