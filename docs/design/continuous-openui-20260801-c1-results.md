# Continuous autotrain cycle 1 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c1` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps | 20 / batch 2 / seed 7 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c1-control | bounds off | 3 | 1.0 | 0.0 | 3939.16 | eval completed; ship gates fail (insufficient n + quality) |
| c1-bounds | bounds **on** | 3 | 1.0 | 0.0 | 3758.55 | eval completed; ship gates fail (same) |

Primary delta (bounds − control) p50 latency: **-180.60 ms** (bounds faster).

## Diagnostics

1. `grammar_completion_bounds=True` reduced smoke p50 latency by 180.6ms, but
   `meaningful_program_rate` held at 0.0 on both arms.
2. Quality-aware Phase A policy rejects pure-latency wins below the mpr floor
   (`mpr=0.0 < 0.333333`) — `latency_win_rejected_low_mpr`. **Non-positive.**

## SDLC Phase A

`positive=False`, `stack_layer=False`, `action=no_stack_layer_non_positive` —
local commit only, no stacked PR for this cycle.

## Next-run priorities

1. Re-test `grammar_completion_bounds` once paired with a lever that also
   moves `meaningful_program_rate` above the win floor.
2. Confirm the latency gain replicates at a higher step budget before
   treating it as durable.

## Artifacts

- Campaign (ephemeral, not committed): `continuous-loop-20260801-c1/`
- JSON twin: `continuous-openui-20260801-c1-results.json`
