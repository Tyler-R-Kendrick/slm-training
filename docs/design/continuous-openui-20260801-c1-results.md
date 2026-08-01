# Continuous autotrain cycle 1 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c1` |
| Source | `2a64db996754d2ce25edcc0d82ee448f7843cd85` |
| Device | CPU |
| Steps | 21 / batch 2 / seed 100001 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c1-control | bounds off | 3 | 1.0 | 0.0 | 3252.90 | eval completed; ship gates fail (insufficient n + quality) |
| c20260801-c1-bounds | bounds **on** | 3 | 1.0 | 0.0 | 2990.25 | eval completed; ship gates fail (same) |

Primary delta (bounds − control) p50 latency: **-262.65 ms** (negative = bounds faster).

## SDLC Phase A classification

**Non-positive.** `sdlc_delivery.json` → `positive=false`, `stack_layer=false`,
`action=no_stack_layer_non_positive`. Reasons:

1. `fixture_insufficient_n` on both arms (`n=3`, need `>=20`).
2. `latency_win_rejected_low_mpr`: the raw latency delta looks like a win
   (-262.65ms) but `meaningful_program_rate` held at `0.0` on both arms
   (`mpr=0.0 < 0.333333` required floor) — a quality-blind latency blip, not a
   genuine improvement. Per the quality-aware tradeoff rule
   (`_classify_metric_tradeoff`), latency wins require held parse/mpr with
   `mpr >= ~1/3`.

No stacked PR opened for this cycle (per `autotrain-iteration-delivery.md`,
fixture `insufficient_n` / quality-blind deltas are never positive alone).
Local commit only.

## Diagnostics

1. `grammar_completion_bounds=true` reduced smoke decode p50 latency
   nominally under this 21-step fixture recipe, but `meaningful_program_rate`
   never left `0.0` on either arm — 21 steps on `wf_smoke_v2` is not enough to
   produce structurally meaningful programs yet, so the latency delta carries
   no quality signal.
2. Eval defaulted correctly to the published `e938_role_safe_all_targets_v2`
   suite (the v1-default footgun flagged in cycle
   `continuous-openui-20260730-c2` was already fixed upstream).

## Next-run priorities

1. **model:** re-run `grammar_completion_bounds` on/off at a higher step
   budget within the wall cap to see if `meaningful_program_rate` clears the
   `mpr >= 1/3` floor before crediting any latency delta.
2. **evaluation:** keep ship gates honest; fixture `n=3` quality/volume gate
   fails are expected diagnostics, not promotion evidence.
3. **infrastructure:** none — eval_version default and AgentV SDK wiring
   (`npm ci`) both resolved cleanly this cycle.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c1/` (local, not tracked)
- Runs: `.../runs/c20260801-c1-control/`, `.../runs/c20260801-c1-bounds/`
- JSON twin: `continuous-openui-20260801-c1-results.json`
- SDLC delivery ledger: `outputs/autoresearch/continuous-loop-20260801-c1/sdlc_delivery.json`
