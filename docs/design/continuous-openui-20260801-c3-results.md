# Continuous autotrain cycle 3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c3` |
| Source | `24c20769c366aeb9e9f7a98eb72089b3a97859c7` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Hypothesis | Combined `grammar_completion_bounds` + `compact_active_canvas` beat either single lever on `smoke.latency_ms_p50` |

## Run matrix

| Arm | Levers | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | --- |
| c3-control | both off | 1.0 | 0.0 | 7723.25 | eval completed; ship gates fail (insufficient n + quality) |
| c3-both | bounds **on**, canvas **on** | 1.0 | 0.0 | 7864.44 | eval completed; ship gates fail (same) |

Primary delta (both − control) p50 latency: **+141.19 ms** (candidate slower).

## Diagnostics

1. Neither the single-lever tests (`grammar_completion_bounds` alone, prior
   loop; `compact_active_canvas` alone, cycle 2 — see
   [`continuous-openui-20260801-c2-results.md`](continuous-openui-20260801-c2-results.md))
   nor the combined pair beat the matched control on smoke p50 latency in
   this recipe. The combined-levers hypothesis is refuted for this recipe.
2. Fixture 20-step models correctly fail quality/volume gates on both arms —
   expected at smoke scale, not evidence against promotion either way.
3. Driver's SDLC Phase A classifier recorded `positive=false`,
   `stack_layer=false` — no stacked PR opened for this cycle.

## Next-run priorities

1. **model:** do not promote the combined bounds+canvas pair.
2. **model:** the thrash rotation already moved to the `steps` lever next
   (cycle 4 — see
   [`continuous-openui-20260801-c4-results.md`](continuous-openui-20260801-c4-results.md)),
   which found a real efficiency win unrelated to these grammar/canvas
   levers.
3. **evaluation:** re-test bounds/canvas combos only past the fixture volume
   gate.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c3/` (local only, gitignored)
- Runs: `.../runs/c20260801-c3-control/`, `.../runs/c20260801-c3-both/`
- SDLC delivery record: `outputs/autoresearch/continuous-loop-20260801-c3/sdlc_delivery.json` (`positive=false`, `stack_layer=false`)
- JSON twin: `continuous-openui-20260801-c3-results.json`
