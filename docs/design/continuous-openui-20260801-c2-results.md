# Continuous autotrain cycle 2 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c2` |
| Source | `24c20769c366aeb9e9f7a98eb72089b3a97859c7` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Hypothesis | `compact_active_canvas` reduces `smoke.latency_ms_p50` vs. matched control without lowering `parse_rate` |

## Run matrix

| Arm | Levers | parse_rate | meaningful_program_rate | latency_ms_p50 | decode_timeout_rate | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c2-control | canvas off | 1.0 | 0.0 | 21665.04 | — | eval completed; ship gates fail (insufficient n + quality) |
| c2-canvas | canvas **on** | 1.0 | 0.0 | 21863.33 | 0.667 | eval completed; ship gates fail (same) |

Primary delta (canvas − control) p50 latency: **+198.29 ms** (candidate slower).

## Diagnostics

1. Both arms ran the full multi-suite ship-gate eval successfully (unlike
   cycle 1 — see
   [`continuous-openui-20260801-c1-results.md`](continuous-openui-20260801-c1-results.md)
   for the infra blocker this replay fixed).
2. `compact_active_canvas=True` did **not** improve smoke decode p50 under
   this recipe; the candidate arm was slower and showed a materially higher
   `decode_timeout_rate` (0.667 vs. unmeasured/low on control), a plausible
   proximate cause worth isolating before any re-test.
3. Fixture 20-step models correctly fail quality/volume gates on both arms
   (`fixture_insufficient_n`) — expected at smoke scale, not evidence against
   promotion either way.
4. Driver's own SDLC Phase A classifier recorded `positive=false`,
   `stack_layer=false` (`primary_metric_null_or_worse`,
   `fixture_insufficient_n_alone`) — no stacked PR opened for this cycle per
   `autotrain-iteration-delivery`.

## Next-run priorities

1. **model:** do not promote `compact_active_canvas`; retest only past the
   fixture volume gate.
2. **model/diagnosis:** root-cause the canvas arm's `decode_timeout_rate=0.667`
   before re-testing the lever — it may explain the latency regression
   directly.
3. **infrastructure:** none open from this cycle (AgentV blocker already
   resolved by the cycle-1 self-heal).

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c2/` (local only, gitignored)
- Runs: `.../runs/c20260801-c2-control/`, `.../runs/c20260801-c2-canvas/`
- SDLC delivery record: `outputs/autoresearch/continuous-loop-20260801-c2/sdlc_delivery.json` (`positive=false`, `stack_layer=false`)
- JSON twin: `continuous-openui-20260801-c2-results.json`
