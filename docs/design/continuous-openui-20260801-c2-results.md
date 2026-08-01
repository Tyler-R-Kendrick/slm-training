# Continuous autotrain cycle 4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-c2` (follows `continuous-loop-20260801-c1`) |
| Source | `4544f049b74e2c2b7cd60a4bbdc74768a4b8588e` |
| Device | CPU |
| Steps | 122 / batch 2 / seed 100002 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` (explicit) |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c20260801-c2-control | bounds/canvas off | 3 | 1.0 | 0.0 | 0.1978 | 18184.97 | eval completed; ship gates fail |
| c20260801-c2-canvas | compact_active_canvas **on** | 3 | 1.0 | 0.0 | 0.1978 | 19131.94 | eval completed; ship gates fail |

Primary delta (canvas − control) p50 latency: **+946.97 ms** (canvas slower — null/worse).

## Diagnostics

1. Bumping steps 25 (c1) → 122 (c2, ~5x) still holds `meaningful_program_rate` at exactly 0.0 on both arms. `structural_similarity` improved (0.0575 → 0.1978), so the model is learning something, but not enough in this fixture/step regime to emit a single "meaningful" program across the 3-sample smoke suite.
2. This cycle's hypothesizer top-ranked candidate was `compact_active_canvas` (not `grammar_completion_bounds`); it made p50 latency worse (+946.97 ms) — correctly classified `primary_metric_null_or_worse` (no quality floor to even consider a latency-only win here, since mpr=0.0 on both arms).
3. Absolute p50 latency roughly quadrupled vs c1 (25 steps → ~3–4.6 s; 122 steps → ~18–19 s), as expected — more steps costs more decode-time compute per doc, not a regression signal by itself.
4. The 3-minute wall cap was not hit; there is headroom to try the `steps=244` arm this cycle's matrix proposed but did not run.

## SDLC Phase A

- positive: **False**
- stack_layer: **False** (`no_stack_layer_non_positive`)
- action: local commits + docs only, no new stacked PR layer this cycle.
- No harness signal reproduced this cycle (both failures are expected fixture/quality gates).

## Next-run priorities

1. **model_build/train_data:** mpr=0.0 held across 25→122 steps on `wf_smoke_v2`; try the unexplored `steps=244` arm, or move off the smoke fixture to a richer `train_version`, before continuing to attribute latency deltas between `grammar_completion_bounds`/`compact_active_canvas` to quality.
2. **experiments:** once mpr clears 0 on a control, replay `grammar_completion_bounds` vs control at the same step budget to re-check the c1 latency signal with a real quality floor.
3. **harness:** none open this cycle.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c2/`
- Runs: `.../runs/c20260801-c2-control/`, `.../runs/c20260801-c2-canvas/`
- JSON twin: `continuous-openui-20260801-c2-results.json`
