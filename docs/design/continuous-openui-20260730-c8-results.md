# Continuous autotrain cycle 8 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-continuous-openui-202607-98199209-c6` |
| Source | `fa01c48e05ea3cd69058f2fd7edd2a8d5790aff2` |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Wall cap | 3 minutes |
| Objective | Confirm cycle 7's root-cause diagnosis by replaying `bounds` now that the JS bridge is installed |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | binder_reference_f1 | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| control | bounds off | 3 | 1.0 | 0.0 | 0.8222 | 2278.16 | eval completed; ship gates fail (fixture n + quality) |
| bounds | bounds **on** | 3 | 1.0 | 0.0 | 0.8222 | 2349.55 | eval completed; ship gates fail (same) |

## Diagnostics

1. **Root cause confirmed.** With `src/apps/openui_bridge` and
   `src/apps/design_md_bridge` JS deps installed (`npm ci`), the
   `evaluate_model` stage that previously timed out for the `canvas`,
   `steps`, and `batch1` arms (cycles 4, 6, 7 in this doc series) completed
   in ~22s wall for both arms — no timeout, no fallback slowdown. This is a
   **session-local environment fix**, not a tracked code change, so it
   earns no stack layer, but it durably removes the dominant infra blocker
   from this session's cycles.
2. `grammar_completion_bounds` again shows a null quality delta —
   4th consecutive null result on this lever (alone or combined) across
   cycles 2, 3, 5, and this one.
3. SDLC Phase A: **non-positive** (`primary_metric_null_or_worse`). No
   stack layer opened.

## Next-run priorities

1. Retire `grammar_completion_bounds` from the active thrash rotation —
   4 consecutive null results is enough signal.
2. Document `npm ci` under `src/apps/openui_bridge` (and
   `design_md_bridge`) as a **required**, not optional, prerequisite for
   `autotrain` continuous-mode sessions — it is the difference between a
   ~20s and a >3-minute eval stage and was the root cause of 3 timed-out
   cycles this session (4, 6 [as originally run], 7).
3. Continue the loop with an untested lever.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c6/`
- JSON twin: `continuous-openui-20260730-c8-results.json`
