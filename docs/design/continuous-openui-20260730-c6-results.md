# Continuous autotrain cycle 6 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-continuous-openui-202607-98199209-c4` |
| Source | `0e4081f9a4faafaa674d79739a77aeceecefb45f` |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suites `smoke,held_out` |
| Wall cap | 3 minutes |
| Cycle role | **promotion** (heavier eval than screening) |
| Objective | Rotate the thrash bank to `steps` per cycle 5's priority (bounds/canvas bank exhausted) |

## Run matrix

| Arm | Levers | Status |
| --- | --- | --- |
| control | steps off | **infrastructure timeout** — wall cap hit running `smoke,held_out` |
| steps | steps **on** | **infrastructure timeout** — same |

No comparative metrics exist this cycle.

## Diagnostics

1. The driver auto-selected `cycle_role=promotion` this time, which
   evaluates `suites=smoke,held_out` instead of screening's `smoke` alone.
   Both arms hit the 3-minute wall cap again, unlike cycle 5's
   screening-role retry (smoke only), which completed cleanly in ~74s on
   the equivalent recipe.
2. This is the **second** wall-timeout across 4 driver invocations this
   session (cycle 4 and cycle 6). Both coincide with a heavier eval load
   than cycle 5's clean run — worth tracking as a possible promotion-tier
   eval capacity issue on this CPU sandbox, but **not yet** a reproduced
   `HarnessSignalV1` (no identical-arm replay proof; `harness_signals=[]`
   both times).
3. SDLC Phase A: **non-positive** (`wall_timeout`, `empty_metrics`,
   `primary_metric_unavailable`). No stack layer opened.

## Next-run priorities

1. Retry the identical promotion-role arms once more before concluding
   anything about promotion-tier eval capacity on this sandbox.
2. If a promotion-role cycle times out a second time on `smoke,held_out`
   specifically, file a `model_build` `HarnessSignalV1` recommending either
   a higher wall cap for promotion-role evals or a narrower `held_out`
   suite `n` for CPU fixture cycles — do not just raise `MAX_RUN_MINUTES`
   globally without evidence it's needed everywhere.
3. Screening-role cycles (smoke only) have not reproduced any timeout after
   retry (cycle 5); keep the two roles' evidence separate.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c4/`
- JSON twin: `continuous-openui-20260730-c6-results.json`
