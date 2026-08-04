# Continuous autotrain cycle 4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-continuous-openui-202607-98199209-c2` |
| Source | `9afc974a1cb1d16518547279cc3f6ff9149432dd` |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Objective | Rotate the thrash lever bank to `compact_active_canvas` per cycle-3's rank-1 priority |

## Run matrix

| Arm | Levers | Status |
| --- | --- | --- |
| control | canvas off | **infrastructure timeout** — `evaluate_model` interrupted mid-decode at the 3-minute wall cap, no typed summary |
| canvas | canvas **on** | **infrastructure timeout** — same wall-cap interruption |

No comparative metrics exist this cycle: both arms failed the same way, so
this is not attributable to the `compact_active_canvas` lever specifically.

## Diagnostics

1. Both the size-matched control and the candidate hit `MAX_RUN_MINUTES=3`
   during `evaluate_model` decode with `KeyboardInterrupt` traces inside the
   grammar completion-forest recursion (`completion_kernel._eval`). Because
   the *control* also timed out, this reads as session-level resource
   variance in this run (this session did a fresh `uv pip install -e ".[dev]"`
   including CUDA wheels moments earlier), not a `compact_active_canvas`
   regression.
2. Per `autotrain` continuous-mode rule 4, a soft failure (timeout) never
   stops the loop; the same blocker must repeat 3 consecutive cycles with no
   new information before reporting `blocked`. This is occurrence 1.
3. SDLC Phase A: **non-positive** (`wall_timeout`, `empty_metrics`,
   `primary_metric_unavailable` on both arms). No stack layer opened.

## Next-run priorities

1. Retry the identical control + canvas arms once. If the control alone
   times out again with no lever difference, escalate to a `model_build`
   `HarnessSignalV1` (infra/perf), not a model hypothesis.
2. Otherwise continue rotating the lever bank away from
   `grammar_completion_bounds` (already null on quality twice: cycles 2, 3).

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c2/`
- JSON twin: `continuous-openui-20260730-c4-results.json`
