# Continuous autotrain cycle 5 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-continuous-openui-202607-98199209-c3` |
| Source | `90a3ff4bdf0a932b2fffc8d9ab0ff03cddeb7126` |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Objective | Retry cycle 4's timed-out recipe, rotated to the `both` (bounds + canvas) lever |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | binder_reference_f1 | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| control | both off | 3 | 1.0 | 0.0 | 0.7222 | 6736.19 | eval completed; ship gates fail (fixture n + quality) |
| both | bounds **on**, canvas **on** | 3 | 1.0 | 0.0 | 0.7222 | 6488.99 | eval completed; ship gates fail (same) |

Primary metric `smoke.binder_reference_f1` delta: **0.0** (identical). Latency
moved -247.2 ms (both faster) but is not the declared primary and is not,
alone, a positive result.

## Diagnostics

1. **Cycle 4's timeout did not reproduce.** This retry of the equivalent
   recipe finished in ~74s wall for both arms — confirms the prior wall-cap
   hit (`continuous-openui-20260730-c4-results.md`) was session-level
   resource variance (this session had just installed a fresh venv +
   CUDA/torch wheels), not a `compact_active_canvas` or `model_build`
   harness defect. No repair lane opened.
2. The combined `bounds + canvas` lever again shows a null quality delta —
   the third consecutive non-positive result on this lever bank (cycles 2,
   3, 5 in this session; cycle 2 of the prior session already showed the
   same for `bounds` alone).
3. SDLC Phase A: **non-positive**
   (`primary_metric_null_or_worse:smoke.binder_reference_f1:control=0.7222
   candidate=0.7222 improvement=0.0`). No stack layer opened.

## Next-run priorities

1. **model:** stop thrashing the `bounds`/`canvas` lever bank — rotate to an
   untested lever family for the next cycle.
2. **infrastructure:** confirmed transient timeout, not a repeated hard
   blocker; no `HarnessSignalV1` filed.
3. **process:** continue the loop; do not promote off screening evidence.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c3/`
- JSON twin: `continuous-openui-20260730-c5-results.json`
