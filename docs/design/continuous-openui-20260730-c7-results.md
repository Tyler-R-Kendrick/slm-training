# Continuous autotrain cycle 7 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-continuous-openui-202607-98199209-c5` |
| Source | `f63a25b36612147ee1b777bfa33a641b87a92549` |
| Device | CPU |
| Steps | 20 / batch 2 (control) vs 1 (candidate) |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Wall cap | 3 minutes |
| Objective | Rotate the thrash bank to `batch1` (`batch_size=1`) |

## Run matrix

| Arm | smoke n | parse_rate | meaningful_program_rate | binder_reference_f1 | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control | 3 | 1.0 | 0.0 | 0.7333 | 3149.53 | eval completed; ship gates fail (fixture n + quality) |
| batch1 | — | — | — | — | — | **infrastructure timeout** — no comparative evidence |

## Diagnostics

1. **Root cause identified for the recurring timeouts:** this session's
   venv never ran `cd src/apps/openui_bridge && npm ci` (README's optional
   acceleration step), so every decode falls back to the pure-Python `lark`
   parser for AST-completeness checks (`Install bridge deps: ...` appears
   in the eval evidence of every run this session, including the ones that
   completed cleanly). The fallback is functionally correct but slow, and
   under `batch_size=1` (more decode calls, smaller batching) it pushed
   this arm over the 3-minute wall cap. The npm bridge itself is not
   installed here; installing it is an environment-setup change, not a
   code fix, and out of scope for this unattended docs cycle.
2. **Metric variance note:** three untouched controls this session scored
   `binder_reference_f1` = 0.6333 (cycle 3), 0.7222 (cycle 5-early /
   cycle 3's second control), 0.7333 (this cycle) — each a fresh training
   run at the same recipe. This confirms the metric has real run-to-run
   variance at fixture scale; a single paired control/candidate comparison
   is not enough to attribute a quality delta to a lever with confidence.
3. SDLC Phase A: **non-positive** (`wall_timeout`, `empty_metrics`,
   `primary_metric_unavailable` on the candidate arm). No stack layer
   opened.

## Next-run priorities

1. Installing the `openui_bridge` JS dependency would remove the slow
   fallback path and should reduce future promotion/`batch1`-class
   timeouts.
2. Any future comparative claim on `binder_reference_f1` needs multiple
   seeds per arm, not one paired run, given the observed 0.63-0.73 spread
   across three untouched controls.
3. Retry `batch1` once the JS bridge is available; it is inconclusive, not
   rejected.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-202607-98199209-c5/`
- JSON twin: `continuous-openui-20260730-c7-results.json`
