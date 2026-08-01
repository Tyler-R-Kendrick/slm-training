# Continuous autotrain cycle — 2026-08-01, campaign `continuous-loop-20260801-c1`

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-c1` (cycle 1) |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps | 20 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | Status | Failure |
| --- | --- | --- | --- |
| c20260801-c1-control | bounds off | **failed** (exit 1) | `evaluate_model` crashed in `evals.agentv._agentv_runtime`: AgentV SDK unavailable |
| c20260801-c1-bounds | bounds **on** | **failed** (exit 1) | same crash |

Partial pre-crash latency (diagnostic only, not gate-evaluated): control 2425.26ms p50, bounds 2388.75ms p50.

## Diagnostics

1. This container's checkout had no `node_modules`; `evaluate_model --ship-gates` calls into `publish_model_evaluation` → `agentv.py::_agentv_runtime`, which raises when the AgentV SDK package isn't installed.
2. `npm ci` initially failed with `node: --import tsx is not allowed in NODE_OPTIONS` — the ambient `NODE_OPTIONS="--import tsx" --max-old-space-size=8192` is rejected by this node build for plain `node`/`npm` invocations. Running with `NODE_OPTIONS` unset (`env -u NODE_OPTIONS npm ci`) succeeded (267 packages installed).
3. The `eval_version` footgun documented in `continuous-openui-20260730-c2-results.md` did **not** recur — `eval_version=e938_role_safe_all_targets_v2` resolved to a real published suite; the crash was purely the missing AgentV SDK, not a path/suite error.

## Next-run priorities

1. **infrastructure:** consider a fail-closed AgentV-availability pre-flight in `run_autotrain_continuous` (or document `npm ci` as a required setup step for fresh checkouts) so a missing SDK reports as a clear setup blocker rather than a mid-eval traceback.
2. **model:** re-run bounds/canvas to completion — done immediately after in campaign `continuous-loop-20260801-c2` (see that doc).

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c1/`
- Runs: `.../runs/c20260801-c1-control/`, `.../runs/c20260801-c1-bounds/`
- JSON twin: `continuous-openui-20260801-c1-results.json`
- SDLC Phase A: non-positive (`empty_metrics` on both arms) — no stack layer opened.
