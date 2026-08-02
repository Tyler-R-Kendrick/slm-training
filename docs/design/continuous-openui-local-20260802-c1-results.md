# Continuous autotrain cycle 1 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1` |
| Source | `b8188a49672f187146bf1a76a353cbe188f9b99d` |
| Device | CPU |
| Steps | 21 / seed 0 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Ship gates | requested (`--ship-gates`) |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c1-control | bounds off | 3 | 1.0 | 0.0 | 0.0575 | 1225.40 | metrics computed; **scoreboard missing** (AgentV publish crash) |
| c1-bounds | bounds on | 3 | 1.0 | 0.0 | 0.0575 | 1260.34 | metrics computed; **scoreboard missing** (AgentV publish crash) |

Primary metric (`smoke.structural_similarity`) delta: **0.0** (tied).

## Diagnostics

1. This was a fresh ephemeral session with no repo bootstrap done yet: no
   Python 3.12 venv, and `npm ci` had never been run at the repo root, so
   `@agentv/core` was absent. `evaluate_model --ship-gates` reached
   `publish_model_evaluation` → `_agentv_runtime`, which raises
   `RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or
   set AGENTV_RUNNER` for **both** arms, so `scoreboard.json` was never
   written and the driver correctly classified the cycle
   `measurement_incomplete` on both arms.
2. Fixed in-session: created `.venv-local` (Python 3.12.3) and ran
   `pip install -e ".[dev]"`, then `npm ci` at the repo root (267 packages,
   AgentV SDK now present). Node also needed `NODE_OPTIONS` unset for this
   session — the ambient `NODE_OPTIONS` value was malformed
   (`"--import tsx" --max-old-space-size=8192`) and broke `node`/`npm`
   invocation entirely; `env -u NODE_OPTIONS` works around it.
3. Even setting the scoreboard/AgentV gap aside, the raw smoke metrics that
   *did* compute show **zero primary-metric delta** and
   `meaningful_program_rate = 0.0` on both arms at 21 steps — not a positive
   result even once the eval pipeline completes cleanly. The driver queued
   `retry_measurement` for the identical frozen arms rather than a new
   hypothesis, per the frozen-replay contract.

## Classification

Non-positive (`SDLC_PHASE_A NON_POSITIVE`, `stack_layer=False`,
`action=no_stack_layer_non_positive`):
`measurement_incomplete` on both arms (missing scoreboard) and
`primary_metric_null_or_worse` (delta 0.0). No stacked PR opened for this
cycle per `sdlc` autotrain-iteration-delivery.

## Next-run priorities

1. **infrastructure:** replay the identical frozen `c1-control` / `c1-bounds`
   arms now that AgentV deps are installed, to get a real scoreboard before
   drawing any model conclusion.
2. **process:** fresh-checkout sessions need a documented bootstrap
   (`python3.12 -m venv .venv-local && pip install -e ".[dev]"` +
   `npm ci`) before the first autotrain cycle; consider a `SessionStart`
   hook so this isn't rediscovered per session.
3. **model:** `grammar_completion_bounds` (bounds arm) shows no smoke-quality
   or latency advantage yet at this step count; re-evaluate after the
   scoreboard is unblocked and/or at higher steps within the wall cap.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-control/`,
  `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-bounds/`
- JSON twin: `continuous-openui-local-20260802-c1-results.json`
