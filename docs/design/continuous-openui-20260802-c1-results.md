# Continuous autotrain cycle 1 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1` |
| Source | `528aadc4712d3f5a76d59a1ba846cdc23abed1ce` |
| Device | CPU |
| Steps | 21 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | smoke n | structural_similarity | Status |
| --- | ---: | ---: | --- |
| c1-control | — | 0.0575 (fallback) | train completed; **eval crashed** (AgentV SDK unavailable) |
| c1-bounds | — | 0.0575 (fallback) | train completed; **eval crashed** (AgentV SDK unavailable) |

Primary delta (bounds − control) `smoke.structural_similarity`: **0.0** — a
degenerate fallback because `evaluate_model` never produced a real
scoreboard, not a measured quality tie. SDLC Phase A correctly classified
the cycle `NON_POSITIVE` on this null delta.

## Diagnostics

1. `evaluate_model --ship-gates` failed closed with
   `RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or
   set AGENTV_RUNNER` for both arms — this fresh sandbox had never run
   `npm ci`, so `src/slm_training/evals/agentv.py::_agentv_runtime` could
   not resolve a runner.
2. Root cause fixed in-cycle: ran `npm ci` at the repo root. It had to be
   invoked with `NODE_OPTIONS` unset — the session's ambient
   `NODE_OPTIONS=--import tsx --max-old-space-size=8192` is rejected by this
   Node 22 build (`--import tsx is not allowed in NODE_OPTIONS`), which was
   silently breaking every `node`/`npm` invocation, including the one
   `evaluate_model` shells out to.
3. Training itself completed cleanly on both arms (21 steps, CPU); the
   failure is isolated to the eval/AgentV bootstrap step, not the model or
   harness code.

## Next-run priorities

1. **infrastructure:** verify the AgentV SDK (`npm ci`) before the first
   `evaluate_model` call in a fresh continuous-loop sandbox, so a cold
   environment doesn't burn a screening cycle on an infra crash instead of a
   real measurement.
2. **model:** replay the c1 bounds/control arms (or the queued
   `component-plan` hypothesis) now that the AgentV SDK is installed, to get
   an honest scoreboard.
3. **evaluation:** do not read the 0.0575/0.0575 pair as a quality signal
   for either arm; it is a crash-fallback constant.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-control/`,
  `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-bounds/`
- JSON twin: `continuous-openui-20260802-c1-results.json`
