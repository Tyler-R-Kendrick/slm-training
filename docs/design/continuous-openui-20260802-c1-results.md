# Continuous autotrain cycle 1 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1` |
| Source | `e8ad8f0da312b1a1d6d03a6a57346f2b51c195b8` |
| Device | CPU |
| Steps | 21 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | structural_similarity | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c1-control | bounds off | 3 | 1.0 | 0.0575 | 1432.71 | **measurement incomplete** — eval crashed before ship-gate publish |
| c1-bounds | bounds **on** | 3 | 1.0 | 0.0575 | 1267.76 | **measurement incomplete** — same root cause |

Primary metric `smoke.structural_similarity`: control=0.0575, bounds=0.0575, delta=0.0 (**not conclusive** — measurement incomplete, see below).

## Diagnostics

1. `evaluate_model --ship-gates` for both arms crashed inside `publish_model_evaluation`
   (`src/slm_training/evals/agentv.py`) with `RuntimeError: AgentV SDK is unavailable;
   run npm ci in the checkout or set AGENTV_RUNNER`. Root `package.json` already pins
   `@agentv/core@4.42.4`, but this checkout had never run `npm ci`, so
   `node_modules/@agentv/core` did not exist.
2. Repaired by running `npm ci` at the repo root — no source change needed, this is an
   environment-bootstrap gap, not a code defect. Both `node_modules/@agentv/core` and
   `scripts/run_agentv_eval.mjs` now resolve.
3. Pre-gate smoke decode metrics (parse_rate, structural_similarity, latency) were
   captured before the crash for both arms and are identical on `structural_similarity`
   (0.0575) with a latency gap (bounds faster by 164.95ms) — **not** a usable primary-metric
   result since ship-gate publish never completed and `smoke.meaningful_program_rate=0.0`
   on both.

## Next-run priorities

1. **infrastructure:** replay the identical frozen control/bounds arms at evaluation only
   (checkpoints + `train_summary.json` already exist; no retraining needed) now that the
   AgentV SDK is installed.
2. **model:** hold off on any structural_similarity/latency conclusion until the replay
   produces complete scoreboards with ship-gate verdicts.
3. **harness:** continuous-loop bootstrap docs should call out `npm ci` as a prerequisite
   alongside `uv sync` so a fresh checkout doesn't fail mid-eval on cycle 1.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-control/`,
  `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-bounds/`
- JSON twin: `continuous-openui-20260802-c1-results.json`
