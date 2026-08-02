# Continuous autotrain cycle 1 results (2026-08-02, `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1` |
| Source | `63b36fbfa0748ed18b8db85b656c8828cc8178c3` |
| Device | CPU |
| Steps | 21 / batch 2 / seed 100001 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c1-control | bounds off | 3 | 1.0 | 0.0 | 0.0575 | 1594.76 | eval scoreboard written; AgentV bundle publish crashed |
| c1-bounds | bounds **on** | 3 | 1.0 | 0.0 | 0.0575 | 1553.54 | eval scoreboard written; AgentV bundle publish crashed (same cause) |

Both arms are size-matched at 1,608,962 trainable params. Primary metric delta
(bounds − control) on `smoke.structural_similarity`: **0.0** (exact tie).

## Diagnostics

1. This sandbox had never run `npm ci`, so `node_modules/@agentv/core` was
   missing. `evaluate_model.py --ship-gates` writes the core `eval.json`
   scoreboard first, then crashes inside `publish_model_evaluation` →
   `_agentv_runtime` with `AgentV SDK is unavailable; run npm ci in the
   checkout or set AGENTV_RUNNER` (`src/slm_training/evals/agentv.py:32`).
2. The autotrain SDLC classifier correctly treated both arms as
   `measurement_incomplete` / `missing_scoreboard` rather than reporting a
   quality result on a partial run — this is the classifier working as
   designed, not a harness defect.
3. `npm ci` was run in this checkout immediately after this cycle to install
   the pinned `@agentv/core`/`agentv` devDependencies; the next cycle replays
   the identical frozen `c1-control` / `c1-bounds` manifests
   (`frozen_manifest_sha256: d40b36c9c2d0ca3ff4c7766decd89d605720c88c755739a1b8ab246093497eb4`)
   per this cycle's `retry_measurement` action before any new hypothesis.

## Next-run priorities

1. **infrastructure:** replay the frozen `c1-control`/`c1-bounds` arms now
   that AgentV is installed (this cycle's `retry_measurement` action).
2. **model:** re-test `grammar_completion_bounds` for a real latency/quality
   delta only once the replay produces a complete AgentV-backed scoreboard.
3. **evaluation:** keep ship gates honest; do not weaken for continuous
   smoke; do not promote/sync/ship either fixture checkpoint.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-control/`,
  `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-bounds/`
- JSON twin: `continuous-openui-20260802-c1-results.json`
