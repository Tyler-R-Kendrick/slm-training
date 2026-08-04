# Continuous autotrain cycle 1 results (2026-08-04)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1` |
| Source | `eba6db3044076285581b80cfe5294a2ecbcee8a1` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Status |
| --- | --- |
| c1-control | trained; **eval harness_failure** |
| c1-bounds | trained; **eval harness_failure** |

No scoreboard was produced for either arm -- `primary_metric` (`smoke.structural_similarity`)
is unavailable this cycle.

## Root cause and repair

`scripts.evaluate_model --ship-gates` calls `publish_model_evaluation`, which requires the pinned
AgentV SDK (`node_modules/@agentv/core`). This scheduled-task session runs in a fresh ephemeral
container that had never run `npm ci`, so `src/slm_training/evals/agentv.py:_agentv_runtime`
raised `RuntimeError: AgentV SDK is unavailable`, and both arms failed at evaluation.

This is the **same known environment-bootstrap gap** already documented for the 2026-08-02 c2
cycle (`docs/design/continuous-openui-20260802-c2-results.md`,
`docs/design/autotrain-cycle-c2-agentv-missing-infra-failure.md`), not a new repository code
defect:

Repair actions (`repair_harness`, family `model_build`, frozen manifest
`8f9d49fc2b0fe5e89b1975d963f031e0efcd028b91611d8f25468c1b0acbb658`):

1. Ran `NODE_OPTIONS= npm ci` to install the pinned SDK (the ambient sandbox `NODE_OPTIONS=--import
   tsx` is rejected by this Node 22 build; `src/slm_training/bridge_utils.py:sanitized_node_env`
   already clears it correctly for every spawned bridge subprocess, so no code change was needed
   there).
2. Verified `tests/test_evals/test_agentv.py` passes 8/8 on this commit -- the stale
   `ast_beq_rate`/`canonical_beq_rate` ship-gate fixtures found and fixed during the 2026-08-02 c2
   cycle are already merged upstream.
3. No code change required this cycle; confirmed bootstrap-only gap.

## Next-run priorities

1. **infrastructure:** replay `c20260804-continuous-openui-local-8c0b60dd-c1-control` /
   `-bounds` now that the harness is repaired (queued via `retry_measurement`).

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1/`
- Runs: `.../runs/c20260804-continuous-openui-local-8c0b60dd-c1-control/`,
  `.../runs/c20260804-continuous-openui-local-8c0b60dd-c1-bounds/`
- JSON twin: `continuous-openui-local-20260804-c1-results.json`
