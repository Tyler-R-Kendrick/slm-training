# Continuous autotrain cycle 3 results (2026-08-01, loop `continuous-openui-bqw0tb`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-bqw0tb` |
| Campaign | `continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c3` |
| Device | CPU |
| Steps | 21 / batch 2 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Status |
| --- | --- |
| c3-control | **failed** — train completed (`binder_reference_f1=0.7222`), eval crashed in AgentV publication |
| c3-both | **failed** — identical crash |

## Diagnostics

1. Both arms trained to completion and evaluate_model's scoreboard capture
   shows identical `smoke.binder_reference_f1=0.7222` for both arms (0.0
   delta) before the process crashed — this is a repeat of the c1 shape:
   an infrastructure gap, not a model or lever result.
2. Root cause, `scripts/evaluate_model.py` → `evaluate_suites` →
   `publish_model_evaluation` → `publish_agentv_evaluation` →
   `_agentv_runtime` (`src/slm_training/evals/agentv.py:15`) raised:
   `RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or
   set AGENTV_RUNNER`.
3. Two compounding causes in this fresh container:
   - `node_modules/@agentv/core` had never been installed (no `npm ci` run
     yet this session).
   - The sandbox sets a global `NODE_OPTIONS='--import tsx ...'` that
     Node.js itself rejects (`--import tsx is not allowed in NODE_OPTIONS`).
     This exact class of gap is already called out and worked around
     elsewhere in this repo (`scripts/publish_cap2_operator_policy_rebase.py`,
     `scripts/run_rsc_a02_depth_aux_mode_factorial.py`) but not inside
     `src/slm_training/evals/agentv.py`'s own subprocess invocation.
4. Fix applied: `NODE_OPTIONS= npm ci` at the repo root (267 packages
   installed, including `@agentv/core`). The driver's own diagnosis recorded
   a `retry_measurement` action against the frozen `c3-control` / `c3-both`
   manifests, consumed by replaying the identical arms with the SDK now
   present and `NODE_OPTIONS` sanitized for the subprocess.

## Next-run priorities

1. **infrastructure:** replay the frozen `c3-control` / `c3-both` arms now
   that `@agentv/core` is installed and `NODE_OPTIONS` is sanitized.
2. **harness (model_build / eval):** consider hardening
   `src/slm_training/evals/agentv.py`'s Node subprocess call to always clear
   `NODE_OPTIONS` itself (mirroring the existing
   `publish_cap2_operator_policy_rebase.py` workaround), so this sandbox gap
   self-heals without a manual `npm ci` step in future fresh containers.
3. **evaluation:** soft ship-gate/infra fails on fixture `n` never stop the
   continuous loop; proceed regardless.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c3/`
- JSON twin: `continuous-openui-bqw0tb-c3-results.json`
