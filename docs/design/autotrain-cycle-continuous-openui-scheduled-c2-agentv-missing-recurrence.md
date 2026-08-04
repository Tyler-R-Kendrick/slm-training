# Autotrain c2 (continuous-openui-scheduled): AgentV SDK missing recurrence — self-healed

**Verdict:** infrastructure failure, not scoreable — and now self-healing.
Training completed for both the `control` and `canvas` `wf_smoke_v2` arms
(1,608,962 params, 22 steps, loss `14.390223503112793` for both, checkpoints
`1bc6370f...9286e` control / `9f73b7a8...053b1a4` canvas, local explicit
no-sync), but `evaluate_model.py --ship-gates` crashed before producing a
scoreboard for either arm: `RuntimeError: AgentV SDK is unavailable; run npm
ci in the checkout or set AGENTV_RUNNER`. Neither arm has smoke metrics; this
is not evidence about the model.

This is a **recurrence** of
[`autotrain-cycle-c2-agentv-missing-infra-failure.md`](autotrain-cycle-c2-agentv-missing-infra-failure.md):
the underlying fix (`npm ci` with sanitized `NODE_OPTIONS`) already existed in
`scripts/setup_dev_env.sh`, but this cycle ran in a fresh checkout (a
scheduled-routine sandbox) that invoked `scripts/run_autotrain_continuous.py`
directly without a human first running the bootstrap script. The driver had
no self-heal path for this and would repeat the same infra failure on every
fresh checkout that skips manual setup.

## Repair (commit `1c48ac95d89f53595a2776359e1a3c1d5ec0cc25`)

Added `_ensure_agentv_sdk()` to `scripts/run_autotrain_continuous.py`, called
at the top of `run_cycle` before any experiment runs. It checks
`node_modules/@agentv/core/package.json` across
`slm_training.bridge_utils.checkout_roots()`, and if missing, self-heals by
running `npm ci` with `sanitized_node_env()` — the same fix
`scripts/setup_dev_env.sh` applies for humans. If `npm ci` itself fails, the
cycle fails closed with a clear `SELF_HEAL_AGENTV_SDK_FAILED` error rather
than a confusing downstream AgentV traceback.

Regression tests: `test_ensure_agentv_sdk_noop_when_already_installed`,
`test_ensure_agentv_sdk_self_heals_via_npm_ci`,
`test_ensure_agentv_sdk_raises_when_npm_ci_fails` in
`tests/test_scripts/test_run_autotrain_continuous.py`.

Version stamp: `harness.autoresearch.experiment_campaign` v177 → v178.

No scoreboard, no smoke metrics, no ship-gate result exists for this cycle;
the checkpoints are local, explicit no-sync, and not reusable, promotable, or
ship evidence. Lean is `not_applicable:screening`.

Next: replay the identical frozen `c2` arms (`retry_measurement`) now that
the AgentV SDK self-heal is in place and this sandbox has an installed SDK.

Machine evidence:
[`autotrain-cycle-continuous-openui-scheduled-c2-agentv-missing-recurrence.json`](autotrain-cycle-continuous-openui-scheduled-c2-agentv-missing-recurrence.json).
