# Autotrain c1 (continuous-openui-local, 2026-08-03 container): AgentV SDK missing again — added a preflight guard

**Verdict:** infrastructure failure, not scoreable. Training completed for
both the control and `-bounds` `wf_smoke_v2` arms (21 steps, loss `22.6219`
for both, checkpoints `d2f2dc4b…e44b` control / `eb81529a…24a2f` bounds,
local explicit no-sync), but `evaluate_model.py --ship-gates` crashed before
producing a scoreboard for either arm:
`RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set
AGENTV_RUNNER`. Neither arm has smoke metrics; this is not evidence about
the model.

## This is a repeat of a known failure

Same root cause and same fix family as
[`autotrain-cycle-c2-agentv-missing-infra-failure.md`](autotrain-cycle-c2-agentv-missing-infra-failure.md):
a fresh checkout never ran `scripts/setup_dev_env.sh`, so
`node_modules/@agentv/core` didn't exist. The `NODE_OPTIONS` sanitization
fix from that incident (commit `72fdffaf054efee2f3dccc4bab0ca97c111072e1`,
`slm_training.bridge_utils.sanitized_node_env()`) was already present and
correct — nothing to change there. The only thing missing in this container
was the one-time `npm ci`, which was re-run manually
(`env -u NODE_OPTIONS npm ci`) alongside a Python 3.12 venv + editable
install to match `scripts/setup_dev_env.sh`.

## What's new this time: fail fast instead of re-training

Two prior cycles (documented in `autotrain-cycle-c2-agentv-missing-infra-failure.md`
and this one) both burned a full wall-capped cycle — training both arms to
completion — before discovering the SDK was missing at evaluation time. That
is a repeatable, checkable-in-advance condition, not something that needs a
live experiment to discover.

Added `missing_dev_env_prerequisites()` to
`scripts/run_autotrain_continuous.py`: before `main()` acquires the driver
lock, it checks `node_modules/@agentv/core/package.json` across
`bridge_utils.checkout_roots()` (this checkout and its Git common checkout).
If absent, the driver now exits 2 immediately with
`DEV_ENV_PREREQUISITES_MISSING agentv_sdk; run scripts/setup_dev_env.sh
(npm ci) before starting a cycle` instead of spending the cycle budget on a
doomed measurement. Regression tests:
`tests/test_scripts/test_run_autotrain_continuous.py::test_missing_dev_env_prerequisites_flags_absent_agentv_sdk`
and `..._clear_when_agentv_sdk_present`. Version-stamped
`harness.autoresearch.experiment_campaign` v170 → v171 (behavior change:
new fail-fast exit path).

No scoreboard, no smoke metrics, no ship-gate result exists for this cycle;
the checkpoints are local, explicit no-sync, and not reusable, promotable,
or ship evidence. Lean is `not_applicable:screening`.

Next: replay the identical frozen `-bounds` arm (`retry_measurement`) now
that the SDK is installed and the preflight guard would have caught this
before training if it weren't.

Machine evidence:
[`autotrain-cycle-c1-agentv-missing-infra-preflight-20260803.json`](autotrain-cycle-c1-agentv-missing-infra-preflight-20260803.json).
