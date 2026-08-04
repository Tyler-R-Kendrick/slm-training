# Autotrain c1 (continuous-openui-local, 2026-08-04): AgentV SDK missing again in a fresh container, not a model result

**Verdict:** infrastructure failure, not scoreable. Training completed for both
the control and `-bounds` `wf_smoke_v2` arms (1,608,962 params, 21 steps,
loss `22.6219` for both, checkpoints `d2f2dc4b...c557e44b` control /
`eb81529a...25b224a2` bounds, local explicit no-sync), but
`evaluate_model.py --ship-gates` crashed before producing a scoreboard for
either arm: `RuntimeError: AgentV SDK is unavailable; run npm ci in the
checkout or set AGENTV_RUNNER`. Neither arm has smoke metrics; this is not
evidence about the model.

This is the same failure class as
[cycle c2 on 2026-08-03](autotrain-cycle-c2-agentv-missing-infra-failure.md):
this session's container was a fresh checkout where `npm ci` had never been
run, so `node_modules/@agentv/core` did not exist. The c2 fix (adding
`npm ci` to `scripts/setup_dev_env.sh` and generalizing `NODE_OPTIONS`
sanitization into `bridge_utils.sanitized_node_env()`) was correct but did not
stop a *later* fresh container from hitting the identical bare-checkout gap,
because nothing in the continuous driver actually enforced the SDK
precondition before starting a cycle — the failure only surfaced deep inside
`--ship-gates` evaluation, after the training stage had already spent its
wall budget.

Fix, commit `b82fdf71`:

1. Added `scripts.run_autotrain_continuous._require_agentv_sdk_available`,
   called from `run_cycle` immediately after the clean-tree check. A missing
   SDK now fails the cycle in well under a second with the same actionable
   message, instead of burning a training wall budget first.
2. Investigating the same gap surfaced a second, independent bug:
   `src/slm_training/dsl/design_md/__init__.py`'s node bridge (DESIGN.md
   lint, used by `build_train_data`'s default-DESIGN.md attachment) never
   adopted `sanitized_node_env()` when the sibling `lang_core.py` /
   `graphql_js.py` / `evals/agentv.py` bridges did. A host `NODE_OPTIONS`
   value this Node build rejects (`--import tsx`) broke it silently — 8 tests
   were failing in `tests/test_scripts/test_build_train_data_cli.py` and
   `tests/test_scripts/test_slm_cli.py` before the fix, all passing after.
3. Environment setup itself (`npm ci` at the repo root, plus
   `src/apps/openui_bridge` and `src/apps/design_md_bridge`) was re-run per
   `scripts/setup_dev_env.sh` / the README Quick start for this session's
   fresh checkout — that step is not new code, it is the documented
   precondition the new preflight now enforces automatically on every future
   cycle.

Regression tests:
`tests/test_scripts/test_run_autotrain_continuous.py::test_require_agentv_sdk_available_fails_fast_without_node_modules`
and `..._passes_when_sdk_present`. `harness.autoresearch.experiment_campaign`
bumped v177 → v178 in `src/slm_training/resources/versions.json`.

No scoreboard, no smoke metrics, no ship-gate result exists for this cycle;
the checkpoints are local, explicit no-sync, and not reusable, promotable, or
ship evidence. Lean is `not_applicable:screening`. No checkpoint was promoted
or synced, so `docs/MODEL_CARD.md` / the README model-card summary are
unchanged.

Next: replay the identical frozen `-bounds` arm (`retry_measurement`, manifest
`47de63eca7855f4451ff8f6cf5decf10a9eb0ebd0976fd6c1dfe9f3682747920`) now that
the AgentV SDK is installed and preflighted.

Machine evidence:
[`autotrain-cycle-20260804-c1-agentv-npm-ci-infra-failure.json`](autotrain-cycle-20260804-c1-agentv-npm-ci-infra-failure.json).
