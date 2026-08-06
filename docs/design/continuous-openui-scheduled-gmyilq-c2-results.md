# Autotrain c2 (continuous-openui-scheduled-gmyilq): AgentV SDK false-positive reuse, not a model result

**Verdict:** infrastructure failure, not scoreable. Training completed for the
control `wf_smoke_v2` arm (1,608,962 trainable params), but
`scripts.evaluate_model --ship-gates` crashed before producing a scoreboard:

```
RuntimeError: AgentV SDK evaluation failed: node:internal/modules/esm/resolve:275
Error [ERR_MODULE_NOT_FOUND]: Cannot find module
'/home/user/slm-training/node_modules/typebox/build/typebox.mjs'
imported from '/home/user/slm-training/node_modules/typebox/build/index.mjs'
```

The canvas arm never ran (the campaign stopped after the control arm's
harness failure). Neither arm has smoke metrics; this is not evidence about
the model. Lean is `not_applicable:screening`.

## Root cause

This sandbox's checkout arrived with a `node_modules` that had
`@agentv/core/package.json` present — but a transitive dependency
(`typebox`) was missing its build output, so real imports failed. The
existing self-heal bootstrap in `_agentv_runtime`
(`src/slm_training/evals/agentv.py`, added in `evals.agentv` v8) only checked
whether `@agentv/core/package.json` existed before trusting the install and
skipping `npm ci`. A `package.json` file existing is not proof the whole
dependency tree resolves, so the drifted/stale install was never repaired.

## Fix (`evals.agentv` v9 -> v10, commit `9a53ea83c4265629f50bcb490f3abfb142059a8c`)

Cross-check `node_modules/.package-lock.json` (the marker npm writes only
when an install actually completed) against the checked-in
`package-lock.json` for every **non-optional** package before trusting an
existing SDK install. Optional platform-mismatched deps (e.g. `fsevents`,
darwin/win32-only binaries) are excluded from the comparison since npm
legitimately skips those on this linux-x64 sandbox — an earlier draft of
this fix that required exact package-set equality false-failed on those and
would have re-run `npm ci` (and therefore three of the pre-existing
integration tests, which exercise the real repo checkout) on every eval.

Added two regression tests to `tests/test_evals/test_agentv.py`:

- `test_agentv_runtime_bootstraps_when_sdk_present_but_lockfile_drifted` —
  reproduces the exact bug: `@agentv/core/package.json` present, lockfile
  marker missing, must still bootstrap.
- `test_agentv_runtime_tolerates_platform_skipped_optional_deps` — an
  optional dep the lockfile lists but npm skipped for this platform must
  not force a spurious re-bootstrap.

Also updated the four pre-existing runtime tests whose fixtures previously
only wrote `@agentv/core/package.json` without a matching lockfile marker,
since they now need one to represent a "genuinely installed" SDK. All 14
tests in `tests/test_evals/test_agentv.py` pass.

For this session, manually running `npm ci` at the repo root (and in
`src/apps/openui_bridge` / `src/apps/design_md_bridge`, per the continuous
loop's fresh-checkout prerequisite) also fixed the immediate environment —
that's a one-time local repair, not durable; the code fix above is what
prevents recurrence in any other sandbox with a similarly drifted install.

## Next

Replay the identical frozen control/canvas arms (`retry_measurement`) now
that the harness repair has a regression-tested fix landed.

Machine evidence:
[`continuous-openui-scheduled-gmyilq-c2-results.json`](continuous-openui-scheduled-gmyilq-c2-results.json).
