# Autotrain continuous-openui-local cycle 1: harness failure, no measurement

**Verdict:** infrastructure null, not a model result. Both 21-step CPU scratch
arms (`control`, `bounds`; 1,608,962 trainable params, `wf_smoke_v2`) trained
to completion, but `evaluate_model.py --ship-gates` failed on both before
producing a scoreboard:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

## Root cause

`publish_agentv_evaluation()` spawns `node scripts/run_agentv_eval.mjs`. The
checkout's gitignored `node_modules/` was never installed, and once installed
the sandbox's ambient `NODE_OPTIONS="--import tsx"` (left over from an
unrelated JS project) is rejected outright by any `node` child process — so
the AgentV SDK could not run even after `npm ci`. The same defect affects the
OpenUI DSL bridge (`dsl/lang_core.py`), which independently broke ~19
`tests/test_evals` tests via a second, unrelated bug: `oracle_scoring_replay
.build_fixture_records()` never projected `metric_gaming` archetypes' dotted
slot markers (e.g. `:card.title`) into the opaque `:slot_<n>` identities
`data/contract.py` requires — unlike `MetricGamingCase`, which already
applies that exact codec to the same archetypes.

## Repair

Fixed in [PR #1354](https://github.com/Tyler-R-Kendrick/slm-training/pull/1354)
(commit `bf83505`): a shared `node_subprocess_env()` helper strips
`NODE_OPTIONS` at both `node`-spawn call sites; an idempotent
`scripts/ensure_node_toolchain.sh` covers the separate `npm ci` prerequisite;
`build_fixture_records()` now projects markers the same way
`MetricGamingCase` does; two test fixtures' hardcoded dotted markers were
renamed to canonical ordinals; the semantic-floor-gate doc pair was
regenerated for the resulting `agentv_owner_sha256` change.

Neither checkpoint (`3783759c...373ba` control, `28c62e64...eefde` bounds) is
reusable, promotable, synced, or ship evidence — the measurement never
completed. AgentV bundles are incomplete, fixture gates fail, and Lean is
`not_applicable:screening`.

The next action is `retry_measurement`: replay the identical frozen
control/candidate arm (`frozen_manifest_sha256`
`7462dc61b1fd1023203cb6df61716a7a4136b39f4dbc52216fa9ceffc0d4c6dd`) now that
the harness is repaired, before any new model hypothesis.

Machine evidence:
[`autotrain-continuous-openui-local-c1-harness-failure.json`](autotrain-continuous-openui-local-c1-harness-failure.json).
