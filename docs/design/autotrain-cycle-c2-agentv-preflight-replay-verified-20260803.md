# Autotrain c2 (continuous-openui-local): frozen-arm replay confirms the AgentV preflight repair, bounds vs control still an exact tie

**Verdict:** the harness repair from
[`autotrain-cycle-c1-agentv-missing-infra-preflight-20260803.md`](autotrain-cycle-c1-agentv-missing-infra-preflight-20260803.md)
is confirmed. `scripts.autoresearch`'s `retry_measurement` action replayed
the identical frozen `-bounds`/`-control` arms (`c20260803-continuous-openui-local-8c0b60dd-c2-{control,bounds}`,
reusing the c1/c2 checkpoints, SHAs `d2f2dc4b...c557e44b` /
`eb81529a...b224a2f`) and both completed train + `evaluate_model.py
--ship-gates` end to end with real AgentEvals bundles and zero execution
errors. No code change was needed for this replay — it is the receipt that
the c1 fix (installing `node_modules/@agentv/core`) actually restores a
working evaluation path, and that the new `missing_dev_env_prerequisites()`
preflight guard did not need to fire because the SDK was present.

## Result

`structural_similarity` is an exact tie: `0.0575` control vs `0.0575`
bounds (`meaningful_program_rate`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, and `reward_score` all `0.0` on both arms, `n=3`).
Ship gates fail on evidence volume (`insufficient_n actual=3 need>=20`,
plus the missing `held_out`/`adversarial`/`ood`/`rico_held` suites), not on
a quality regression.

Per the SDLC Phase A tradeoff rule, `fixture_insufficient_n_alone` is
**not** a positive result — the frozen `bounds` hypothesis is rejected as a
model result (byte-identical to control) and no stack layer opens for this
classification. The harness repair itself is the positive event for this
session (a proven executable unblock), already delivered in the prior
commit `9c58f8416d1c56b19d32a2bbf1166ff60cfea854`.

Next priority (rank 1 from the driver): test the distinct size-matched
`component-plan` quality hypothesis
(`c20260803-continuous-openui-local-8c0b60dd-c2-component-plan`).

Machine evidence:
[`autotrain-cycle-c2-agentv-preflight-replay-verified-20260803.json`](autotrain-cycle-c2-agentv-preflight-replay-verified-20260803.json).
