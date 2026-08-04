# Autotrain c1 (continuous-openui-scheduled-tqctmw): AgentV SDK missing in fresh sandbox, now self-heals

**Verdict:** infrastructure failure, not scoreable — same class of failure as
[`autotrain-cycle-c2-agentv-missing-infra-failure.md`](autotrain-cycle-c2-agentv-missing-infra-failure.md),
recurring because this is a *fresh ephemeral container* rather than a
regression. Training completed for both the control and `-bounds`
`wf_smoke_v2` arms (1,608,962 params, seed 100001, both `parse_rate=1.0`,
`structural_similarity=0.0575`), but `evaluate_model.py --ship-gates` crashed
before producing a scoreboard for either arm:
`RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set
AGENTV_RUNNER`. Neither arm has smoke metrics; this is not evidence about the
model. Lean is `not_applicable:screening`.

Machine evidence:
[`continuous-openui-scheduled-tqctmw-c1-agentv-provisioning-fixed.json`](continuous-openui-scheduled-tqctmw-c1-agentv-provisioning-fixed.json).

## Root cause

This sandbox checkout had never run `npm ci`, so `node_modules/@agentv/core`
did not exist. The generalized `NODE_OPTIONS` sanitization from
`72fdffaf` (#1360, `evals.agentv` v7) already covers the *subprocess*
environment correctly — it was never the bug here. The gap is that every
fresh ephemeral sandbox reproduces the exact same missing-`node_modules`
infra failure and burns a full documented train+eval cycle before an agent
notices and manually reruns `npm ci` per `scripts/setup_dev_env.sh`.

## Repair (harness, not model)

Commit `6968d13d` (`fix(autotrain): self-heal missing AgentV SDK in fresh
sandbox checkouts`, `evals.agentv` v7 → v8): `_agentv_runtime` now attempts
one bounded, `NODE_OPTIONS`-sanitized `npm ci` per process per checkout root
before raising, only when a `package.json` exists at that root and the SDK is
missing. Opt out with `AGENTV_NO_AUTO_PROVISION=1`. Any provisioning failure
(no network, no `npm`, timeout) falls through to the original honest
`RuntimeError` — this is a self-heal, not a gate weakening. Regression tests:
`tests/test_evals/test_agentv.py::test_agentv_runtime_auto_provisions_missing_sdk_via_npm_ci`
and `::test_agentv_runtime_auto_provision_disabled_by_env`.

## Delivery

Non-positive cycle (`measurement_incomplete`, `harness_failure`,
`primary_metric_null_or_worse` — see the JSON `reasons`): no stack layer per
`sdlc` autotrain-iteration-delivery. Local commit only for this repair; kept
on the loop's working branch.

Next: `retry_measurement` — replay the identical frozen `-bounds`/`-control`
arms now that SDK provisioning self-heals in this sandbox.
