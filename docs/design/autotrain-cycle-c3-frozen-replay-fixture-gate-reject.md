# Autotrain c3 (continuous-openui-local): frozen replay reaches a real scoreboard, expected fixture gate reject — not ship

**Verdict:** expected fixture-scale ship-gate rejection, **not** a ship
result and **not positive** for SDLC Phase A (null primary-metric delta).
This is the `retry_measurement` replay of the frozen c2 arm
(`c20260803-continuous-openui-local-8c0b60dd-c2-{control,canvas}` →
`c20260803-continuous-openui-local-8c0b60dd-c3-{control,canvas}`), now that
the AgentV SDK gap from
[`autotrain-cycle-c2-agentv-missing-infra-failure-repro2.md`](autotrain-cycle-c2-agentv-missing-infra-failure-repro2.md)
is resolved.

Both arms (1,608,962 params, 22 steps, loss `14.3902`, `wf_smoke_v2`, 101
records) trained and this time reached a complete `scoreboard.json` via
`evaluate_model.py --ship-gates --honest-slot-contract
--slot-contract-constrained-decode`. Both tie exactly on every metric —
`smoke.structural_similarity=0.32667`, `meaningful_program_rate=0.0`,
`reward_score=0.0` — so the primary-metric delta is `0.0`. Ship gates reject
for the expected reasons at this scale: `smoke:insufficient_n actual=3
need>=20`, plus every quality threshold (`meaningful_program_rate`,
`structural_similarity`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, `placeholder_fidelity`, `reward_score`) below its bar,
and the `held_out` / `adversarial` / `ood` / `rico_held` suites are absent
(smoke-only fixture run). None of this is evidence against the model —
`fixture_insufficient_n` alone is explicitly not a stop condition for the
continuous loop.

SDLC Phase A: **not positive** — `primary_metric_null_or_worse` (control ==
candidate, `improvement=0.0`) and `fixture_insufficient_n_alone`. No stack
layer opened per the delivery gate; local commit + docs only.

Checkpoints (`1bc6370f...286e` control / `9f73b7a8...053b1a4` canvas) stay
local, explicit no-sync, not reusable/promotable/ship evidence.

Next (per the driver's ranked successor priorities): the size-matched
"component-plan" quality hypothesis
(`c20260803-continuous-openui-local-8c0b60dd-c3-component-plan`).

Machine evidence:
[`autotrain-cycle-c3-frozen-replay-fixture-gate-reject.json`](autotrain-cycle-c3-frozen-replay-fixture-gate-reject.json).
