# Autotrain c2 (continuous-openui-scheduled-tqctmw): frozen replay confirms honest fixture reject

**Verdict:** measurement now complete (self-heal from c1 worked); non-positive.
Replays the identical frozen `wf_smoke_v2` control/bounds arms from cycle 1
(`continuous-loop-...-c1`), now that the AgentV SDK self-heals in this
sandbox per
[`continuous-openui-scheduled-tqctmw-c1-agentv-provisioning-fixed.md`](continuous-openui-scheduled-tqctmw-c1-agentv-provisioning-fixed.md).

Machine evidence:
[`continuous-openui-scheduled-tqctmw-c2-frozen-replay-results.json`](continuous-openui-scheduled-tqctmw-c2-frozen-replay-results.json).

## Recipe

- Host: CPU sandbox, local `.venv` (Python 3.12), `--local-files-only`.
- `train_version=wf_smoke_v2`, 20 steps requested, 1,608,962 trainable params,
  seed 100001, both arms size-matched.
- `evaluate_model.py --ship-gates --honest-slot-contract
  --slot-contract-constrained-decode --suites smoke`.

## Result

Both arms (`control`, `bounds`) now produce a complete `scoreboard.json` and
`gates.json` (previously blocked on the missing AgentV SDK). Ship gates
correctly **reject** on fixture scale:

| Arm | parse_rate | structural_similarity | meaningful_program_rate | gate |
| --- | --- | --- | --- | --- |
| control | 1.0 | 0.0575 | 0.0 | reject |
| bounds | 1.0 | 0.0575 | 0.0 | reject |

`smoke:insufficient_n` (`n=3` vs required `>=20`) alone would already fail
closed; `structural_similarity`, `meaningful_program_rate`,
`component_type_recall`, `ast_beq_rate`, `canonical_beq_rate`, and
`reward_score` also miss their thresholds. `held_out` / `adversarial` / `ood`
/ `rico_held` suites are not present at this fixture scale (`missing_suite`).
This is an honest **fixture-demo** result, not a ship claim — consistent with
`honest-ship-eval`: no gate was weakened to pass it.

Primary metric `smoke.structural_similarity` is an exact tie between control
and candidate (`0.0575` vs `0.0575`, `improvement=0.0`) — a null delta, not a
model win. Non-positive per `sdlc` autotrain-iteration-delivery
(`fixture_insufficient_n_alone` + `primary_metric_null_or_worse`); no stack
layer.

## Next

Handoff ranks a size-matched `component-plan` quality hypothesis
(`c...-c2-component-plan`) as the next candidate arm.
