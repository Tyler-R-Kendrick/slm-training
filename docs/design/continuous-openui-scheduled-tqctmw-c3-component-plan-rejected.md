# Autotrain c3 (continuous-openui-scheduled-tqctmw): component-plan candidate rejected (non-positive)

**Verdict:** honest fixture reject; candidate regresses vs matched control.
Non-positive, no stack layer.

Machine evidence:
[`continuous-openui-scheduled-tqctmw-c3-component-plan-rejected.json`](continuous-openui-scheduled-tqctmw-c3-component-plan-rejected.json).

## Recipe

`train_version=wf_smoke_v2`, size-matched control vs `component-plan`
candidate, seed 100003, 1,755,764 trainable params each,
`evaluate_model.py --ship-gates --suites smoke`. Both arms completed
training and evaluation cleanly (SDK self-heal from
[`continuous-openui-scheduled-tqctmw-c1-agentv-provisioning-fixed.md`](continuous-openui-scheduled-tqctmw-c1-agentv-provisioning-fixed.md)
continues to hold).

## Result

| Arm | structural_similarity | meaningful_program_rate | binder_reference_f1 |
| --- | --- | --- | --- |
| control | 0.2308 | 0.333 | 0.733 |
| component-plan (candidate) | 0.1725 | 0.0 | 0.633 |

`primary_metric_null_or_worse`: `smoke.structural_similarity` moves
`0.2308 -> 0.1725` (`improvement=-0.0583`), and `binder_reference_f1`
regresses `0.7333 -> 0.6333` (`non_regression_fail`). Both arms still fail
`smoke:insufficient_n` (`n=3` vs `>=20`) at this fixture scale. This falsifies
the `component-plan` hypothesis at this seed; per repository law a rejected
experiment closes this approach, not the underlying quality goal.

## Next

Handoff ranks a distinct size-matched `component-edge` quality hypothesis
next.
