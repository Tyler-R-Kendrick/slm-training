# Autotrain c3 (continuous-openui-20260803): component-plan hypothesis regresses quality

**Verdict:** rejected, quality regression. Fresh size-matched control/candidate
pair (1,755,760 params, 20 steps, `wf_smoke_v2`), testing rank-1 of the c2
driver priorities (a "component-plan" quality hypothesis). The candidate
regresses every held quality metric versus its matched control:
structural similarity `.2308 -> .1725`, meaningful program rate
`.3333 -> 0`, binder reference F1 `.7333 -> .6333`, component type recall
`.1667 -> 0`. Latency improves (`3988.50 -> 3801.13` ms p50) but that does
not offset the quality loss — not a positive tradeoff.

Fixture ship gates fail as expected at `n=3` (need `>=20`); `held_out`/
`adversarial`/`ood`/`rico_held` suites are not published for this fixture
recipe. SDLC Phase A classifies this cycle **non-positive**
(`non_regression_fail` + `primary_metric_null_or_worse`): no stacked PR,
local commit only.

Checkpoints `30b43c75...9274` (control) and `058f3226...562c5d`
(component-plan) are local, explicit no-sync, never reusable, promoted, or
ship evidence.

Next: test the distinct size-matched `component-edge` quality hypothesis
(`c3-component-edge`, rank 1 in the driver's speculative priorities), keeping
the matched control every cycle.

Machine evidence:
[`continuous-openui-20260803-c3-component-plan-regression.json`](continuous-openui-20260803-c3-component-plan-regression.json).
