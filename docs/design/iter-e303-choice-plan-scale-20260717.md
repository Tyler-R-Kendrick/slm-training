# E303 component-plan scale check (2026-07-17)

E303 repeats E302 with the same E293 checkpoint and E301 concise connected
policy, changing only `component_plan_decode_weight` from 1 to 4. Effective
weights are now persisted in every suite's evaluation policy.

The complete quality board remains exactly equal to E301/E302: parse 1.0,
seven failed thresholds, and AgentV 2/5. Plan applications/choice changes are
3/0 smoke, 5/0 held-out, 4/0 adversarial, 4/0 OOD, and 40/7 RICO. Higher scale
therefore changes more RICO decisions but no aggregate metric and still never
changes the component choice on the four broader suites.

**Verdict:** stop the decode-weight sweep. Component-plan ranking, training
coverage, or targets—not logit scale—is the bottleneck. No promotion or ship
claim.

Artifacts:

- `outputs/runs/e303-choice-plan4-connected-close-honest-r1/`
- [machine-readable result](choice-plan-scale-results-iter-e303-20260717.json)
