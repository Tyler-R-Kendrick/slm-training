# E538 — semantic-role plus component-plan composition

E538 tests whether the previously useful component-plan decode head composes
with E536's visible semantic-role policy on the E531 checkpoint. It holds the
E536 OOD four-record diagnostic recipe fixed and adds only
`component_plan_decode_weight=4`.

The CPU evaluation completed in 28.6 seconds under the external 170-second cap
from clean commit `50a1823`. It emitted AgentEvals JSONL, a pinned AgentV bundle,
and `choice_decision_trace/v1` evidence without execution errors.

| Metric | E536 role 4 | E538 role 4 + plan 4 | Delta |
| --- | ---: | ---: | ---: |
| Syntax parse rate | 1.0000 | 1.0000 | 0.0000 |
| Meaningful program rate | 0.2500 | 0.0000 | -0.2500 |
| Placeholder fidelity | 1.0000 | 0.8500 | -0.1500 |
| Structural similarity | 0.1959 | 0.1079 | -0.0881 |
| Component type recall | 0.5417 | 0.2708 | -0.2708 |
| Reward | 0.7403 | 0.0000 | -0.7403 |
| AST node F1 | 0.1627 | 0.2664 | +0.1037 |
| AST edge F1 | 0.0417 | 0.1124 | +0.0708 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0 / 1 | 0 / 1 | unchanged |

The lever was reachable: the plan head applied at 95 decisions and changed four
component choices. All four outputs became a single inline `Stack` root,
eliminating the generated declarations and reference reuse visible in E536.
Although local AST overlap rose, every output remained trivial, placeholder
spam affected all four records, and semantic-role mismatch remained universal.

Reject the composition and do not train it. The older component-plan benefit
does not transfer to E531 under the direct visible-role policy. The next lever
must plan declaration/reference structure explicitly instead of adding another
component-type bias. Machine-readable evidence is in
[the E538 JSON](iter-e538-role-plan-composition-20260719.json).
