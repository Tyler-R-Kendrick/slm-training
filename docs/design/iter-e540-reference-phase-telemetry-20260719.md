# E540 — reference phase telemetry

E540 upgrades bounded choice evidence to `choice_decision_trace/v2`. Each legal
reference decision now records an eight-frame maximum generated-state path and
classifies its aggregation scope. Active reference-bias applications also reuse
the existing bounded constrained-selection trace to persist the legal
counterfactual choice before and after the bias. The trace uses generated state
and legal candidates only; it has no access to the gold reference graph.

The CPU OOD four-record replay completed in about 31 seconds under the external
170-second cap from clean commit `c61a4b9`. It emitted AgentEvals JSONL and a
pinned AgentV bundle without execution errors. Every quality metric and both
reference-bias counters exactly reproduce E539.

| Aggregation scope | Applications | Changed choices |
| --- | ---: | ---: |
| Structural root list | 9 | 6 |
| Structural nested list | 1 | 1 |
| Total | 10 | 7 |

The result corrects the broad E539 diagnosis: most intervention activity is
already in a terminal root list. Only one changed choice occurs in a nested
`Modal` list. A root-list-only policy is therefore the smallest honest next
intervention, but it may remove the modal placeholder insertion that contributed
to E539's fidelity gain.

**Verdict:** accept the bounded telemetry as a harness improvement with no
model-quality claim. Do not promote or train reference weight 4. Test the
root-list-only criterion against this exact replay next. Machine-readable
evidence:
[JSON](iter-e540-reference-phase-telemetry-20260719.json).
