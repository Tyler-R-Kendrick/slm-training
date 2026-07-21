# E702 — joint role cardinality

Date: 2026-07-21  
Status: completed negative; reverted; not ship

E702 tested whether every repeated visible-role namespace should add another
direct joint carrier after the counted instances exhaust their compatible public
string properties. The matched five-suite scratch replay evaluated all 19
committed records under constrained-only decoding, with no timeout or fallback,
and emitted AgentEvals plus an AgentV SDK bundle.

The intervention fixed Rico placeholder fidelity and validity to 1.0 but did not
improve strict meaning. Instead, the additional Callouts became root-level plan
obligations, displaced the authored Cards, placed visible placeholders in
`Callout.variant`, and produced duplicate placeholder/subtree spam.

| Suite / metric | E701 v171 | E702 v172 |
| --- | ---: | ---: |
| Held-out strict v2 | 1.0000 | 0.8000 |
| Held-out structure / recall | 0.8104 / 0.8933 | 0.7615 / 0.8267 |
| Held-out AST node / edge F1 | 0.8831 / 0.7174 | 0.8152 / 0.6364 |
| Rico strict v2 | 0.0000 | 0.0000 |
| Rico fidelity / validity | 0.9583 / 0.9750 | 1.0000 / 1.0000 |
| Rico structure / recall | 0.7611 / 1.0000 | 0.4678 / 0.5556 |
| Rico AST node / edge F1 | 0.7942 / 0.7757 | 0.4365 / 0.3828 |

Smoke, adversarial, and OOD tracked quality metrics are unchanged. Reject v172
and restore v171 behavior as v173. The next lever must keep joint role carriers
nested beneath authored containers instead of counting them as additional root
families. AgentV remains 0/5 and the Rico 190-token p95 requirement exceeds the
160-token canvas. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e702-joint-role-cardinality-20260721.json).
