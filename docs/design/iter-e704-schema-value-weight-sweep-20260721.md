# E704 — schema-value weight sweep

Date: 2026-07-21  
Status: completed negative; rejected; not ship

E704 tests whether increasing the existing schema-value penalty can stop Rico
body placeholders from occupying enum-valued fields. The representative weight-8
arm evaluated all five committed suites (`n=19`) under constrained-only decode,
with no timeout or fallback, and emitted AgentEvals plus an AgentV SDK bundle.
A bounded Rico-only sweep covered weights 5, 6, and 7 with the same policy.

Weights 5–8 all produce the same Rico result. They remove every
`TextContent.size` semantic-role mismatch but drop one required placeholder per
record instead. Strict meaning stays 0.0, while fidelity falls 0.9583→0.8750,
validity 0.9750→0.9250, and reward 0.9875→0.9625. Structure, component recall,
and AST F1 are unchanged; Smoke, Held-out, adversarial, and OOD tracked quality
metrics are also unchanged in the full weight-8 replay.

Reject weights 5–8 and retain weight 4. The missing capability is routing a
rejected enum-field slot into another compatible content carrier, not applying
a stronger scalar penalty. AgentV remains 0/5 and the Rico length budget still
fails. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e704-schema-value-weight-sweep-20260721.json).
