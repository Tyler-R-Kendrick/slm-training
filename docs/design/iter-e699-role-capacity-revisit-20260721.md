# E699 — role capacity revisit

Date: 2026-07-21
Status: completed negative; reverted; not ship

E699 retests distinct public string-property capacity after E698 removes the
false Form owner. Three independently capped Held-out runs completed with exit
0, no timeout or fallback, and emitted AgentEvals JSONL plus AgentV bundles.

The capacity treatment does not create the expected Callout/TextContent split.
R1 collapses email+submit into RadioItem and moves display roles into an
incompatible Button/Callout. Fidelity reaches 1.0 and reward 0.9634, but strict
stays 4/5, structure falls to 0.7638, node F1 to 0.8431, and edge F1 to 0.6174.
R2 proves the generic form aliases independently preserve that bad joint-role
collapse. Both are rejected.

R3 removes capacity and the aliases, advances the candidate metric to the
restoration version 2.9.0, and is byte-identical to E698 r2: fidelity 0.96,
structure 0.7724, recall 0.8433, reward 0.9514, node F1 0.8609, and edge F1
0.6888. Retain E698 behavior as v163. This is scratch Held-out `n=5`, not ship
evidence; no checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e699-role-capacity-revisit-20260721.json).
