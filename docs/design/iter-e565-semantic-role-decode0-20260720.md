# E565 — semantic-role decode weight 0

E565 reuses E561's checkpoint, retains visible semantic-role context, and
changes only semantic-role decode weight from 4 to 0. On matched OOD `n=4`,
every quality aggregate and failure-reason prevalence is identical to E561
and the E564 midpoint.

Fidelity is 0.5750, structure 0.2419, component recall 0.1458, reward 0.5753,
AST-node F1 0.3125, and AST-edge F1 0.0385. Meaning-v1/v2 remain 0, and
AgentV remains 0/1. No checkpoint was created.

**Verdict:** close the semantic-role decode-weight ladder as inactive for
E561's selected outputs. Do not promote; move to a different semantic
mechanism. Evidence: [JSON](iter-e565-semantic-role-decode0-20260720.json).
