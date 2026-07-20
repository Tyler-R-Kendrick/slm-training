# E564 — semantic-role decode weight 2

E564 reuses E561's checkpoint and changes only semantic-role decode weight
from 4 to 2. On matched OOD `n=4`, every quality aggregate is identical to
E561: fidelity 0.5750, structure 0.2419, component recall 0.1458, reward
0.5753, AST-node F1 0.3125, and AST-edge F1 0.0385.

Meaning-v1/v2 remain 0, and AgentV remains 0/1. The role bias shares the
slot-component choice path, and halving it changed no selected-output
aggregate. No checkpoint was created.

**Verdict:** retain as no-effect negative evidence, do not promote, and use
weight 0 for the decisive on/off ablation. Evidence:
[JSON](iter-e564-semantic-role-decode2-20260720.json).
