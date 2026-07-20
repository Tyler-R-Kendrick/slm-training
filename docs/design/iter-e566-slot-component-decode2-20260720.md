# E566 — slot-component decode weight 2

E566 reuses E561's checkpoint and changes only learned slot-component decode
weight from 4 to 2. The head remains active at 16 applications and 14 choice
changes.

On matched OOD `n=4`, every quality aggregate is identical to E561: fidelity
0.5750, structure 0.2419, component recall 0.1458, reward 0.5753, AST-node F1
0.3125, and AST-edge F1 0.0385. Meaning-v1/v2 remain 0, and AgentV remains
0/1. No checkpoint was created.

**Verdict:** weights 2–4 occupy the same saturated selection regime. Retain
the negative evidence, do not promote, and use weight 0 for the decisive head
on/off ablation. Evidence:
[JSON](iter-e566-slot-component-decode2-20260720.json).
