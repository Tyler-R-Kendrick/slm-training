# E567 — slot-component decode weight 0

E567 reuses E561's checkpoint and changes only learned slot-component decode
weight from 4 to 0.

On matched OOD `n=4`, disabling the head drops fidelity 0.5750→0.5333,
structure 0.2419→0.2194, component recall 0.1458→0.0833, reward
0.5753→0.4110, and AST-node F1 0.3125→0.2292. AST-edge F1 remains 0.0385.
Meaning-v1/v2 remain 0, and AgentV remains 0/1. No checkpoint was created.

**Verdict:** the learned head materially helps non-semantic quality but is not
the missing semantic mechanism. Retain weight 4, close this ladder, and do
not promote. Evidence:
[JSON](iter-e567-slot-component-decode0-20260720.json).
