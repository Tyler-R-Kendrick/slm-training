# E562 — component-plan decode weight 1

E562 reuses E561's checkpoint and changes only component-plan decode weight
from 0 to 1. The head applies 136 times and changes five choices.

On matched OOD `n=4`, fidelity improves 0.5750→0.7417, structure
0.2419→0.2732, and AST-node F1 0.3125→0.3236. Component recall stays 0.1458.
Meaning-v1/v2 remain 0, reward falls 0.5753→0.3985, and AgentV remains 0/1.
No checkpoint was created.

**Verdict:** reject weight 1 as a semantic fix. Preserve the non-semantic
decode evidence and test weight 0.5 in E563. Evidence:
[JSON](iter-e562-component-plan-decode1-20260720.json).
