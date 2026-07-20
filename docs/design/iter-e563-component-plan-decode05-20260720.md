# E563 — component-plan decode weight 0.5

E563 reuses E561's checkpoint and changes only component-plan decode weight
from 0 to 0.5. The head applies 130 times and changes seven choices.

On matched OOD `n=4`, fidelity falls 0.5750→0.4083, structure
0.2419→0.2019, reward 0.5753→0.5178, and AST-node F1 0.3125→0.2500.
Component recall and AST-edge F1 remain 0.1458 and 0.0385. Meaning-v1/v2
remain 0, and AgentV remains 0/1. No checkpoint was created.

**Verdict:** reject weight 0.5 as a semantic fix and close the component-plan
decode-weight ladder. Evidence:
[JSON](iter-e563-component-plan-decode05-20260720.json).
