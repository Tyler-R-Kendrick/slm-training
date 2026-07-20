# E570 — E569 component-plan decode weight 1

E570 reuses E569's checkpoint and enables its trained component-plan head at
decode weight 1. The head applies 131 times and changes three choices.

On matched OOD `n=4`, fidelity improves 0.2583→0.4917, structure
0.2031→0.3350, reward 0.6920→0.7695, AST-node F1 0.3389→0.3821, and
AST-edge F1 0→0.0455. Component recall falls 0.3333→0.2083 and
meaningful-v1 falls 0.25→0. Binding-aware meaningful-v2 remains 0, and
AgentV remains 0/1. No checkpoint was created.

**Verdict:** retain weight 1 as a topology/reward decode Pareto without
promotion. Test weight 0.5 for a coverage-quality compromise. Evidence:
[JSON](iter-e570-e569-component-plan1-20260720.json).
