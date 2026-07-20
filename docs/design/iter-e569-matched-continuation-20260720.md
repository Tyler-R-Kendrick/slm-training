# E569 — matched E561 continuation

E569 warm-starts E561 for 48 steps with its no-design-metadata-context,
non-compiled recipe and threshold-7 twofold rare-owner sampler. It completes
in 75.20 seconds, sees 2,561 target tokens, ends at loss 12.1355, and writes
local checkpoint SHA `8254fcf7…c6535f73`.

On OOD `n=4`, meaningful-v1 rises 0→0.25, component recall
0.1458→0.3333, reward 0.5753→0.6920, and AST-node F1 0.3125→0.3389.
Fidelity falls 0.5750→0.2583, structure 0.2419→0.2031, and AST-edge F1
0.0385→0. Binding-aware meaningful-v2 remains 0 and AgentV remains 0/1.

**Verdict:** retain as a local semantic-coverage Pareto for targeted
strict-meaning research. Do not promote or sync. Evidence:
[JSON](iter-e569-matched-continuation-20260720.json).
