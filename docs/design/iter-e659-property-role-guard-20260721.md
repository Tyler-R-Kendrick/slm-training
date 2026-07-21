# E659 — final property-role candidate guard

Date: 2026-07-21
Status: completed negative; reverted; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy and no unconstrained fallback. It emitted AgentEvals and
AgentV with no timeout or fallback, after the treatment passed all 115 compiler
tests.

E659 hard-masks visible slots that do not match the active positional property,
after every positive plan and reference margin. The guard removes the invalid
Carousel metric, but also removes required visible content elsewhere: Gallery
becomes a long placeholder-free literal list. Versus E653, meaningful v1 falls
1.0000→0.7500, strict v2 0.7500→0.5000, fidelity 1.0000→0.6500, validity
1.0000→0.6900, structure 0.7692→0.6567, recall 0.8750→0.7500, reward
0.9730→0.6968, node F1 0.8222→0.6722, edge F1 0.7102→0.5852, and p95
7045.22→14215.63 ms. Reject v115 and restore E653 behavior as v116. The
current role map is not precise enough for hard exclusion. No checkpoint was
created, synced, or promoted; AgentV remained 0/1.

Evidence: [JSON](iter-e659-property-role-guard-20260721.json).
