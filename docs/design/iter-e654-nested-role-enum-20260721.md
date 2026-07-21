# E654 — nested ownership plus enum flooring

Date: 2026-07-21
Status: completed negative; reverted; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy plus enum-close weight 4. It emitted AgentEvals and AgentV,
with no timeout or fallback.

Versus E653, meaningful v1 (1.0000), strict v2 (0.7500), recall (0.8750), and
AST F1 hold, while p95 improves 7045.22→6512.43 ms. Fidelity falls
1.0000→0.9500, validity 1.0000→0.9700, structure 0.7692→0.7629, and reward
0.9730→0.9580. Reject v105 and restore E653 as v106: quality outranks latency.
No checkpoint was created, synced, or promoted; AgentV remained 0/1.

Evidence: [JSON](iter-e654-nested-role-enum-20260721.json).
