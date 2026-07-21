# E641 — visible-role semantic-plan completion

Date: 2026-07-20
Status: completed mixed negative; reverted; not ship

E641 filled visible roles that no prompt-planned family could own using the
first schema-compatible component from the repository's house-style preference
order. Gallery therefore gained planned TextContent and Button leaves without
hard-coding Gallery-specific component names.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E637 policy. It
completed without timeout or fallback and emitted AgentEvals JSONL plus an
AgentV SDK bundle without execution errors.

| OOD `n=4` | E637 r2 baseline | E641 r1 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 0.7500 / 0.5000 | 0.7500 / 0.2500 |
| fidelity / validity | 0.6750 / 0.8050 | 0.7583 / 0.8550 |
| structure / component recall | 0.5817 / 0.6250 | 0.5406 / 0.6875 |
| reward | 0.8545 | 0.8840 |
| AST node / edge F1 | 0.6437 / 0.4554 | 0.6306 / 0.4750 |
| latency p50 / p95 | 2303.87 / 6831.74 ms | 3263.87 / 14511.96 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Coverage improved, but Gallery spammed nested Button arguments, Modal duplicated
title/confirm roles, strict v2 fell from 2/4 to 1/4, and tail latency more than
doubled. Reject the treatment stamped v82 and restore the baseline behavior.
After rebasing onto E640 restoration v84, the append-only lineage records E641
as treatment v85 and restoration v86. A future role-plan lever must bind each
inferred family to its specific uncovered slot rather than only increasing
family cardinality. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e641-role-plan-completion-20260720.json).
