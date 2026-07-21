# E642 — root-only inferred role plans

Date: 2026-07-20
Status: completed negative; reverted; not ship

E642 retained E641's schema-derived, house-style-preferred inferred families but
allowed their semantic-plan sampling bias only at completed top-level section
boundaries.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E637 policy. It
completed without timeout or fallback and emitted AgentEvals JSONL plus an
AgentV SDK bundle without execution errors.

| OOD `n=4` | E637 r2 baseline | E642 r1 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 0.7500 / 0.5000 | 0.5000 / 0.5000 |
| fidelity / validity | 0.6750 / 0.8050 | 0.6750 / 0.8050 |
| structure / component recall | 0.5817 / 0.6250 | 0.4884 / 0.6250 |
| reward | 0.8545 | 0.8440 |
| AST node / edge F1 | 0.6437 / 0.4554 | 0.5331 / 0.3929 |
| latency p50 / p95 | 2303.87 / 6831.74 ms | 3342.87 / 36906.76 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Modal and Auth remained strict, but Gallery became a 1,290-character nested
Button/Carousel/Modal program and Dashboard lost meaning. Reject the treatment
stamped v84. After rebasing onto E641 restoration v86, the append-only lineage
records E642 as treatment v87 and restoration v88. Merely changing the sampling
boundary does not bind inferred families to roles. No checkpoint was created,
synced, or promoted.

Evidence: [JSON](iter-e642-root-only-role-plans-20260720.json).
