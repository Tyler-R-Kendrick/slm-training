# E648 — root-only inferred role plans

Date: 2026-07-20
Status: completed negative; reverted; not ship

E648 retained E647's schema-derived, house-style-preferred inferred families but
allowed their semantic-plan sampling bias only at completed top-level section
boundaries.

The capped run was originally launched as E642. It was renumbered E648 when
the rebase incorporated the independently landed E639–E644 sequence; the
committed JSON retains the original run ID explicitly.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E637 policy. It
completed without timeout or fallback and emitted AgentEvals JSONL plus an
AgentV SDK bundle without execution errors.

| OOD `n=4` | E637 r2 baseline | E648 r1 |
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
stamped v84. After rebasing onto E647 restoration v87, the append-only lineage
records E648 as treatment v88 and restoration v89. Merely changing the sampling
boundary does not bind inferred families to roles. No checkpoint was created,
synced, or promoted.

Evidence: [JSON](iter-e648-root-only-role-plans-20260720.json).
