# E646 — root slot-bearing references

Date: 2026-07-20
Status: completed neutral; reverted; not ship

E646 tracked the visible slots contained by every completed top-level section.
During verifier-checked semantic root construction, it added an unplanned
section reference only when that section covered a slot absent from the planned
references.

The capped run was originally launched as E640. It was renumbered E646 when
the rebase incorporated the independently landed E639–E644 sequence; the
committed JSON retains the original run ID explicitly.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E637 policy. It
completed without timeout or fallback and emitted AgentEvals JSONL plus an
AgentV SDK bundle without execution errors.

| OOD `n=4` | E637 r2 baseline | E646 r1 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 0.7500 / 0.5000 | 0.7500 / 0.5000 |
| fidelity / validity | 0.6750 / 0.8050 | 0.6750 / 0.8050 |
| structure / component recall | 0.5817 / 0.6250 | 0.5817 / 0.6250 |
| reward | 0.8545 | 0.8545 |
| AST node / edge F1 | 0.6437 / 0.4554 | 0.6437 / 0.4554 |
| latency p50 / p95 | 2303.87 / 6831.74 ms | 2185.94 / 7137.66 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Every prediction and quality metric matched E637. The missing Gallery roles
therefore are not present in completed supplemental sections when the verified
root is built. Reject the treatment stamped v80 and restore the baseline
behavior. After rebasing onto E645 restoration v83, the append-only lineage
records E646 as treatment v84 and restoration v85. The next lever must improve
coverage inside the planned ImageGallery subtree. No checkpoint was created,
synced, or promoted.

Evidence: [JSON](iter-e646-root-slot-references-20260720.json).
