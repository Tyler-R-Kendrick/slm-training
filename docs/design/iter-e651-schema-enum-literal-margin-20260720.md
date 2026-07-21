# E651 — schema enum-literal margin

Date: 2026-07-20
Status: completed negative scratch result; reverted; not ship

E651 generalized the existing schema-value bias into a margin for legal enum
literals or optional closure. This was intended to replace dynamic strings in
enum-valued properties without changing content-slot decisions.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E650 policy. It
completed without timeout or fallback and emitted AgentEvals JSONL plus an
AgentV SDK bundle without execution errors.

| OOD `n=4` | E650 baseline | E651 r1 |
| --- | ---: | ---: |
| meaningful v1 / strict v2 | 1.0000 / 0.7500 | 0.7500 / 0.7500 |
| fidelity / validity | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / component recall | 0.7355 / 0.8750 | 0.7629 / 0.8750 |
| reward | 0.9790 | 0.9790 |
| AST node / edge F1 | 0.7987 / 0.5798 | 0.8222 / 0.6003 |
| latency p50 / p95 | 2844.86 / 7879.73 ms | 2778.22 / 6448.99 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The Dashboard Callout now uses legal `"info"`, but the global margin perturbs
content allocation: metric value replaces status body in the Callout and the
second planned Card is empty. Meaningful v1 flags the result as trivial while
strict v2 does not improve. Reject the treatment stamped v97. After rebasing
onto retained E650 v99, the append-only lineage records E651 as treatment v100
and restoration v101. Enum validity must be sequenced after content-slot ownership, not compete with it.
No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e651-schema-enum-literal-margin-20260720.json).
