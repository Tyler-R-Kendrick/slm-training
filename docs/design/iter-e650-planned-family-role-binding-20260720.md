# E650 — planned-family visible-role binding

Date: 2026-07-20
Status: completed positive scratch result; retained; not ship

E650 binds compatible visible roles to component families already required by
the prompt-derived semantic plan before inferring any additional family. It
also recognizes `refresh` as an action-label role. This turns the plan's one
Button into a concrete refresh obligation without changing its cardinality.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E648 policy. It
completed without timeout or fallback and emitted AgentEvals JSONL plus an
AgentV SDK bundle without execution errors.

| OOD `n=4` | E648 baseline | E650 r1 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 1.0000 / 0.7500 | 1.0000 / 0.7500 |
| fidelity / validity | 0.9500 / 0.9700 | 1.0000 / 1.0000 |
| structure / component recall | 0.7355 / 0.8750 | 0.7355 / 0.8750 |
| reward | 0.9640 | 0.9790 |
| AST node / edge F1 | 0.7987 / 0.5798 | 0.7987 / 0.5798 |
| latency p50 / p95 | 2750.02 / 7705.93 ms | 2844.86 / 7879.73 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Dashboard now emits `Button(":ood.dash.refresh")` as a direct section. Its
other sections and all three other predictions retain E648's structure, so the
fix closes the missing-slot gap without the nested component and trivial-layout
regression seen in E649. Retain the treatment stamped v96. After rebasing onto
E649 restoration v98, the append-only lineage records the retained E650
behavior as v99. Strict v2 remains 3/4 because Dashboard still places metric values in schema-incompatible nested
properties; the `n=4` diagnostic and failed AgentV evidence floor are not ship
evidence. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e650-planned-family-role-binding-20260720.json).
