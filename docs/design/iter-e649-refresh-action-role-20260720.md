# E649 — refresh action semantic-role alias

Date: 2026-07-20
Status: completed negative scratch result; reverted; not ship

E649 mapped the visible `refresh` role to the existing `action` and `label`
schema properties. The intent was to bind Dashboard's prompt-planned Button to
`:ood.dash.refresh` instead of letting that component consume status body.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E648 policy. It
completed without timeout or fallback and emitted AgentEvals JSONL plus an
AgentV SDK bundle without execution errors.

| OOD `n=4` | E648 baseline | E649 r1 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 1.0000 / 0.7500 | 0.7500 / 0.7500 |
| fidelity / validity | 0.9500 / 0.9700 | 1.0000 / 1.0000 |
| structure / component recall | 0.7355 / 0.8750 | 0.7512 / 0.8750 |
| reward | 0.9640 | 0.9790 |
| AST node / edge F1 | 0.7987 / 0.5798 | 0.8097 / 0.5893 |
| latency p50 / p95 | 2750.02 / 7705.93 ms | 2900.60 / 7572.60 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Coverage improved, but the Dashboard output became qualitatively worse: it
nested the refresh Button inside a duplicate status-body Button, placed metric
content in incompatible Callout/Carousel positions, and emitted an empty Card.
The meaningful-v1 judge correctly flags the result as a trivial layout, while
strict v2 remains 3/4. Reject the treatment stamped v94. After rebasing onto
retained E648 v96, the append-only lineage records E649 as treatment v97 and
restoration v98. Future work must bind an already-planned family to one compatible slot without increasing
or nesting the family. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e649-refresh-action-role-20260720.json).
