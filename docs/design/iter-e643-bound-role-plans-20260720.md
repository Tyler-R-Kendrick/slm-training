# E643 — required-string role binding

Date: 2026-07-20
Status: completed negative; reverted; not ship

E643 paired E641's visible-role-inferred semantic-plan families with direct
role-compatible slot bias at required plain-string content arguments. A
plain-string argument immediately preceding an explicit OpenUI placeholder
argument remained excluded, preserving operational literals such as
`Input.name`.

No training ran and no checkpoint was created. One capped CPU OOD `n=4` run
reused E620's rejected local-only checkpoint with the exact E637 policy. It
completed in 25.4 seconds without timeout or fallback and emitted AgentEvals
JSONL plus an AgentV SDK bundle without execution errors.

| OOD `n=4` | E637 r2 baseline | E643 r1 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 0.7500 / 0.5000 | 0.7500 / 0.2500 |
| fidelity / validity | 0.6750 / 0.8050 | 0.7583 / 0.8550 |
| structure / component recall | 0.5817 / 0.6250 | 0.5406 / 0.6875 |
| reward | 0.8545 | 0.8840 |
| AST node / edge F1 | 0.6437 / 0.4554 | 0.6306 / 0.4750 |
| latency p50 / p95 | 2303.87 / 6831.74 ms | 3399.88 / 14044.45 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The result exactly reproduced E641's quality profile apart from runtime noise.
Direct argument compatibility did not make family-cardinality inference
slot-specific: two predictions still had semantic-role mismatch, placeholder
spam, and required-placeholder loss. Reject the treatment stamped v86. After
rebasing onto E642 restoration v88, the append-only lineage records E643 as
treatment v89 and restoration v90. A future lever must plan concrete `(family,
role)` obligations rather than
infer family counts and bind them independently. No checkpoint was created,
synced, or promoted.

Evidence: [JSON](iter-e643-bound-role-plans-20260720.json).
