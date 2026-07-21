# E639 — root sibling coverage

Date: 2026-07-20
Status: completed neutral; reverted; not ship

E639 extended the existing visible-slot coverage bias to completed root-section
boundaries. While slots remained, it floored the best legal component whose
public schema could cover one of them.

## Runs and recipe

No training ran and no checkpoint was created. Both capped CPU runs reused
E620's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`,
and emitted AgentEvals JSONL plus AgentV SDK bundles without execution errors.
R1 completed but accidentally omitted E637's explicit compiler and semantic-plan
weights; it is retained as a non-comparable recipe diagnostic. R2 pinned the
exact E637 OOD `n=4` policy and is authoritative.

## Measured result

| OOD `n=4` | E637 r2 baseline | E639 r2 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 0.7500 / 0.5000 | 0.7500 / 0.5000 |
| fidelity / validity | 0.6750 / 0.8050 | 0.6750 / 0.8050 |
| structure / component recall | 0.5817 / 0.6250 | 0.5817 / 0.6250 |
| reward | 0.8545 | 0.8545 |
| AST node / edge F1 | 0.6437 / 0.4554 | 0.6437 / 0.4554 |
| latency p50 / p95 | 2303.87 / 6831.74 ms | 2206.60 / 6719.44 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

The new bias activated: Gallery traces selected additional TextContent siblings
for missing hint roles. The final output still matched E637 exactly because the
verified semantic root includes only prompt-planned family references and
discarded those extra sections.

## Decision

Reject the treatment stamped model v78 and restore the baseline behavior. After
rebasing onto the E638 production lineage at v80, the append-only lineage
records E639 as treatment v81 and restoration v82. Sampling extra sibling
sections without adding verifier-safe root references is neutral complexity.
The next lever should select supplemental, visible-slot-bearing section
references during verified root construction. No checkpoint was created,
synced, or promoted.

Evidence: [authoritative JSON](iter-e639-root-sibling-coverage-20260720.json)
and [r1 recipe diagnostic](iter-e639-root-sibling-coverage-r1-20260720.json).
