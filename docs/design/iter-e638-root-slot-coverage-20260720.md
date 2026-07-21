# E638 — root slot-coverage gate

Date: 2026-07-20
Status: completed negative; reverted; not ship

E637 left Gallery missing three visible roles after producing a valid
three-slot ImageGallery. E638 required complete slot-contract coverage before
semantic-plan root closure, testing whether root abstention alone would allow
the missing hint and CTA roles to be emitted.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. One clean CPU evaluation reused
E620's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`,
with the exact E637 OOD `n=4` recipe. It completed under the three-minute cap
with no timeout or fallback and emitted AgentEvals JSONL plus an AgentV SDK
bundle without execution errors.

## Measured result

| OOD `n=4` | E637 r2 baseline | E638 r1 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 / strict v2 | 0.7500 / 0.5000 | 0.7500 / 0.5000 |
| fidelity / validity | 0.6750 / 0.8050 | 0.5917 / 0.7550 |
| structure / component recall | 0.5817 / 0.6250 | 0.5500 / 0.6875 |
| reward | 0.8545 | 0.8190 |
| AST node / edge F1 | 0.6437 / 0.4554 | 0.6556 / 0.4554 |
| latency p50 / p95 | 2303.87 / 6831.74 ms | 3207.05 / 14401.40 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Gallery collapsed from `ImageGallery(img, alt, caption)` to only
`TextContent(hint.title)`. Modal and Auth remained strict-v2 clean, but the
unchanged 2/4 strict rate does not offset broad continuous-quality and latency
regressions.

## Decision

Reject the treatment stamped model v76 and restore the baseline behavior. After
rebasing over upstream model revisions v77-v78, the append-only lineage records
E638 as treatment v79 and restoration v80. Root abstention does not supply a
structurally compatible next family; a future Gallery lever must positively
select missing role-compatible sibling components instead of only preventing
closure. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e638-root-slot-coverage-20260720.json).
