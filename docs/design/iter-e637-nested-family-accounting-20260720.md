# E637 — nested semantic-plan family accounting

Date: 2026-07-20
Status: completed positive mixed; confirmed; default-off; not ship

E636 made Modal schema-valid but emitted the confirm Button twice: once through
the required `Buttons` wrapper and once as a root sibling. Semantic-plan missing
family scoring and verified root closure counted only completed top-level
sections, so nested descendants were invisible. E637 counts emitted component
families from the choice prefix at every nesting depth while retaining root
references to completed top-level sections only.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. Two clean CPU evaluations reused
E620's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`,
with the exact E636 OOD `n=4` recipe. Both completed under the three-minute cap
with no timeout or fallback and emitted AgentEvals JSONL plus AgentV SDK bundles
without execution errors. Predictions and all non-latency metrics are
byte-identical across r1 and r2.

## Measured result

| OOD `n=4` | E635 r2 | E636 r2 | E637 r2 |
| --- | ---: | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.7500 | 0.7500 |
| strict meaning v2 | 0.2500 | 0.5000 | 0.5000 |
| v2 judgment coverage | 1.0000 | 1.0000 | 1.0000 |
| fidelity / validity | 0.6750 / 0.8050 | 0.6750 / 0.8050 | 0.6750 / 0.8050 |
| structure / component recall | 0.5729 / 0.6250 | 0.5459 / 0.6250 | 0.5817 / 0.6250 |
| reward | 0.8515 | 0.8575 | 0.8545 |
| AST node / edge F1 | 0.6357 / 0.5125 | 0.6214 / 0.4375 | 0.6437 / 0.4554 |
| latency p50 / p95 | 2222.19 / 6095.12 ms | 2467.64 / 6821.42 ms | 2303.87 / 6831.74 ms |
| timeout / fallback | 0 / 0 | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 | 0/1 |

Modal keeps its legal wrapper and loses the duplicate root action:

```openui
root = Stack([v0], "column")
v0 = Modal(":ood.modal.title", false, [TextContent(":ood.modal.body"), Buttons([Button(":ood.modal.confirm")])])
```

Modal remains strict-v2 clean, its structure rises from 0.6167 to 0.7600, and
Auth remains byte-identical and strict-v2 clean. Dashboard and Gallery remain
unchanged.

## Decision

Retain model v75 as default-off positive mixed evidence. It preserves E636's
2/4 strict rate, removes the known duplicate, lifts structure above both E635
and E636, and gives the best AST node F1 of the three. Reward stays above E635,
but edge F1 remains below E635; AgentV and the partial suite still fail. No
checkpoint was created, synced, or promoted. The next experiment should target
Gallery's missing required placeholder without disturbing Modal or Auth.

Evidence: [authoritative r2 JSON](iter-e637-nested-family-accounting-20260720.json)
and [r1 JSON](iter-e637-nested-family-accounting-r1-20260720.json).
