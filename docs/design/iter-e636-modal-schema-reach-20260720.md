# E636 — Modal schema reachability

Date: 2026-07-20
Status: completed positive mixed; default-off; not ship

E635 left Modal with three strict-v2 failures: its valid title placeholder was
incorrectly rejected by a hard-coded evaluator family rule, its Button was an
illegal direct `Modal.children` item, and the model filled optional `Modal.size`
with an invalid literal. E636 derives recognized role placement from public
schema properties, follows schema references through legal wrapper components,
enforces exact typed-array variants, and closes legal optional tails after all
visible slots are covered.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. Two clean CPU evaluations reused
E620's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`,
with the exact E635 OOD `n=4` recipe. Both completed under the three-minute cap
with no timeout or fallback and emitted AgentEvals JSONL plus AgentV SDK bundles
without execution errors. R1 exposed a metric 2.2.0 compatibility regression
for established form slots in `Input.placeholder`; metric 2.2.1 restored that
contract, and r2 is authoritative. Predictions are byte-identical across runs.

## Measured result

| OOD `n=4` | E635 r2 baseline | E636 r2 |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.7500 |
| strict meaning v2 | 0.2500 | 0.5000 |
| v2 judgment coverage | 1.0000 | 1.0000 |
| placeholder fidelity / validity | 0.6750 / 0.8050 | 0.6750 / 0.8050 |
| structure / component recall | 0.5729 / 0.6250 | 0.5459 / 0.6250 |
| reward | 0.8515 | 0.8575 |
| AST node / edge F1 | 0.6357 / 0.5125 | 0.6214 / 0.4375 |
| latency p50 / p95 | 2222.19 / 6095.12 ms | 2467.64 / 6821.42 ms |
| timeout / fallback | 0 / 0 | 0 / 0 |
| AgentV | 0/1 | 0/1 |

Modal becomes schema-valid and strict-v2 clean:

```openui
root = Stack([v0, v1], "column")
v0 = Modal(":ood.modal.title", false, [TextContent(":ood.modal.body"), Buttons([Button(":ood.modal.confirm")])])
v1 = Button(":ood.modal.confirm")
```

Its reward rises from 0.937 to 0.961, while Auth remains strict-v2 clean and
unchanged. The extra root-level Button duplicates the already nested confirm
action, causing the aggregate structure and AST-F1 regressions. Dashboard and
Gallery remain unresolved.

## Decision

Retain model v74 as default-off positive mixed evidence. The suite advances
from 1/4 to 2/4 strict passes and reward improves, but duplicate action emission
must be removed before promotion; AgentV and the partial suite still fail. No
checkpoint was created, synced, or promoted. The next experiment should make
semantic-plan family accounting include legal nested wrapper descendants so it
does not emit the same required family again at root scope.

Evidence: [authoritative r2 JSON](iter-e636-modal-schema-reach-20260720.json)
and [diagnostic r1 JSON](iter-e636-modal-schema-reach-r1-20260720.json).
