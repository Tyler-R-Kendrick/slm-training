# E631 — frame-aware incompatible-owner escape

Date: 2026-07-20
Status: completed positive mixed; default-off scratch policy; not ship

E630 proved that selecting a correct component family at the wrong structural
depth can be worse than broad schema matching. E631 changes the existing
default-off slot-coverage closure policy only when the active component owner
cannot own any missing visible role: it prefers legal closure instead of
nesting a different role-compatible component. Structural child lists and
object-property continuation retain E621 behavior.

## Reused checkpoint and recipe

No training ran and no checkpoint was created. The clean CPU eval reused
E620's rejected local-only checkpoint, SHA-256
`3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`,
with the exact E622/E621 r2 OOD `n=4` recipe. It completed under the
three-minute cap with no timeout or fallback and emitted AgentEvals JSONL plus
an AgentV SDK bundle without execution errors.

## Measured result

| OOD `n=4` | E622 baseline | E631 treatment |
| --- | ---: | ---: |
| syntax parse | 1.0000 | 1.0000 |
| meaningful v1 | 0.7500 | 0.7500 |
| strict meaning v2 | 0.0000 | 0.0000 |
| v2 judgment coverage | 1.0000 | 1.0000 |
| placeholder fidelity | 0.5917 | 0.6750 |
| placeholder validity | 0.7550 | 0.8050 |
| structural similarity | 0.4029 | 0.5729 |
| component recall | 0.5000 | 0.6250 |
| reward | 0.8175 | 0.8515 |
| AST node / edge F1 | 0.4690 / 0.2625 | 0.6357 / 0.5125 |
| latency p50 / p95 | 3067.47 / 6277.99 ms | 3025.78 / 6394.56 ms |
| closure applications / changes | 11 / 8 | 25 / 12 |
| AgentV | 0/1 | 0/1 |

The strongest result is Auth:
`Stack([Button, Input, Input], "column")` now has perfect placeholder fidelity,
validity, structural similarity, and component recall. The owner escape closes
Button instead of nesting SwitchGroup/Input, after which existing semantic-plan
inventory creates two sibling Inputs and the Stack root. Modal also removes the
spurious TextContent size token and keeps complete coverage. Dashboard and
Gallery remain unchanged.

Strict v2 remains zero. Auth places name/email slots in both `Input.name` and
`Input.placeholder`; the evaluator flags the `Input.name` uses as semantic-role
mismatches. Modal still has title/children/size role judgments, while Dashboard
and Gallery remain incomplete. The aggregate gains therefore do not establish
ship readiness.

## Decision

Retain model v65 as a default-off scratch policy. Do not sync, promote, or make
a ship claim. The next experiment should resolve Input property-role assignment
against the gold/evaluator contract before adding more inventory pressure:
prefer literal control names/types and place visible form-field content slots in
the schema property justified by the authored prompt. Dashboard root inventory
and Gallery hint/CTA coverage remain separate follow-ups.

Evidence: [JSON](iter-e631-frame-aware-owner-escape-20260720.json).
