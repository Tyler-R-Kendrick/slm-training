# E581 — count-aware predicted components

Date: 2026-07-20  
Status: structural gain; not promotable or ship

E581 changes prompt-plan component scoring from a permanent family preference
to a remaining-instance preference. Once a generated family reaches its
authored count, it receives no further plan score. Missing repeated families
continue to receive the same soft score. Candidate legality is unchanged and
the mechanism remains default-off.

## Matched result

All arms use clean commit `f0780acd`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, honest visible
slot and semantic-role context, root weight 4, choice-codec constrained LTR,
8 generation steps, 4 attempts, and a 160-token canvas. Every process
completed under the 170-second hard cap. Stamps carry eval v16 and TwoTower
v18.

| Component weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | plan changes | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e581-e569-count-components0-r1` | 0.00 / 0.00 | 0.3417 / 0.6050 | 0.1250 | 0.1458 | 0.7095 | 0.1833 / 0.0000 | 0 | 0/1 |
| 1 | `e581-e569-count-components1-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 2 | 0/1 |
| 2 | `e581-e569-count-components2-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 2 | 0/1 |
| 4 | `e581-e569-count-components4-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.3231 | 0.4583 | 0.7480 | 0.4532 / 0.2000 | 4 | 0/1 |

Weight 4 improves every non-strict aggregate versus the no-plan control:
meaning-v1 +0.25, fidelity +0.0833, validity +0.05, structure +0.1981,
component recall +0.3125, reward +0.0385, AST-node F1 +0.2698, and AST-edge
F1 +0.20.

Strict meaning-v2 remains zero. Auth serializes a Stack with `v0` and `v2`,
but no `v1`; the second latent Input is not slot-distinct and collapses:

```openui
root = Stack([v0, v2, ":ood.auth.create", ":ood.auth.create"])
v0 = Input(":ood.auth.name", ":ood.auth.email", ":ood.auth.create")
v2 = Button(":ood.auth.create", ":ood.auth.create", ":ood.auth.create", ":ood.auth.create", ":ood.auth.create")
```

## Verdict

Retain count-aware scoring default-off as a positive structural diagnostic.
Do not promote or sync a checkpoint. The next experiment must assign distinct
visible slots to repeated predicted instances so canonicalization cannot
collapse them and strict semantic binding can improve.

Machine-readable evidence:
[iter-e581-count-aware-components-20260720.json](iter-e581-count-aware-components-20260720.json).
