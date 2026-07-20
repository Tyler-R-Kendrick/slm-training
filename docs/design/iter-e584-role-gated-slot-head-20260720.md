# E584 — visible-role-gated slot head

Date: 2026-07-20  
Status: mixed negative; not promotable or ship

E584 prevents visible-role-mismatched families from receiving the auxiliary
learned slot-head bonus when an honest role candidate exists. Base decoder
scores and the legal candidate set remain unchanged; role weight zero is
behaviorally inactive.

## Matched result

Both arms use clean commit `5e6db51d`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, E583's honest
role candidates, plan/root weight 4, constrained LTR, 8 generation steps,
4 attempts, and a 160-token canvas. Each process completed under the
170-second cap. Stamps carry eval v16 and TwoTower v21.

| Role weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e584-e569-role-gate0-r1` | 0.25 / 0.00 | 0.5083 / 0.7050 | 0.3119 | 0.4583 | 0.7760 | 0.4264 / 0.1667 | 0/1 |
| 4 | `e584-e569-role-gate4-r1` | 0.25 / 0.00 | 0.3417 / 0.5050 | 0.3944 | 0.3750 | 0.5743 | 0.4889 / 0.2500 | 0/1 |

The targeted auth record reaches perfect AST node/edge/tree similarity and an
exact reference graph:

```openui
root = Stack([v0, v1, v2], ":ood.auth.create")
v0 = Input(":ood.auth.name")
v1 = Input(":ood.auth.email", ":ood.auth.create", ":ood.auth.create")
v2 = Button(":ood.auth.create", ":ood.auth.create", ":ood.auth.create", ":ood.auth.create", ":ood.auth.create")
```

Strict meaning-v2 still fails on schema-value roles and placeholder spam.
Worse, modal collapses to `root = Stack([])`. Aggregate reward falls by
0.2018, fidelity by 0.1667, validity by 0.20, and recall by 0.0833. The
structural and AST gains therefore do not justify the intervention.

## Verdict

Reject unconditional role gating and keep it default-off. Do not promote or
sync a checkpoint. The next experiment should use confidence-aware arbitration
between visible role evidence and the learned slot head, with abstention when
role coverage is incomplete.

Machine-readable evidence:
[iter-e584-role-gated-slot-head-20260720.json](iter-e584-role-gated-slot-head-20260720.json).
