# E586 — original-contract role coverage

Date: 2026-07-20  
Status: mixed negative; not promotable or ship

E586 gates the auxiliary learned slot-head only when every slot in the
*original* visible contract has a role-family candidate. Base scores and legal
candidates remain unchanged.

## Matched result

Both arms use clean commit `b4faabf7`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, plan/root
weight 4, constrained LTR, 8 steps, 4 attempts, and a 160-token canvas.
Each process completed under 170 seconds. Stamps carry eval v16 and TwoTower
v23.

| Role weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e586-e569-original-coverage0-r1` | 0.25 / 0.00 | 0.5083 / 0.7050 | 0.3119 | 0.4583 | 0.7760 | 0.4264 / 0.1667 | 0/1 |
| 4 | `e586-e569-original-coverage4-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.3819 | 0.4583 | 0.7510 | 0.4889 / 0.2500 | 0/1 |

The stable coverage condition prevents E585's empty modal Stack. Auth retains
perfect AST node/edge/tree similarity with distinct name Input, email Input,
and create Button nodes. Aggregate structure improves 0.0700, AST-node F1
0.0625, and AST-edge F1 0.0833 without losing component recall.

The treatment is still not a quality win: fidelity falls 0.0833, validity
0.0500, and reward 0.0250. The modal chooses confirm instead of body, while
auth repeats the create placeholder in optional enum-like properties. Strict
meaning-v2 remains zero and AgentV fails.

## Verdict

Retain original-contract coverage as the safer confidence boundary, but reject
role weight 4 for promotion and keep it default-off. Do not create or sync a
checkpoint. The next iteration should use schema property roles to prevent
visible placeholders from filling enum-like optional properties.

Machine-readable evidence:
[iter-e586-original-role-coverage-20260720.json](iter-e586-original-role-coverage-20260720.json).
