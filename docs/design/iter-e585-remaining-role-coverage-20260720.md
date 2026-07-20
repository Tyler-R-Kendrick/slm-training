# E585 — remaining-role coverage abstention

Date: 2026-07-20  
Status: negative; not promotable or ship

E585 gates the auxiliary learned slot-head only when every *remaining* visible
slot has a role-family candidate. Base scores and legal candidates remain
unchanged.

## Matched result

Both arms use clean commit `0ff8c653`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, plan/root
weight 4, constrained LTR, 8 steps, 4 attempts, and a 160-token canvas.
Each process completed under 170 seconds. Stamps carry eval v16 and TwoTower
v22.

| Role weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e585-e569-role-coverage0-r1` | 0.25 / 0.00 | 0.5083 / 0.7050 | 0.3119 | 0.4583 | 0.7760 | 0.4264 / 0.1667 | 0/1 |
| 4 | `e585-e569-role-coverage4-r1` | 0.25 / 0.00 | 0.3417 / 0.5050 | 0.3944 | 0.3750 | 0.5743 | 0.4889 / 0.2500 | 0/1 |

The treatment exactly reproduces E584: auth topology is perfect, modal is
`root = Stack([])`, reward falls by 0.2018, and strict meaning-v2 remains zero.
The remaining-only check can become true after an uncovered role has already
been consumed, so it is not a stable confidence condition.

## Verdict

Reject remaining-only coverage gating and keep it default-off. Do not promote
or sync a checkpoint. The next iteration must bind confidence to complete
coverage of the original visible slot contract.

Machine-readable evidence:
[iter-e585-remaining-role-coverage-20260720.json](iter-e585-remaining-role-coverage-20260720.json).
