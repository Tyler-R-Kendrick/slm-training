# E588 — plan-root closure strength

Date: 2026-07-20  
Status: positive diagnostic; not promotable or ship

E588 tests whether E587's aggressive schema-value correction failed because
the existing plan-root closure factor was too weak to terminate after a valid
predicted Stack. It changes no code or legal candidate set.

## Matched result

All arms use clean commit `43a114dd`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, semantic-role
and schema-value weights 4, constrained LTR, 8 steps, 4 attempts, and a
160-token canvas. Each process completed under 170 seconds. Stamps carry eval
v17, scoring v11, and TwoTower v24.

| Root weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `e588-e587-root4-control-r1` | 0.25 / 0.00 | 0.2583 / 0.5550 | 0.2156 | 0.3333 | 0.6920 | 0.3389 / 0.0000 | 0/1 |
| 8 | `e588-e587-root8-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.4069 | 0.4583 | 0.7585 | 0.4889 / 0.2500 | 0/1 |
| 12 | `e588-e587-root12-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.4069 | 0.4583 | 0.7585 | 0.4889 / 0.2500 | 0/1 |

Weights 8 and 12 are byte- and metric-equivalent. Auth closes as a Stack over
distinct name Input, email Input, and create Button nodes, uses the literal
`"column"` direction, and closes before the enum-valued Input and Button
properties. Relative to E586's schema-weight-0 recipe, structure improves
0.3819→0.4069 and reward 0.7510→0.7585 with fidelity, validity, recall, and AST
F1 unchanged.

## Verdict

Use root weight 8 as the next scratch diagnostic baseline; 12 adds no value.
Do not promote or sync a checkpoint because strict meaning-v2 remains zero,
AgentV fails, and the OOD subset has only four records. The next lever should
target the remaining non-enum placeholder spam and modal body/confirm binding
without weakening the recovered closure.

Machine-readable evidence:
[iter-e588-root-closure-strength-20260720.json](iter-e588-root-closure-strength-20260720.json).
