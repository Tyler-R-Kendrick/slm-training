# E587 — schema-value role bias

Date: 2026-07-20  
Status: mixed negative; not promotable or ship

E587 adds a default-off score penalty for visible slot pointers when the active
component argument is enum-valued in the canonical schema. Candidate legality,
required content arguments, and the base decoder remain unchanged.

## Matched result

All arms use clean commit `a7aaf8da`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, E586 role weight
4, plan/root weight 4, constrained LTR, 8 steps, 4 attempts, and a 160-token
canvas. Each process completed under 170 seconds. Stamps carry eval v17,
scoring v11, and TwoTower v24.

| Schema weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e587-e586-schema-value0-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.3819 | 0.4583 | 0.7510 | 0.4889 / 0.2500 | 0/1 |
| 1 | `e587-e586-schema-value1-r1` | 0.25 / 0.00 | 0.4667 / 0.6800 | 0.3469 | 0.3958 | 0.7635 | 0.4056 / 0.2500 | 0/1 |
| 4 | `e587-e586-schema-value4-r1` | 0.25 / 0.00 | 0.2583 / 0.5550 | 0.2156 | 0.3333 | 0.6920 | 0.3389 / 0.0000 | 0/1 |

Weight 1 replaces the incorrect Stack direction placeholder with `"column"`
and raises fidelity, validity, and reward. It does not remove the nested Input
and Button enum-property spam, changes gallery from TextContent to Image, and
loses structure, component recall, and AST-node F1. Weight 4 overdrives the
factor: auth collapses to a single Input and the aggregate metrics regress
sharply.

## Verdict

Keep the generalized schema-role signal and its regression test, but leave the
weight default-off. Do not promote or sync a checkpoint. A follow-up must
separate required content-slot consumption from optional schema-value choices
before applying stronger enum correction.

Machine-readable evidence:
[iter-e587-schema-value-role-bias-20260720.json](iter-e587-schema-value-role-bias-20260720.json).
