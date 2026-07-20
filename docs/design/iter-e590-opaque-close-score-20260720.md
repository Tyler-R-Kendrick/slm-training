# E590 — optional opaque-argument close score

Date: 2026-07-20  
Status: behaviorally positive, metric-neutral; not promotable or ship

E590 replaces E589's failed alternative suppression with a default-off,
legality-preserving positive score on the already-legal close token at an
optional unconstrained (`{}`) component argument. No candidate is removed.

## Matched result

All arms use E588's root-weight-8 recipe, clean E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, constrained LTR,
8 steps, 4 attempts, and a 160-token canvas. Each process completed under 170
seconds. Stamps carry eval v19, scoring v11, and TwoTower v26.

| Close weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e590-e589-close0-control-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.4069 | 0.4583 | 0.7585 | 0.4889 / 0.2500 | 0/1 |
| 2 | `e590-e589-close2-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.4069 | 0.4583 | 0.7585 | 0.4889 / 0.2500 | 0/1 |
| 4 | `e590-e589-close4-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.4069 | 0.4583 | 0.7585 | 0.4889 / 0.2500 | 0/1 |
| 8 | `e590-e589-close8-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.4069 | 0.4583 | 0.7585 | 0.4889 / 0.2500 | 0/1 |

Weights 0 and 2 retain
`Button(":ood.auth.create", ":ood.auth.create")`. Weights 4 and 8 are
byte-identical and correctly emit `Button(":ood.auth.create")`, without E589's
nested-list diversion. Current aggregate metrics do not score this optional
action-field correction, so the result is not an authoritative quality gain.

## Verdict

Weight 4 is the smallest demonstrated behavioral threshold and may be used in
the next scratch diagnostic. Keep the lever default-off. Do not promote or sync
a checkpoint because aggregate quality is flat, strict meaning-v2 remains
zero, AgentV fails, and the subset has only four records.

Machine-readable evidence:
[iter-e590-opaque-close-score-20260720.json](iter-e590-opaque-close-score-20260720.json).
