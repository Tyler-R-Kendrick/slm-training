# E589 — optional opaque-argument slot penalty

Date: 2026-07-20  
Status: negative result; not promotable or ship

E589 tests a default-off, legality-preserving score that penalizes visible
placeholder pointers only in optional component arguments whose pinned schema
is unconstrained (`{}`). Required and user-visible content arguments, enum
arguments, and the legal candidate set are unchanged.

## Matched result

All arms use E588's root-weight-8 recipe, clean E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, constrained LTR,
8 steps, 4 attempts, and a 160-token canvas. Each process completed under 170
seconds. Stamps carry eval v18, scoring v11, and TwoTower v25.

| Opaque weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e589-e588-opaque0-control-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.4069 | 0.4583 | 0.7585 | 0.4889 / 0.2500 | 0/1 |
| 4 | `e589-e588-opaque4-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.3319 | 0.4583 | 0.7585 | 0.4611 / 0.2143 | 0/1 |
| 8 | `e589-e588-opaque8-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.3319 | 0.4583 | 0.7585 | 0.4611 / 0.2143 | 0/1 |

Weights 4 and 8 are byte- and metric-equivalent. The treatment does not make
the Button close after its label. Because `{}` accepts any expression,
suppressing the slot pointer diverts generation into a deeply nested legal
list containing the same placeholder. Structure falls by 0.075, AST-node F1
by 0.0278, and AST-edge F1 by 0.0357; no authoritative quality metric improves.

## Verdict

Keep the new lever default-off and preserve the negative result. Do not promote
or sync a checkpoint. The next scratch lever should directly reward the legal
component-close token for optional unconstrained arguments instead of
penalizing only one alternative expression class.

Machine-readable evidence:
[iter-e589-opaque-slot-penalty-20260720.json](iter-e589-opaque-slot-penalty-20260720.json).
