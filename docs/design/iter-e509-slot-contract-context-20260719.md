# E509 — honest slot contract in context

E509 asks whether exposing the honest request slot contract to the model fixes
the semantic failures that remain under E508's constrained decode. It evaluates
the same rejected E505 checkpoint on all four OOD records with the same
length-safe 160-token canvas, default eight generation steps, four attempts,
and no DESIGN context.

The evaluation completed under its external 170-second cap and emitted
AgentEvals plus pinned AgentV evidence without execution errors.

| Policy | n | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | v2 coverage | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E508 decode only | 4 | 1.0 | 0.25 | 0.2583 | 0.2281 | 0.3333 | 0.692 | 0.3389 | 0.75 | 0/1 |
| E509 context + decode | 4 | 1.0 | 0.25 | 0.2583 | 0.2406 | 0.3333 | 0.692 | 0.3389 | 1.0 | 0/1 |

Contract context adds `0.0125` structure and makes binding-aware coverage known
for every record. It does not improve meaningfulness, fidelity, component
recall, reward, AST F1, strict binding-aware meaningfulness, or AgentV.
Failures remain concentrated in missing required placeholders/components and
placeholder or schema semantic-role mismatches.

## Decision

Do not promote. Inventory visibility is not the remaining semantic blocker.
Target component selection and placeholder semantic-role mapping next. No
checkpoint was created or synced.

Exact metrics:
[machine-readable record](iter-e509-slot-contract-context-20260719.json).
