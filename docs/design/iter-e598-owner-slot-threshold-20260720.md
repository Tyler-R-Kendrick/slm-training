# E598 — schema owner-slot threshold

Date: 2026-07-20
Status: semantic repair; not promotable or ship

E598 raises the existing schema property-owner slot score on E597's
schema-candidate treatment. Capped CPU OOD `n=4` arms at weights 4, 6, and 8
all completed within 170 seconds.

Weights 4 and 6 preserve E597's modal misbinding. Weight 8 changes the modal
from `Button(":ood.modal.body")` to `Button(":ood.modal.confirm")` and moves
TextContent to body, reducing reported placeholder semantic-role mismatches
from 3 to 2.

All three arms retain meaning-v1/v2 0.50/0.00, fidelity 0.5917, validity
0.7550, structure 0.4694, recall 0.6250, reward 0.8115, and AST node/edge F1
0.5532/0.3875. AgentV remains 0/1. Weight 8 is therefore a useful scratch
threshold only, not a default or promotion. No checkpoint was created,
promoted, or synced.

Evidence: [JSON](iter-e598-owner-slot-threshold-20260720.json).
