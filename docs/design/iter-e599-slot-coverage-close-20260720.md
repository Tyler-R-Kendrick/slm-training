# E599 — visible-slot coverage close score

Date: 2026-07-20
Status: structurally positive; default-off, not promotable or ship

E599 scores legal closure of typed component arrays after all visible slots
have been covered. It abstains on root structural lists and incomplete
coverage. Capped CPU OOD `n=4` treatments at weights 2 and 4 completed within
170 seconds and are prediction-identical.

Against E598 weight 8, both treatments remove the duplicate Modal body child
and leaked size argument while preserving `Button(confirm)` and
`TextContent(body)`. Structure improves 0.4694→0.5169, AST-node F1
0.5532→0.5754, and AST-edge F1 0.3875→0.4143. Fidelity 0.5917, validity
0.7550, recall 0.6250, and reward 0.8115 remain unchanged.

Weight 2 is the minimal tested effective setting. Keep the lever default-off
and use 2 as the next scratch treatment. Strict meaning-v2 remains 0 and
AgentV is 0/1, so no checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e599-slot-coverage-close-20260720.json).
