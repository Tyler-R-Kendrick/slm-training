# E596 — visible slot-role aliases

Date: 2026-07-20
Status: neutral; not promotable or ship

E596 maps visible roles such as `body` to schema property `text` and action
roles such as `confirm/create/submit` to `label/action`. The matched CPU OOD
`n=4` run completed within 170 seconds.

E596 is prediction- and metric-identical to E595: meaning-v1/v2 0.50/0.00,
fidelity 0.5917, validity 0.7550, structure 0.4694, recall 0.6250, reward
0.8115, and AST node/edge F1 0.5532/0.3875. AgentV is 0/1.

The aliases are correct, but Button is opened while body is still the first
remaining slot; property-role scoring occurs too late to repair assignment.
Do not promote or sync. Next repair component-to-slot assignment order.

Evidence: [JSON](iter-e596-slot-role-alias-20260720.json).
