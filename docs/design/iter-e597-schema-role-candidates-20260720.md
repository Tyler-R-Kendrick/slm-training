# E597 — schema-derived semantic-role candidates

Date: 2026-07-20
Status: neutral; default-off, not promotable or ship

E597 tested whether stronger existing role weights or public-schema component
candidates could repair E595's Button-to-body misbinding. Four capped CPU OOD
`n=4` arms completed within 170 seconds: existing role weights 8 and 12, then
schema-derived candidates at weights 4 and 8.

Weights 8 and 12 without schema candidates are prediction- and metric-identical
to E596. Enabling schema candidates changes the dashboard root from an
overgeneralized Button to TextContent, but weights 4 and 8 are again identical
and the modal still opens `Button(":ood.modal.body")`.

Every arm reports meaning-v1/v2 0.50/0.00, fidelity 0.5917, validity 0.7550,
structure 0.4694, recall 0.6250, reward 0.8115, and AST node/edge F1
0.5532/0.3875. AgentV is 0/1 for every arm. The public-schema switch is
therefore preserved default-off. No checkpoint was created, promoted, or
synced. The next repair must operate on component-to-slot assignment rather
than candidate coverage or scalar weight.

Evidence: [JSON](iter-e597-schema-role-candidates-20260720.json).
