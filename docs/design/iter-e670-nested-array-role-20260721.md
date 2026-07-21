# E670 — nested array role-aware wrapper

Date: 2026-07-21
Status: completed negative scratch; rejected; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy, honest slot contracts, and no unconstrained fallback. This
was an evaluation-only scratch arm (`steps=0`, context backend `scratch`). It
emitted AgentEvals and AgentV with no timeout or fallback after all 117 compiler
tests passed.

E670 combines E669's nested item-schema propagation with E668's preference for
a schema-allowed semantic-role wrapper. The combination still emits
`Form(":ood.dash.m1.value", Buttons([]), [])` inside the Carousel. Strict v2
remains 0.7500, structure remains 0.7230, and node/edge F1 remain
0.7987/0.6845. Meaningful v1, fidelity, validity, recall, and reward hold. The
p95 change from 9376.16 to 9156.36 ms is not attributable on this small run.

Reject v124 and restore retained E666 behavior as v125. The next retry needs to
diagnose why the role-compatible candidate is not selected on the real decoder
state before changing another score. No checkpoint was created, synced, or
promoted; AgentV remained 0/1.

Evidence: [JSON](iter-e670-nested-array-role-20260721.json).
