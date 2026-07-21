# E669 — nested array schema propagation

Date: 2026-07-21
Status: completed negative scratch; rejected; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy, honest slot contracts, and no unconstrained fallback. This
was an evaluation-only scratch arm (`steps=0`, context backend `scratch`). It
emitted AgentEvals and AgentV with no timeout or fallback after all 116 compiler
tests passed.

E669 propagates an active array item schema when opening a nested list, rather
than retaining schemas only for arrays directly under components. The change
reaches the target: Dashboard's invalid raw Carousel child becomes the
schema-valid `Form(":ood.dash.m1.value", Buttons([]), [])`, and the
`Carousel.children` schema mismatch disappears. The selected wrapper is
semantically wrong, however. Strict v2 remains 0.7500, structure falls
0.7692→0.7230, node/edge F1 fall 0.8222/0.7102→0.7987/0.6845, and p95 rises
7386.72→9376.16 ms. Meaningful v1, fidelity, validity, recall, and reward hold.

Reject v122 and restore retained E666 behavior as v123. A future retry must
combine preserved inner schemas with role-aware selection, rather than relying
on the model's highest-scoring schema-valid component. No checkpoint was
created, synced, or promoted; AgentV remained 0/1.

Evidence: [JSON](iter-e669-nested-array-schema-20260721.json).
