# E655 — direct role ownership before raw slots

Date: 2026-07-21
Status: completed neutral; reverted; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy and no unconstrained fallback. It emitted AgentEvals and
AgentV with no timeout or fallback.

E655 required the active component to directly own a semantic role before the
coverage floor could force its raw placeholder; transitive owners instead
preferred a compatible child. The invariant passed the full 115-test compiler
suite, but the matched run produced E653's exact prediction set and all the
same quality metrics: meaningful v1 1.0000, strict v2 0.7500, fidelity and
validity 1.0000, structure 0.7692, reward 0.9730, node F1 0.8222, and edge F1
0.7102. The `Callout.variant` and `Carousel.children` failures remained. Reject
neutral v107 and restore E653 behavior as v108; the remaining fault is earlier
than this raw-slot selection branch. No checkpoint was created, synced, or
promoted; AgentV remained 0/1.

Evidence: [JSON](iter-e655-direct-role-slot-ownership-20260721.json).
