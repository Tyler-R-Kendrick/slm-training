# E668 — typed-array semantic-role wrapper

Date: 2026-07-21
Status: completed neutral scratch; rejected; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy, honest slot contracts, and no unconstrained fallback. This
was an evaluation-only scratch arm (`steps=0`, context backend `scratch`). It
emitted AgentEvals and AgentV with no timeout or fallback after all 116 compiler
tests passed.

E668 prefers, at the first item of an empty typed array, a component reference
that is both allowed by the active item schema and directly compatible with a
missing slot's semantic role. The focused invariant passes, but all four model
predictions are byte-for-byte identical to E667. Meaningful v1 (1.0000), strict
v2 (0.7500), fidelity/validity (1.0000), structure (0.7692), recall (0.8750),
reward (0.9730), and AST F1 (0.8222/0.7102) are unchanged. Dashboard still
emits the raw metric inside `Carousel.children`, so both strict failure reasons
remain.

Reject neutral v120 and restore retained E666 behavior as v121. The result
shows that the runtime item schema does not expose a usable allowed-reference
and semantic-role intersection at this decision. P95 was 7290.63 ms versus
E667's 7386.72 ms; the tiny run is not attributable performance evidence. No
checkpoint was created, synced, or promoted; AgentV remained 0/1.

Evidence: [JSON](iter-e668-typed-array-role-wrapper-20260721.json).
