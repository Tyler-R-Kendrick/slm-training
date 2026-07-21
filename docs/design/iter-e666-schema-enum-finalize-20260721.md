# E666 — post-decode schema-enum normalization

Date: 2026-07-21
Status: completed positive scratch; retained; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy, honest slot contracts, and no unconstrained fallback. It
emitted AgentEvals and AgentV with no timeout or fallback after all 115 compiler
tests passed.

E666 replays the completed choice stream and replaces only a framed dynamic
literal at an enum-valued schema property. The replacement happens after model
decoding, so later choices are unchanged. Dashboard's invalid
`Callout("itet", ...)` becomes `Callout("info", ...)`; the other three outputs
are byte-for-byte unchanged. The `Callout.variant` strict failure disappears,
while meaningful v1 (1.0000), strict v2 (0.7500), fidelity/validity (1.0000),
structure (0.7692), recall (0.8750), reward (0.9730), and AST F1
(0.8222/0.7102) all match E653. Dashboard still fails strict ownership for
`Carousel.children`, so this is not ship evidence. P95 was 8502.59 ms versus
E653's 7045.22 ms; this tiny run does not establish a performance regression or
claim. Retain v117 as a generalized schema-validity correction. No checkpoint
was created, synced, or promoted; AgentV remained 0/1.

Evidence: [JSON](iter-e666-schema-enum-finalize-20260721.json).
