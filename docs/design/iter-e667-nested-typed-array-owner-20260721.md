# E667 — nested typed-array component ownership

Date: 2026-07-21
Status: completed neutral scratch; rejected; not ship

One capped CPU OOD `n=4` run reused E620's rejected local checkpoint with the
exact E653 policy, honest slot contracts, and no unconstrained fallback. This
was an evaluation-only scratch arm (`steps=0`, context backend `scratch`). It
emitted AgentEvals and AgentV with no timeout or fallback after all 116 compiler
tests passed.

E667 resolves typed-array item bias against the nearest enclosing component
rather than assuming the immediate parent frame is the owner. The focused
nested-array invariant passes, but all four model predictions are byte-for-byte
identical to E666. Meaningful v1 (1.0000), strict v2 (0.7500),
fidelity/validity (1.0000), structure (0.7692), recall (0.8750), reward
(0.9730), and AST F1 (0.8222/0.7102) are unchanged. Dashboard still emits the
raw metric inside `Carousel.children`, so both strict failure reasons remain.

Reject neutral v118 and restore retained E666 behavior as v119. P95 was 7386.72
ms versus E666's 8502.59 ms; this tiny unmatched timing difference is not an
attributable performance claim. No checkpoint was created, synced, or promoted;
AgentV remained 0/1.

Evidence: [JSON](iter-e667-nested-typed-array-owner-20260721.json).
