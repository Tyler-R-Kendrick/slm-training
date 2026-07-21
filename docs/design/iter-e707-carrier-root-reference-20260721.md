# E707 — carrier root-reference completeness

Date: 2026-07-21  
Status: completed five-suite mixed result; rejected; not ship

E707 follows E706's trace: the bounded carrier was created and filled, but the
terminal `Stack` omitted its reference and canonicalization pruned the orphan.
The treatment combines schema-value weight 5, one bounded missing-slot carrier,
and a true margin floor that includes each unused top-level element exactly once
in terminal root aggregation. All runs completed under the three-minute cap with
no decode timeout or unconstrained fallback and emitted AgentEvals plus AgentV.

The Rico result is useful: contract recall, fidelity, and validity rise to 1.0,
structure improves 0.7611→0.7915, node F1 0.7942→0.8419, and reward
0.9625→1.0. Strict remains 0.0 because the independent
`TextContent.size` semantic-role mismatch persists.

The complete five-suite replay rejects the treatment. Adversarial fidelity falls
0.8333→0.5833, validity 0.90→0.65, component recall 1.0→0.75, and reward
0.9035→0.6768. Restore v177 behavior as v179. The next lever must prevent the
enum-role misuse at value selection without globally forcing orphan references.
AgentV is 0/5 and Rico still exceeds the 160-token budget. This is scratch-matrix
evidence, not a ship evaluation; no checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e707-carrier-root-reference-20260721.json).
