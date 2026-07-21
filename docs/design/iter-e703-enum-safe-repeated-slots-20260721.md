# E703 — enum-safe repeated slots

Date: 2026-07-21  
Status: completed neutral; reverted; not ship

E703 made the repeated-instance visible-slot margin abstain at enum-valued
component properties, targeting the Rico `TextContent.size` mismatch without
the broad property gating that regressed earlier experiments.

The matched five-suite scratch replay evaluated all 19 committed records under
constrained-only decoding with no timeout or fallback and emitted AgentEvals
plus an AgentV SDK bundle. All 19 prediction hashes and every tracked quality
metric are identical to E701. Rico strict remains 0.0 and its fidelity,
structure, recall, and AST scores remain 0.9583, 0.7611, 1.0, and
0.7942/0.7757 respectively.

Reject v174 as dead complexity and restore v173 behavior as v175. The observed
role-mismatched slot is not introduced by this repeated-slot margin at an enum
frame, so the next arm must trace another scoring phase. AgentV is 0/5 and the
Rico length budget still fails. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e703-enum-safe-repeated-slots-20260721.json).
