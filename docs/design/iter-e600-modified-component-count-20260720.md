# E600 — modified component-count parsing

Date: 2026-07-20
Status: planner fix; decode-neutral at the retained weight, not promotable or ship

E600 fixes prompt-derived semantic-plan cardinality when one descriptive word
separates an explicit count from a component family. For example, “two metric
cards” now produces two required `Card` roles instead of one. The rule uses
only authored prompt text and does not inspect gold programs or placeholders.

Two capped CPU OOD `n=4` evaluations completed within 170 seconds. At the
retained semantic-plan weight 4, the corrected plan is prediction- and
metric-identical to E599: syntax 1.0, meaningful-v1 0.5, strict meaning-v2 0,
fidelity 0.5917, validity 0.7550, structure 0.5169, component recall 0.6250,
reward 0.8115, AST-node F1 0.5754, and AST-edge F1 0.4143.

A weight-8 threshold treatment raises structure to 0.5756, AST-node F1 to
0.6111, and AST-edge F1 to 0.4643, but neither collapsed dashboard/gallery
case changes. The apparent gain comes from adding a duplicate `Input` to the
auth case, introducing placeholder spam and an `Input.type` role mismatch.
Therefore weight 8 is rejected and weight 4 remains the retained setting.

Strict meaning-v2 remains 0 and AgentV is 0/1 in both arms. No checkpoint was
created, promoted, or synced. The next useful lever must affect initial
component-family selection rather than increase global plan pressure.

Evidence: [JSON](iter-e600-modified-component-count-20260720.json).
