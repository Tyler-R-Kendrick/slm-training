# E541 — root-only reference completeness

E541 restricts the default-off generated-reference completeness hook to v0.5
roots and terminal structural root lists. Nested component lists fail closed.

The exact E540 four-record OOD recipe completed in about 30 seconds under the
external 170-second cap from clean commit `909a43b`. It emitted AgentEvals
JSONL, a pinned AgentV bundle, and untruncated `choice_decision_trace/v2`
evidence without execution errors.

The guard behaves as predicted: nine root-list applications change six legal
choices, and no nested-list intervention occurs. Despite that causal activity,
every quality metric exactly matches the E539 weight-zero control. Relative to
E540, fidelity falls `0.4667→0.3833` and validity `0.58→0.53`, while structure
rises `0.1119→0.1159`, AST node F1 `0.1556→0.1627`, and AST edge F1
`0.0385→0.0417`. Meaningful rate, recall, reward, strict meaning, and AgentV
remain unchanged.

The one nested Modal choice removed by E541 therefore explains both E539's
placeholder gain and its structural regression. The six root changes have no
aggregate quality effect on this diagnostic.

**Verdict:** retain the safer root-only guard as a default-off diagnostic, but
reject weight 4 for training or promotion. Stop iterating on hand-written
reference completeness; the next lever must learn topology or aggregation
targets. Machine-readable evidence:
[JSON](iter-e541-root-reference-only-20260719.json).
