# E539 — structural reference aggregation

E539 extends the default-off generated-reference completeness hook to the
choice codec's reachable structural lists. The hook uses only generated
top-level element types and legal references, fails closed outside a list/root
frame, and has no access to the gold reference graph.

Both CPU OOD four-record runs completed in about 29 seconds each under the
external 170-second cap from clean commit `af94c1f`. They emitted AgentEvals
JSONL, pinned AgentV bundles, and `choice_decision_trace/v1` evidence without
execution errors.

E536 is not a valid numeric control because it used evaluator/codec v9. E539
therefore includes a same-commit v10 weight-zero control.

| Metric | E539 control, weight 0 | E539, weight 4 | Delta |
| --- | ---: | ---: | ---: |
| Syntax parse rate | 1.0000 | 1.0000 | 0.0000 |
| Meaningful program rate | 0.0000 | 0.0000 | 0.0000 |
| Placeholder fidelity | 0.3833 | 0.4667 | +0.0833 |
| Placeholder validity | 0.5300 | 0.5800 | +0.0500 |
| Structural similarity | 0.1159 | 0.1119 | -0.0040 |
| Component type recall | 0.2292 | 0.2292 | 0.0000 |
| Reward | 0.3685 | 0.3685 | 0.0000 |
| AST node F1 | 0.1627 | 0.1556 | -0.0071 |
| AST edge F1 | 0.0417 | 0.0385 | -0.0032 |
| Strict binding-aware meaning | 0.0000 | 0.0000 | 0.0000 |
| AgentV | 0 / 1 | 0 / 1 | unchanged |

The hook is causally reachable: 10 applications changed seven legal choices.
It improves placeholder coverage, including inserting `v0` into the modal body,
but unconditional list-level preference also selects references in early nested
lists. That does not improve meaningfulness, recall, reward, strict semantics,
or AgentV and slightly reduces structural overlap.

Retain the fail-closed implementation as a default-off diagnostic capability,
but reject weight 4 for training and promotion. The next topology intervention
must learn or explicitly identify an aggregation phase instead of treating
every list as the terminal root. Machine-readable evidence:
[control](iter-e539-structural-reference-control-20260719.json) and
[intervention](iter-e539-structural-reference-aggregation-20260719.json).
