# E760-E762 — local sampling and marker uniqueness

**Date:** 2026-07-22
**Decision:** retain eval v44, model v220, and meaningful-v2 2.13.0; no checkpoint promotion
**Evidence:** [`iter-e760-e762-local-sampling-marker-uniqueness-20260722.json`](iter-e760-e762-local-sampling-marker-uniqueness-20260722.json)

E760 adds one canonical diagnostic selector, `eval_offset`, to the existing
evaluation harness. Offset is applied before limit and is persisted in the
scoreboard and cache identity. This makes non-prefix local sampling
reproducible without copying datasets or adding a parallel evaluator. An
offset-40 run against the 35-record snapshot correctly emitted `n=0` and
undefined quality metrics; it is retained as valid negative evidence.

Shifted slices found two accepted-path defects. At offset 29, E760 omitted the
standalone `:toolbar.text` marker because nested `TextContent` roles were bound
globally. E761 keeps nested roles under their reachable structural owner and
globally binds only direct obligations, restoring the six-record slice and the
original n=3 control to 1.0 across contract and structure metrics.

The version-matched v219 sweep then exposed `rico_eval_test_69`: all required
markers were present, but two appeared twice, reducing that record's structure
to 0.75 while meaningful-v2 incorrectly passed. E762 makes declared lexer
markers single-use under the existing slot-contract constraint and updates
meaningful-v2 2.13.0 to reject duplicate marker identities. The matched
offset-27 n=8 replay returns parse, fidelity, validity, structure, tree edit,
component recall, and strict-v2 to 1.0, with reward 0.9295, p50 8883.50 ms,
p95 10781.45 ms, zero timeouts, and zero fallbacks.

All runs were local and completed under the 110-second command cap. AgentV
remains 0/1 because these are partial scratch diagnostics, not ship evidence.
No checkpoint was created or synced. No free-form output strings were emitted.

Post-run harness hardening keeps those constraints structural. Positive
metric-gaming archetypes now declare markers for `SwitchItem.name` and
`TabItem.value`; the shared tokenizer-sidecar loader preserves lexer-native
identity through checkpoint migration; and V9/V10 inherit design-context
conditioning from one strict compiler-tree policy. These follow-on changes are
versioned as eval scoring v14, TwoTower v221, and quality matrix v5. They do not
retroactively change the v220 E762 run stamp.
