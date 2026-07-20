# E609 — evaluation canvas contract

Date: 2026-07-20
Status: completed and rejected for promotion

E609 repairs a harness contract mismatch: evaluation reported
`grammar_ltr_max_tokens=160`, but request-aware generation silently used the
checkpoint's learned `gen_len=58`. The canonical evaluator now passes its
reported canvas cap into generation. Plugins that do not accept `max_len`
retain the existing compatibility fallback.

The capped, matched OOD `n=4` replay completes normally. Dashboard now finishes
the verifier-approved root as `Stack([v0, v1, v2, v3])`, retaining Button,
Callout, and both planned Card bindings. That confirms E608's missing Card
references were a generation-budget artifact rather than a root-verifier or
reference-legality failure.

The longer canvas is not a quality lever. Dashboard consumes the added budget
with deeply nested duplicate Card content, reaches 163 output symbols, scores
zero reward, and raises p95 latency from 14.43 s to 29.61 s. Aggregate
meaningful-v1 falls from 0.75 to 0.50 and reward from 0.6788 to 0.4835. Strict
meaning-v2 remains zero. Fidelity, validity, and component recall improve, but
the preregistered no-regression constraint fails.

Retain the harness correction because evidence must report the canvas actually
used. Reject a 160-token canvas as a promotable decode policy for this
checkpoint. The next iteration should constrain component-local content
cardinality and close schema-valid planned components before constructing the
root, rather than buying completeness with unbounded generation.

No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e609-eval-canvas-contract-20260720.json).
