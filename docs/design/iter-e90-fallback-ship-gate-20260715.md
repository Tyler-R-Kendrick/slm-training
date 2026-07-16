# E90 certified-fallback ship gate — 2026-07-15

E90 extends the fallback telemetry correction into `evaluate_ship_gates`.
Every scored suite now requires `fallback_count == 0` for a learned-quality
claim. A certified template can remain useful as a serving safety path, but it
cannot make a checkpoint pass learned-model ship gates.

Focused gate/evaluator tests passed: 20. The E82-style result would now fail
the added `suite:certified_fallback` check despite its perfect structural and
placeholder metrics.

Decision: retain the gate. This is a policy/harness correction, not a model
result.
