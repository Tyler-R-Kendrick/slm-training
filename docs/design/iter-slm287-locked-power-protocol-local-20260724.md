# SLM-287: bounded local locked-power preflight (2026-07-24)

Machine-readable record: [JSON](iter-slm287-locked-power-protocol-local-20260724.json).

The first local CPU preflight used a zero-update TwoTower initialization, one immutable `locked_test` record, five declared seeds, and two scratch configurations. It ran the canonical raw/constrained/repaired evaluator under the repository's 170-second interrupt budget.

It was interrupted during the first constrained grammar-completion decode. No cell finished, no numeric metric or MDE was emitted, and this is not promotion or ship evidence. The protocol now sets a per-record timeout and rejects any timed-out cell rather than aggregating an empty decode. Human ratings were not used and are not a gate.

The full protocol remains fixed at 226 locked records and only emits a result after all ten cells complete with bit-identical repeated initialization and paired raw/constrained/repaired evidence.
