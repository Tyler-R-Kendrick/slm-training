# SLM-287: bounded local locked-power preflight (2026-07-25)

> **Historical policy artifact.** This preflight retains the emitted
> `raw`/`constrained`/`repaired` labels that were current when it ran. Mandatory
> constrained generation supersedes those live arms with
> `constrained_native` and `constrained_compiler`; the new campaign digest has
> no measured result yet.

Machine-readable record: [JSON](iter-slm287-locked-power-protocol-local-20260724.json).

The second local CPU preflight used a zero-update TwoTower initialization, one immutable `locked_test` record, five declared seeds, and two scratch configurations. It ran the canonical raw/constrained/repaired evaluator under the repository's 170-second interrupt budget.

The constrained decode reached the configured five-second per-record deadline and the runner emitted `interrupted_not_evidence` rather than waiting for the outer kill or aggregating partial values. No cell finished, no numeric metric or MDE was emitted, and this is not promotion or ship evidence. Human ratings were not used and are not a gate.

The full protocol remains fixed at 226 locked records and only emits a result after all ten cells complete with bit-identical repeated initialization and paired raw/constrained/repaired evidence.

After the timeout repair, one balanced local shard (`seed=0`,
`scratch_design_off`, `shard=0/64`, four locked records) completed on CPU in
the capped process. It produced raw/constrained/repaired rows, cell-scoped
compute/memory telemetry, and an AgentV bundle. The rows were all zero on
meaning-v2 and binder/reference F1; this is partial execution evidence only,
not a grid result, MDE, promotion, or ship claim. The remaining exact shards
must merge fail-closed before any aggregate is emitted.
