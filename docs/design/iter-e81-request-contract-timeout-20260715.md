# E81 request-contract decode timeout — 2026-07-15

E81 evaluated the E80 checkpoint through the corrected
`GenerationRequest` path with an explicit four-slot contract and a 20-second
per-record timeout.

The request-aware path emitted complete telemetry but timed out: parse,
syntax, fidelity, structure, and reward were all 0.0; latency p50/p95 was
20,001.8 ms; `decode_timeout_count` was 1. AgentEvals was persisted with 5
failed metric instances and no execution errors.

Decision: reject the model result and retain the evaluator fix. The next
harness iteration must reduce contract-constrained decode latency or expose a
bounded fast path; increasing the timeout would only hide the failure.

This is a bounded scratch diagnostic, not a ship claim.
