# E85 template-path telemetry — 2026-07-15

E85 validates the request-aware decode telemetry added after E84. The E82
checkpoint was evaluated with an explicit slot contract and the opt-in
template fast path.

The smoke result was parse/fidelity 1.0/1.0 with 2,554.61 ms p50 latency, but
the important evidence is `template_fastpath_count_sum=1` and
`template_fallback_count_sum=0`. Request-aware `DecodeStats` are now collected
and persisted for the same path.

Decision: retain the telemetry. This confirms the E82 quality result is a
certified template result, not learned generation, and keeps those claims
separate in future scoreboards.

This is a harness validation, not a ship claim.
