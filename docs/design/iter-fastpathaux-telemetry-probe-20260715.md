# Fastpath auxiliary telemetry probe — 2026-07-15

A one-step CPU training probe with `fastpath_aux_weight=1.0` now emits a dedicated `fastpath_aux_loss` telemetry span. The span was observed with count 1, total 1.748 ms, proving the structural auxiliary objective is active rather than silently swallowed.

The harness change removes the broad exception suppression around `force_align_loss`; failures now fail the run instead of producing an unevaluable apparent objective. This is instrumentation/correctness work, not a quality result.
